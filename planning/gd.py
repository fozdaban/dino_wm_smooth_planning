import torch
import numpy as np
from einops import rearrange
from .base_planner import BasePlanner
from utils import move_to_device


class GDPlanner(BasePlanner):
    def __init__(
        self,
        horizon,
        action_noise,
        sample_type,
        lr,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
        )
        self.horizon = horizon
        self.action_noise = action_noise
        self.sample_type = sample_type
        self.lr = lr
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        # episodes are independent, so backprop them in chunks to bound GPU memory;
        # the summed gradient is identical to one big batch
        self.episode_chunk = kwargs.get("episode_chunk", 10)
        # upstream stops planning once the simulator reports every episode solved,
        # which uses information a real planner doesn't have; false = fixed budget
        self.oracle_stop = kwargs.get("oracle_stop", True)
        # ---- phase 3 options (defaults reproduce upstream GD exactly) ----
        # optimizer: "sgd" = upstream update a <- a - lr * grad (+ noise);
        #            "adam" = torch.optim.Adam step (+ the same noise)
        self.optimizer_name = kwargs.get("optimizer", "sgd")
        # restarts: R independent initial plans per episode, optimized together;
        # the returned plan is the one with the lowest MODEL cost (never the simulator)
        self.restarts = int(kwargs.get("restarts", 1))
        # init_scale: std of the "randn" initialization
        self.init_scale = float(kwargs.get("init_scale", 1.0))
        assert self.optimizer_name in ("sgd", "adam"), self.optimizer_name
        assert self.restarts >= 1
        # phase 6b (E3): also save every restart's final plan, (n, R, H, A), to
        # restart_actions.pt in the run directory; does not change the returned plan
        self.save_restarts = bool(kwargs.get("save_restarts", False))

    def init_actions(self, obs_0, actions=None, n_rows=None):
        """
        Initializes or appends actions for planning, ensuring the output shape is (b, self.horizon, action_dim).
        n_rows: number of plans to initialize (n_evals * restarts); defaults to n_evals.
        """
        n_evals = obs_0["visual"].shape[0] if n_rows is None else n_rows
        if actions is None:
            actions = torch.zeros(n_evals, 0, self.action_dim)
        device = actions.device
        t = actions.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            if self.sample_type == "randn":
                new_actions = self.init_scale * torch.randn(
                    n_evals, remaining_t, self.action_dim
                )
            elif self.sample_type == "normzero":  # 0 in normalized units = CEM's initial mean
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
            elif self.sample_type == "zero":  # zero action of env
                new_actions = torch.zeros(n_evals, remaining_t, self.action_dim)
                new_actions = rearrange(
                    new_actions, "... (f d) -> ... f d", f=self.evaluator.frameskip
                )
                new_actions = self.preprocessor.normalize_actions(new_actions)
                new_actions = rearrange(new_actions, "... f d -> ... (f d)")
            actions = torch.cat([actions, new_actions.to(device)], dim=1)
        return actions

    def get_action_optimizer(self, actions):
        if self.optimizer_name == "adam":
            return torch.optim.Adam([actions], lr=self.lr)
        return torch.optim.SGD([actions], lr=self.lr)

    def _repeat(self, x):
        """(n, ...) -> (n * R, ...), each episode's rows contiguous."""
        return x.repeat_interleave(self.restarts, dim=0) if self.restarts > 1 else x

    def _select(self, actions, costs, n_evals):
        """Pick, per episode, the restart with the lowest model cost."""
        if self.restarts == 1:
            return actions
        R = self.restarts
        best = costs.view(n_evals, R).argmin(dim=1)
        return actions.view(n_evals, R, *actions.shape[1:])[torch.arange(n_evals, device=actions.device), best]

    def _model_costs(self, actions, trans_obs_0, z_obs_g):
        """No-grad model cost of every row (counted as planner compute)."""
        costs = []
        with torch.no_grad():
            for s in range(0, actions.shape[0], self.episode_chunk):
                e = s + self.episode_chunk
                i_z_obses, _ = self.wm.rollout(
                    obs_0={k: v[s:e] for k, v in trans_obs_0.items()},
                    act=actions[s:e],
                )
                costs.append(
                    self.objective_fn(i_z_obses, {k: v[s:e] for k, v in z_obs_g.items()})
                )
        return torch.cat(costs)

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        z_obs_g_detached = {key: value.detach() for key, value in z_obs_g.items()}

        n_evals = obs_0["visual"].shape[0]
        if self.restarts > 1:
            # each episode gets R rows (contiguous); a given initial plan seeds every restart
            if actions is not None:
                actions = self._repeat(actions)
            trans_obs_0 = {k: self._repeat(v) for k, v in trans_obs_0.items()}
            z_obs_g_detached = {k: self._repeat(v) for k, v in z_obs_g_detached.items()}
        actions = self.init_actions(
            obs_0, actions, n_rows=n_evals * self.restarts
        ).to(self.device)
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        n_rows = actions.shape[0]
        last_costs = None

        for i in range(self.opt_steps):
            optimizer.zero_grad()
            total_loss = 0.0
            row_costs = []
            for s in range(0, n_rows, self.episode_chunk):
                e = s + self.episode_chunk
                i_z_obses, _ = self.wm.rollout(
                    obs_0={k: v[s:e] for k, v in trans_obs_0.items()},
                    act=actions[s:e],
                )
                loss = self.objective_fn(
                    i_z_obses, {k: v[s:e] for k, v in z_obs_g_detached.items()}
                )  # (chunk, )
                loss.sum().backward()  # grads accumulate into actions.grad
                total_loss += loss.sum().item()
                row_costs.append(loss.detach())
            last_costs = torch.cat(row_costs)  # cost of the plans *before* this update
            if self.optimizer_name == "adam":
                optimizer.step()
                with torch.no_grad():
                    actions.add_(torch.randn_like(actions) * self.action_noise)
            else:  # upstream update, unchanged
                with torch.no_grad():
                    actions_new = actions - optimizer.param_groups[0]["lr"] * actions.grad
                    actions_new += (
                        torch.randn_like(actions_new) * self.action_noise
                    )  # Add Gaussian noise
                    actions.copy_(actions_new)

            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss, "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                # with restarts, intermediate evaluations show the restart that had the
                # lowest cost at this step's forward pass (logging only, no extra compute)
                logs, successes, _, _ = self.evaluator.eval_actions(
                    self._select(actions.detach(), last_costs, n_evals),
                    filename=f"{self.logging_prefix}_output_{i+1}",
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if self.oracle_stop and np.all(successes):
                    break  # terminate planning if all success
        if self.restarts > 1:
            final_costs = self._model_costs(actions.detach(), trans_obs_0, z_obs_g_detached)
            if self.save_restarts:  # phase 6b (E3)
                torch.save(
                    actions.detach().cpu().view(n_evals, self.restarts, *actions.shape[1:]),
                    "restart_actions.pt",
                )
            actions = self._select(actions.detach(), final_costs, n_evals)
        return actions, np.full(n_evals, np.inf)  # all actions are valid
