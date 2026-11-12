import torch
import numpy as np
from .gd import GDPlanner
from utils import move_to_device


class SmoothGDPlanner(GDPlanner):
    """GD on a Gaussian-smoothed objective.

    Each step, every plan a is perturbed K times, a + sigma * eps_k, and the update uses
    the mean gradient of the model cost over the K perturbed copies (a Monte-Carlo
    estimate of grad E_eps[J(a + sigma eps)]). sigma follows a schedule over the steps.
    The optimizer update, action_noise, evaluation and oracle_stop are those of
    GDPlanner; the plan that is evaluated and returned is always the unperturbed center.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # perturbations per plan per step
        self.K = int(kwargs.get("K", 8))
        self.sigma0 = float(kwargs.get("sigma0", 0.3))
        self.sigma_min = float(kwargs.get("sigma_min", 0.03))
        # const | linear | geom (sigma0 at the first step, sigma_min at the last)
        self.sigma_schedule = kwargs.get("sigma_schedule", "geom")
        # antithetic: perturbations come in +eps / -eps pairs
        self.antithetic = bool(kwargs.get("antithetic", True))
        assert self.K >= 1, self.K
        assert self.sigma_schedule in ("const", "linear", "geom"), self.sigma_schedule
        if self.antithetic:
            assert self.K % 2 == 0, f"antithetic needs an even K, got {self.K}"
        if self.sigma_schedule == "geom":
            assert self.sigma0 > 0 and self.sigma_min > 0, "geom schedule needs sigma0, sigma_min > 0"
        assert self.restarts == 1, "smoothing + restarts is out of scope; set restarts=1"

    def sigma_at(self, i):
        """Noise scale at step i of S = opt_steps."""
        S = self.opt_steps
        if self.sigma_schedule == "const" or S == 1:
            return self.sigma0
        t = i / (S - 1)
        if self.sigma_schedule == "linear":
            return self.sigma0 + (self.sigma_min - self.sigma0) * t
        return self.sigma0 * (self.sigma_min / self.sigma0) ** t  # geom

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
        K = self.K
        # each episode gets K rows (contiguous), one per perturbation
        trans_obs_0 = {k: v.repeat_interleave(K, 0) for k, v in trans_obs_0.items()}
        z_obs_g_detached = {k: v.repeat_interleave(K, 0) for k, v in z_obs_g_detached.items()}
        actions = self.init_actions(obs_0, actions).to(self.device)
        actions.requires_grad = True
        optimizer = self.get_action_optimizer(actions)
        n_rows = n_evals * K
        H, A = actions.shape[1], actions.shape[2]

        for i in range(self.opt_steps):
            sigma = self.sigma_at(i)
            eps = None
            if sigma > 0:
                if self.antithetic:
                    half = torch.randn(n_evals, K // 2, H, A, device=self.device)
                    eps = torch.cat([half, -half], dim=1).reshape(n_rows, H, A)
                else:
                    eps = torch.randn(n_rows, H, A, device=self.device)
            optimizer.zero_grad()
            total_loss = 0.0
            for s in range(0, n_rows, self.episode_chunk):
                e = s + self.episode_chunk
                # rebuilt per chunk: each backward frees the graph it went through
                a_rows = actions.repeat_interleave(K, 0)[s:e]
                if eps is not None:
                    a_rows = a_rows + sigma * eps[s:e]
                i_z_obses, _ = self.wm.rollout(
                    obs_0={k: v[s:e] for k, v in trans_obs_0.items()},
                    act=a_rows,
                )
                loss = self.objective_fn(
                    i_z_obses, {k: v[s:e] for k, v in z_obs_g_detached.items()}
                )  # (chunk, )
                # / K: actions.grad = mean of the K perturbed gradients
                (loss.sum() / K).backward()
                total_loss += loss.sum().item() / K
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

            # loss here is the smoothed (perturbed) cost, averaged over K
            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": total_loss, "step": i + 1}
            )
            if self.evaluator is not None and i % self.eval_every == 0:
                # the center plan, never a perturbed copy
                logs, successes, _, _ = self.evaluator.eval_actions(
                    actions.detach(),
                    filename=f"{self.logging_prefix}_output_{i+1}",
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({f"{self.logging_prefix}/sigma": sigma, "step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if self.oracle_stop and np.all(successes):
                    break  # terminate planning if all success
        return actions.detach(), np.full(n_evals, np.inf)  # all actions are valid
