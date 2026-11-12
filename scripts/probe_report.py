"""Summarize a probe run (planner=probe) and apply the pre-registered smoothing gate.

usage: python scripts/probe_report.py <probe_run_dir>

Reads <probe_run_dir>/probe.npz and prints, per sigma:
  barrier   median over episodes of  max_{t in [0,1]} J_sigma(a(t)) - J_sigma(a(0))  along
            a(t) = (1 - t) a_GD + t a_CEM, and its ratio to the sigma = 0 median
  cos>0     episodes with cos(-g_sigma, a_CEM - a_GD) > 0 (a descent step heads toward CEM)
and per episode whether J(a_GD) < J(a_CEM) at sigma = 0 ("fooled": the model prefers GD's plan).

Gate (pre-registered, unchanged): PASS if for some sigma the median barrier is
<= 0.5 x its sigma = 0 value AND cos > 0 in >= 6 of the 8 episodes. Only sigma > 0 are
candidates (sigma = 0 is the reference). Prints PASS/FAIL, the smallest passing sigma and
the sigma0 candidates {1/3, 1, 3} x that sigma.

Also saves <probe_run_dir>/probe_lines.png: J_sigma vs t per episode, one line per sigma,
GD->CEM line (left) and the random direction through a_GD (right).
"""
import os
import sys

import numpy as np

GATE_BARRIER_FRAC = 0.5  # median barrier <= 0.5 x the sigma = 0 median
GATE_MIN_COS = 6         # cos(-g_sigma, d) > 0 in >= 6 ...
GATE_N_EPS = 8           # ... of the 8 episodes


def main(run_dir):
    z = np.load(os.path.join(run_dir, "probe.npz"))
    eps, sigmas, t = z["episodes"], z["sigmas"], z["t"]
    J_line, J_rand = z["J_line"], z["J_rand"]  # (E, S, n_t)
    E = len(eps)
    zero = np.flatnonzero(sigmas == 0)
    if len(zero) != 1:
        sys.exit(f"probe.npz has sigmas {sigmas.tolist()}: need exactly one sigma = 0 as the reference")
    s0 = zero[0]
    i0 = np.argmin(np.abs(t))
    if abs(t[i0]) > 1e-6:
        sys.exit(f"t grid {t.tolist()} has no point at t = 0")
    inside = (t >= -1e-6) & (t <= 1 + 1e-6)

    barrier = J_line[:, :, inside].max(-1) - J_line[:, :, i0]  # (E, S), >= 0
    med = np.median(barrier, axis=0)                              # (S,)
    n_cos = (-z["grad_cos"] > 0).sum(0)                           # (S,)

    print(f"{run_dir}: {E} episodes {eps.tolist()}, M={int(z['M'])}, t in [{t[0]:g}, {t[-1]:g}] ({len(t)} pts)")
    if E != GATE_N_EPS:
        print(f"warning: the gate is defined for {GATE_N_EPS} episodes; this probe has {E}")
    print(f"\n{'sigma':>7} {'median barrier':>15} {'/ sigma=0':>10} {'cos>0':>7}   gate")
    passing = []
    for s, sig in enumerate(sigmas):
        ratio = med[s] / med[s0] if med[s0] > 0 else float("nan")
        if s == s0:
            tag = "(reference)"
        else:
            ok_b = med[s] <= GATE_BARRIER_FRAC * med[s0]
            ok_c = n_cos[s] >= GATE_MIN_COS
            tag = f"barrier {'ok' if ok_b else 'no'}, cos {'ok' if ok_c else 'no'}"
            if ok_b and ok_c:
                passing.append(float(sig))
        print(f"{sig:7g} {med[s]:15.4f} {ratio:10.3f} {n_cos[s]:4d}/{E}   {tag}")

    J_gd, J_cem = z["J_gd"], z["J_cem"]
    fooled = J_gd < J_cem
    print(f"\nper episode at sigma = 0 (fooled = J(a_GD) < J(a_CEM)):")
    print(f"{'episode':>8} {'J(a_GD)':>9} {'J(a_CEM)':>9} {'barrier':>8}  fooled")
    for i, e in enumerate(eps):
        print(f"{int(e):8d} {J_gd[i]:9.4f} {J_cem[i]:9.4f} {barrier[i, s0]:8.4f}  {'yes' if fooled[i] else 'no'}")
    print(f"fooled in {int(fooled.sum())}/{E} episodes")

    print(f"\ngate: some sigma > 0 with median barrier <= {GATE_BARRIER_FRAC} x sigma=0 "
          f"and cos > 0 in >= {GATE_MIN_COS} of {GATE_N_EPS} episodes")
    if passing:
        best = min(passing)
        cands = [best / 3, best, best * 3]
        print(f"PASS  smallest passing sigma = {best:g}  "
              f"sigma0 candidates = {' '.join(f'{c:.3g}' for c in cands)}")
    else:
        print("FAIL  no sigma passes")

    plot(run_dir, eps, sigmas, t, J_line, J_rand)


def plot(run_dir, eps, sigmas, t, J_line, J_rand):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E = len(eps)
    fig, axes = plt.subplots(E, 2, figsize=(10, 2.6 * E), squeeze=False, sharex=True)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(sigmas)))
    for i, e in enumerate(eps):
        for j, (J, title) in enumerate(((J_line, "GD -> CEM line"), (J_rand, "random direction"))):
            ax = axes[i, j]
            for s, sig in enumerate(sigmas):
                ax.plot(t, J[i, s], color=colors[s], label=f"sigma={sig:g}")
            ax.axvline(0, color="gray", lw=0.6, ls="--")
            if j == 0:
                ax.axvline(1, color="gray", lw=0.6, ls="--")
            ax.set_title(f"episode {int(e)}: {title}", fontsize=9)
            ax.set_ylabel("J")
    for ax in axes[-1]:
        ax.set_xlabel("t (0 = a_GD; left: 1 = a_CEM; right: |t| x |a_CEM - a_GD| along u)")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(run_dir, "probe_lines.png")
    fig.savefig(out, dpi=110)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__.split("\n\n")[1])
    main(sys.argv[1].rstrip("/"))
