"""Experiment 3 — a sparsity phase diagram for superposition.

A statistical-physicist's instinct: pick an *order parameter* (a number that
summarises the macroscopic state of the system) and watch how it responds as we
sweep a *control parameter* (here, the sparsity). A sharp change in the order
parameter signals a phase transition.

We use a larger model (n = 40 features, m = 10 dimensions) and sweep sparsity
from dense to very sparse. We track two complementary order parameters:

  * features-per-dimension  φ = (number of features actually stored) / m.
        A feature counts as "stored" if its representation vector exceeds a
        norm threshold τ = 0.5. This is well defined because the learned norms
        are bimodal — stored features have ||W_i|| ≈ 1, dropped ones ≈ 0 — so any
        τ inside that gap gives the same count.
        - φ = 1  means each stored feature owns a dimension (no superposition).
        - φ > 1  means superposition. The ceiling is n/m = 4.

  * bottleneck usage = (sum_i D_i) / m, the fraction of representational
        capacity in use. It saturates near 1.

Unlike Experiments 1 and 2 (which show a single clean ground state via
best-of-seeds), a phase diagram should *average over disorder*: for each
sparsity we train many independent seeds and report the mean order parameter
with its standard deviation. The sparsities are spaced logarithmically in
1/density so the sparse (superposition) regime is well sampled.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from _common import figure_path, log
from superposition import TrainConfig, train
from superposition.metrics import fraction_dimensions_used, num_represented

N_FEATURES = 40
N_HIDDEN = 10
IMPORTANCE = 0.95          # geometric importance decay; clean no-superposition baseline
THRESHOLD = 0.5           # a feature is "stored" if ||W_i|| > THRESHOLD
SEEDS = 10                # independent runs averaged per sparsity (disorder average)
STEPS = 4000
INV_DENSITY = np.logspace(0.0, 2.0, 15)   # 1 .. 100, log-spaced
SPARSITIES = 1.0 - 1.0 / INV_DENSITY        # 0 .. 0.99


def main() -> None:
    phi_mean, phi_std, usage_mean = [], [], []

    for sparsity in SPARSITIES:
        phis, usages = [], []
        for seed in range(SEEDS):
            cfg = TrainConfig(
                n_features=N_FEATURES,
                n_hidden=N_HIDDEN,
                sparsity=float(sparsity),
                importance=IMPORTANCE,
                steps=STEPS,
                lr=1e-3,
                seed=seed,
            )
            W = train(cfg).W.detach()
            phis.append(num_represented(W, THRESHOLD) / N_HIDDEN)
            usages.append(fraction_dimensions_used(W))
        phis, usages = np.array(phis), np.array(usages)
        phi_mean.append(phis.mean())
        phi_std.append(phis.std())
        usage_mean.append(usages.mean())
        log(
            f"sparsity={sparsity:.3f}  phi={phis.mean():.2f}±{phis.std():.2f}  "
            f"usage={usages.mean():.2f}"
        )

    phi_mean = np.array(phi_mean)
    phi_std = np.array(phi_std)
    usage_mean = np.array(usage_mean)
    density = 1.0 - SPARSITIES
    inv_density = 1.0 / density
    ceiling = N_FEATURES / N_HIDDEN

    # Single panel: the order parameter vs feature rarity (1/density), log axis.
    # Both order parameters on one plot — features-per-dimension climbing while
    # bottleneck usage stays pinned at 1 is the signature of pure superposition.
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.errorbar(
        inv_density, phi_mean, yerr=phi_std, fmt="o-", color="#c33", lw=2, capsize=3,
        label=r"features per dimension  $\phi$  (mean $\pm$ std)",
    )
    ax.plot(inv_density, usage_mean, "s-", color="#36c", lw=2,
            label=r"bottleneck usage  $\frac{1}{m}\sum_i D_i$")
    ax.axhline(ceiling, ls=":", color="#c33", lw=1.2,
               label=rf"ceiling  $n/m = {int(ceiling)}$")
    ax.axhline(1.0, ls="--", color="0.6", lw=1,
               label=r"no superposition  ($\phi = 1$)")
    ax.set_xscale("log")
    ax.set_ylim(0.9, ceiling + 0.3)
    ax.set_xlabel(r"feature rarity     $1/\mathrm{density} = 1/(1-\mathrm{sparsity})$")
    ax.set_ylabel("order parameter")
    ax.set_title(
        f"Sparsity drives a superposition phase transition   ($n={N_FEATURES}$, $m={N_HIDDEN}$)\n"
        rf"$\phi$ climbs from 1 (no superposition) to the ceiling $n/m={int(ceiling)}$ as features get rarer"
    )
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    out = figure_path("03_phase_diagram.png")
    fig.savefig(out, dpi=150)
    log(f"saved {out}")


if __name__ == "__main__":
    main()
