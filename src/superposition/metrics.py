"""Geometric observables for a trained weight matrix.

These are the "order parameters" of the project — scalar or vector quantities,
computed directly from the learned matrix ``W`` (shape ``(m, n)``), that tell us
*how* the network chose to store its features.

The central quantity is the **feature dimensionality** introduced in
Elhage et al., "Toy Models of Superposition" (2022):

    D_i = ||W_i||^2 / sum_j (Ŵ_i · W_j)^2

where ``W_i`` is the representation vector of feature ``i`` (column ``i`` of
``W``) and ``Ŵ_i`` is its unit vector. Intuitively, ``D_i`` is the number of
bottleneck dimensions that feature ``i`` effectively gets to itself:

    D_i = 1   -> the feature owns a full dimension (no superposition)
    D_i = 0   -> the feature is not represented at all
    0 < D_i < 1 -> the feature shares its dimensions with others (superposition)

The summed dimensionality ``sum_i D_i`` can never exceed ``m``: the network has
only ``m`` dimensions to hand out. So ``(sum_i D_i) / m`` is a clean measure of
how fully the bottleneck is being used.
"""

from __future__ import annotations

import torch


def gram_matrix(W: torch.Tensor) -> torch.Tensor:
    """Return the ``(n, n)`` matrix of feature-vector overlaps ``WᵀW``.

    Entry ``(i, j)`` is ``W_i · W_j``. Diagonal entries are squared norms;
    off-diagonal entries are the *interference* between features.
    """
    return W.T @ W


def feature_norms(W: torch.Tensor) -> torch.Tensor:
    """Return ``||W_i||`` for each feature ``i`` (shape ``(n,)``)."""
    return W.norm(dim=0)


def feature_dimensionality(W: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return the per-feature dimensionality ``D_i`` (shape ``(n,)``).

    Features with negligible norm (not represented) are assigned ``D_i = 0``.
    """
    norms = feature_norms(W)  # (n,)
    safe = norms.clamp_min(eps)
    W_hat = W / safe  # unit vectors as columns
    # projections[i, j] = Ŵ_i · W_j
    projections = W_hat.T @ W  # (n, n)
    denom = (projections ** 2).sum(dim=1)  # sum over j
    dims = norms ** 2 / denom.clamp_min(eps)
    dims[norms < eps] = 0.0
    return dims


def total_dimensionality(W: torch.Tensor) -> float:
    """Return ``sum_i D_i`` — total bottleneck dimensions in use."""
    return float(feature_dimensionality(W).sum())


def fraction_dimensions_used(W: torch.Tensor) -> float:
    """Return ``(sum_i D_i) / m`` — fraction of the bottleneck that is used.

    This is our primary scalar order parameter for the phase diagram. It rises
    above the "no superposition" ceiling exactly when the network starts
    packing more features than it has dimensions.
    """
    m = W.shape[0]
    return total_dimensionality(W) / m


def num_represented(W: torch.Tensor, threshold: float = 0.5) -> int:
    """Count features whose norm exceeds ``threshold`` (i.e. are stored at all)."""
    return int((feature_norms(W) > threshold).sum())


def frustration_energy(
    W: torch.Tensor, threshold: float = 0.5, importance: torch.Tensor | None = None
) -> float:
    """(Importance-weighted) squared overlap between *represented* directions.

    Uniform energy (``importance=None``):

        E   = sum over i<j of (Ŵ_i · Ŵ_j)^2

    Importance-weighted ("generalized Thomson") energy, when ``importance`` is
    the per-feature importance vector I (shape (n,)):

        E_I = sum over i<j of (I_i + I_j) (Ŵ_i · Ŵ_j)^2

    The weighting follows from the loss: the error in reconstructing feature i
    is weighted by I_i, and when features i and j collide that error picks up a
    term ∝ (Ŵ_i · Ŵ_j)^2, contributing I_i (Ŵ_i·Ŵ_j)^2; summing the symmetric
    i- and j-contributions over each pair gives the (I_i + I_j) factor. Setting
    all I_i = 1 recovers 2E — the uniform-importance case, which is exactly the
    regime in which the model's interference term is a *generalized Thomson
    problem* (Elhage et al.). With non-uniform importance the minimum-energy
    polytopes deform.

    This is the network's analogue of a physical repulsion energy: features
    "want" to be orthogonal (overlap 0), but there are more of them than
    dimensions, so they cannot all be mutually orthogonal — the system is
    *frustrated*. The network, by minimising reconstruction loss, ends up
    solving a closely related packing problem.
    """
    norms = feature_norms(W)
    rep = norms > threshold
    if int(rep.sum()) < 2:
        return 0.0
    Wr = W[:, rep]
    W_hat = Wr / Wr.norm(dim=0)
    overlaps = W_hat.T @ W_hat  # (k, k)
    k = overlaps.shape[0]
    iu, ju = torch.triu_indices(k, k, offset=1)
    pair = overlaps[iu, ju] ** 2
    if importance is None:
        return float(pair.sum())
    I = torch.as_tensor(importance, dtype=W.dtype, device=W.device)[rep]
    return float(((I[iu] + I[ju]) * pair).sum())
