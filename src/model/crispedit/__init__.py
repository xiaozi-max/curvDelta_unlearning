import re

import torch


def resolve_target_modules(model, target_modules, layers=None):
    """Collect the model's 2-D Linear layers whose leaf name is in `target_modules`.

    `layers` restricts to transformer layers `model.layers.<i>`; empty = all.
    """
    wanted = set(target_modules)
    keep = set(layers) if layers else None
    resolved = {}
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2 or name.rsplit(".", 1)[-1] not in wanted:
            continue
        if keep is not None:
            hit = re.search(r"\.layers\.(\d+)\.", name)
            if hit is None or int(hit.group(1)) not in keep:
                continue
        resolved[name] = module
    return resolved


def _rank_and_threshold(eigenvalues, percent):
    """(rank, cutoff) such that top-`percent` Kronecker energy is above cutoff."""
    total = eigenvalues.sum()
    if total <= 0:
        return 0, 0.0
    sorted_vals, _ = torch.sort(eigenvalues, descending=True)
    cum = torch.cumsum(sorted_vals, dim=0) / total
    rank = torch.searchsorted(cum, percent).item() + 1
    rank = min(rank, len(sorted_vals))
    return rank, sorted_vals[rank - 1].item()


def projection_cache_from_covariances(A, B, energy_threshold=0.9):
    """Return CPU ``{"Ua", "Ub", "M"}`` with ``M`` True on low curvature."""
    Sa, Ua = torch.linalg.eigh(A.float())
    Sb, Ub = torch.linalg.eigh(B.float())
    Sa = Sa.clamp(min=0)
    Sb = Sb.clamp(min=0)
    vals = torch.outer(Sa, Sb)
    if vals.sum() <= 0:
        # No curvature information: allow every direction.
        M = torch.ones_like(vals, dtype=torch.bool)
    else:
        _, threshold = _rank_and_threshold(vals.view(-1), energy_threshold)
        M = vals < threshold
    return {"Ua": Ua.cpu(), "Ub": Ub.cpu(), "M": M.cpu()}


def project_matrix_gradient(grad, cache):
    """CrispEdit projection: ``U_b ((U_b^T grad U_a) * M^T) U_a^T``."""
    Ua = cache["Ua"].to(grad.device, grad.dtype)
    Ub = cache["Ub"].to(grad.device, grad.dtype)
    M = cache["M"].to(grad.device)
    return Ub @ ((Ub.T @ grad @ Ua) * M.T) @ Ua.T
