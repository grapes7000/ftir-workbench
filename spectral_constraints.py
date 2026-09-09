from __future__ import annotations

"""Explicit hard spectral constraints for FTIR Workbench.

A hard exclusion is not an optimization choice. The requested wavelengths are
removed from X and the wavenumber axis before downstream analysis receives them.
"""

import numpy as np

import core


def _merge_ranges(ranges):
    if not ranges:
        return []
    ordered = sorted((float(a), float(b)) for a, b in ranges)
    merged = [list(ordered[0])]
    for a, b in ordered[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def normalize_exclusion(text: str | None) -> str:
    if not str(text or "").strip():
        return ""
    ranges = _merge_ranges(core.parse_ranges(text))
    return ",".join(f"{a:g}-{b:g}" for a, b in ranges)


def apply_hard_exclusion(X, wn, text: str | None):
    """Remove excluded wavelengths before any analysis/model can see them."""
    x = np.asarray(X, dtype=float)
    w = np.asarray(wn, dtype=float)
    if x.ndim != 2:
        raise ValueError("Spectral matrix must be two-dimensional.")
    if len(w) != x.shape[1]:
        raise ValueError("Wavenumber axis length does not match the spectral matrix.")

    normalized = normalize_exclusion(text)
    ranges = core.parse_ranges(normalized) if normalized else []
    keep = np.ones(len(w), dtype=bool)
    for a, b in ranges:
        keep &= ~((w >= a) & (w <= b))

    retained = int(keep.sum())
    if retained < 2:
        raise ValueError(
            "Hard spectral exclusion removes too many variables; at least two must remain."
        )

    info = {
        "normalized": normalized,
        "ranges": ranges,
        "removed_variables": int((~keep).sum()),
        "retained_variables": retained,
        "original_variables": int(len(w)),
        "mask": keep,
    }
    return x[:, keep], w[keep], info


def describe_exclusion(info: dict) -> str:
    text = info.get("normalized", "") or "none"
    return (
        f"Hard exclusion: {text} cm⁻¹ · removed {int(info.get('removed_variables', 0))} "
        f"of {int(info.get('original_variables', info.get('retained_variables', 0)))} "
        f"spectral variables · {int(info.get('retained_variables', 0))} retained."
    )
