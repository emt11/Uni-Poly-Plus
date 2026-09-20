"""Persistent-homology profiles over the frozen open Trimer (GLT-GALPH P0).

Three channels per record, all Vietoris-Rips over Euclidean distances in the
frozen coordinates so translation/rotation/reflection and atom-permutation
invariance hold exactly:

* channel 0: heavy atoms (Z > 1), H0 death histogram
* channel 1: heavy atoms (Z > 1), H1 death histogram
* channel 2: all atoms (Z >= 1, including explicit isotope H), H0 histogram

Filtration range 0.8-6.0 A in 32 equal bins; histograms are normalized by the
channel point count so a profile is an intensive descriptor.  Computation is
offline only: training reads the sidecar with ``mmap_mode='r'`` and never
imports gudhi.
"""
import numpy as np

PH_RADIUS_MIN = 0.8
PH_RADIUS_MAX = 6.0
PH_BINS = 32
PH_CHANNELS = 3
PH_SCHEMA = 'glt-ph-h0h1-v1'

_EDGES = np.linspace(PH_RADIUS_MIN, PH_RADIUS_MAX, PH_BINS + 1)


def _diagram_deaths(points, dimension, max_edge=PH_RADIUS_MAX):
    """Finite death values of one homology dimension in the VR filtration."""
    import gudhi

    if points.shape[0] < (2 if dimension == 0 else 4):
        return np.empty(0, dtype=np.float64)
    rips = gudhi.RipsComplex(points=np.ascontiguousarray(points, dtype=np.float64).tolist(),
                             max_edge_length=float(max_edge))
    tree = rips.create_simplex_tree(max_dimension=dimension + 1)
    deaths = []
    for dim, (birth, death) in tree.persistence():
        if dim != dimension:
            continue
        if not np.isfinite(death):
            continue
        deaths.append(float(death))
    return np.asarray(deaths, dtype=np.float64)


def _histogram(values, count):
    if count <= 0:
        return np.zeros(PH_BINS, dtype=np.float32)
    hist, _ = np.histogram(values, bins=_EDGES)
    return (hist / float(count)).astype(np.float32)


def ph_profile(positions, atomic_numbers):
    """[3, PH_BINS] float32 profile plus a validity flag.

    ``positions`` [N,3] and ``atomic_numbers`` [N] describe one frozen Trimer
    (all atoms, including hydrogens); invalid or degenerate inputs return
    zeros with ``valid=False`` instead of raising, so a single bad record
    cannot poison a batch.
    """
    positions = np.asarray(positions, dtype=np.float64)
    numbers = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    if positions.ndim != 2 or positions.shape[1] != 3 or numbers.shape[0] != positions.shape[0]:
        raise ValueError('PH input shape mismatch')
    if numbers.shape[0] < 4 or not np.isfinite(positions).all():
        return np.zeros((PH_CHANNELS, PH_BINS), dtype=np.float32), False
    heavy = numbers > 1
    all_atom = numbers >= 1
    heavy_points = positions[heavy]
    all_points = positions[all_atom]
    if heavy_points.shape[0] < 2 or all_points.shape[0] < 2:
        return np.zeros((PH_CHANNELS, PH_BINS), dtype=np.float32), False
    profile = np.zeros((PH_CHANNELS, PH_BINS), dtype=np.float32)
    profile[0] = _histogram(_diagram_deaths(heavy_points, 0), heavy_points.shape[0])
    if heavy_points.shape[0] >= 4:
        profile[1] = _histogram(_diagram_deaths(heavy_points, 1), heavy_points.shape[0])
    profile[2] = _histogram(_diagram_deaths(all_points, 0), all_points.shape[0])
    if not np.isfinite(profile).all():
        return np.zeros((PH_CHANNELS, PH_BINS), dtype=np.float32), False
    return profile, True


def channel_masks(atomic_numbers):
    """Which atoms enter each channel; exposed for tests and audits."""
    numbers = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    return {'heavy': numbers > 1, 'all': numbers >= 1}
