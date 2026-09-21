"""Persistent-homology profiles over the frozen open Trimer (GLT-GALPH r2).

Schema ``glt-ph-betti-v2``: three normalized Betti curves sampled on 32 radii
in 0.8-6.0 A, all Vietoris-Rips over Euclidean distances in the frozen
coordinates, so translation/rotation/reflection and atom-permutation
invariance hold exactly:

* channel 0: heavy atoms (Z > 1), beta_0(r)
* channel 1: heavy atoms (Z > 1), beta_1(r)
* channel 2: all atoms (Z >= 1, including explicit isotope H), beta_0(r)

beta_k(r) = #{(b, d) : b <= r < d} divided by the channel atom count.  The
legacy death-value histogram of schema v1 is kept only for audit comparison.
Computation is offline only: training reads the sidecar with
``mmap_mode='r'`` and never imports gudhi.
"""
import numpy as np

PH_RADIUS_MIN = 0.8
PH_RADIUS_MAX = 6.0
PH_BINS = 32
PH_CHANNELS = 3
PH_SCHEMA = 'glt-ph-betti-v2'

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


_GRID = np.linspace(PH_RADIUS_MIN, PH_RADIUS_MAX, PH_BINS)


def radius_grid():
    """The 32 sampling radii actually used by this schema (read, never re-derived)."""
    return _GRID.copy()


def _histogram(values, count):
    """Legacy death-value histogram (schema v1); kept for audit comparison."""
    if count <= 0:
        return np.zeros(PH_BINS, dtype=np.float32)
    hist, _ = np.histogram(values, bins=_EDGES)
    return (hist / float(count)).astype(np.float32)


def _persistence_pairs(points, dimension, max_edge=PH_RADIUS_MAX):
    """(birth, death) pairs of one dimension; infinite deaths kept as inf."""
    import gudhi

    if points.shape[0] < (2 if dimension == 0 else 4):
        return []
    rips = gudhi.RipsComplex(points=np.ascontiguousarray(points, dtype=np.float64).tolist(),
                             max_edge_length=float(max_edge))
    tree = rips.create_simplex_tree(max_dimension=dimension + 1)
    pairs = []
    for dim, (birth, death) in tree.persistence():
        if dim != dimension:
            continue
        pairs.append((float(birth), float(death)))
    return pairs


def betti_curve(points, dimension, count):
    """Normalized Betti curve: beta_k(r) = #{(b, d) : b <= r < d} / count."""
    pairs = _persistence_pairs(points, dimension)
    curve = np.zeros(PH_BINS, dtype=np.float32)
    if not pairs or count <= 0:
        return curve
    for birth, death in pairs:
        active = (_GRID >= birth) & (_GRID < death) if np.isfinite(death) \
            else (_GRID >= birth)
        curve += active.astype(np.float32)
    return (curve / float(count)).astype(np.float32)


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
    profile[0] = betti_curve(heavy_points, 0, heavy_points.shape[0])
    if heavy_points.shape[0] >= 4:
        profile[1] = betti_curve(heavy_points, 1, heavy_points.shape[0])
    profile[2] = betti_curve(all_points, 0, all_points.shape[0])
    if not np.isfinite(profile).all():
        return np.zeros((PH_CHANNELS, PH_BINS), dtype=np.float32), False
    return profile, True


def channel_masks(atomic_numbers):
    """Which atoms enter each channel; exposed for tests and audits."""
    numbers = np.asarray(atomic_numbers, dtype=np.int64).reshape(-1)
    return {'heavy': numbers > 1, 'all': numbers >= 1}
