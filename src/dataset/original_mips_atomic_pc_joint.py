"""50K fixed-subset collator for the joint Original-MIPS/Atomic-PC pilot."""

from __future__ import annotations

from .original_mips_atomic_pc import OriginalMIPSAtomicPCCollator


class OriginalMIPSAtomicPCJointCollator(OriginalMIPSAtomicPCCollator):
    """Join restored MD200 with frozen full-Trimer LMDB geometry.

    Geometry is already attached to each dataset item by the immutable
    ``mips_trimer_scage`` cache.  This collator does not regenerate, retry or
    fallback any record; callers must pre-filter invalid geometry rows and keep
    the excluded keys in the coverage report.
    """

    pass


__all__ = ["OriginalMIPSAtomicPCJointCollator"]
