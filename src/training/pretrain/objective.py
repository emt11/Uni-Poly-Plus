"""Compatibility spelling for the renamed :mod:`objectives` module."""

from __future__ import annotations

from .objectives import compose_joint_payload_loss, joint_masked_atom_angle_loss


__all__ = ["compose_joint_payload_loss", "joint_masked_atom_angle_loss"]
