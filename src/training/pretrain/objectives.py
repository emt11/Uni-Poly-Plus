"""Pure objective composition for the MTS pretraining engine."""

from __future__ import annotations


def joint_masked_atom_angle_loss(payload, args, *, atom_count, angle_count):
    """Compose globally normalized masked-atom and angle losses."""
    atom_mean = payload["loss_terms"]["masked_atom_sum"] / max(1, int(atom_count))
    angle_mean = payload["loss_terms"]["angle_sum"] / max(1, int(angle_count))
    return (
        float(args.scage_mips_mask_weight) * atom_mean
        + float(args.graph_angle_weight) * angle_mean
        + payload["zero_reference"]
    )


def compose_joint_payload_loss(payload, args, global_counts):
    """Compose one forward payload using synchronized global counts."""
    return joint_masked_atom_angle_loss(
        payload,
        args,
        atom_count=global_counts["masked_atoms"],
        angle_count=global_counts["angle_graphs"],
    )
