"""Top-level package marker.

Avoid eager imports here: KG CLI modules import src.kg_pipeline and should not require
RDKit, PyG, or model dependencies just to show --help.
"""

__all__ = ["kg_pipeline", "utils", "modules", "dataset"]
