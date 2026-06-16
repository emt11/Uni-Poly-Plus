"""Reserved interfaces for future online literature retrieval.

The MVP intentionally operates on local documents and never queries with labels.
"""


def retrieve_literature(*args, **kwargs):
    raise NotImplementedError("MVP supports local documents only; populate kg_work/documents first")

