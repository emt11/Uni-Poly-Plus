"""Entry points for the retained MTS-GLT-v2 pretraining route."""

from .config import dataset_kwargs_from_args, parse_arguments

__all__ = ["dataset_kwargs_from_args", "parse_arguments"]
