"""Entry points for the retained v2 and independent v3 pretraining routes."""

from .config import dataset_kwargs_from_args, parse_arguments

__all__ = ["dataset_kwargs_from_args", "parse_arguments"]
