"""Categorical-only baseline implementations; no support estimator is included."""

from .registry import MODELS, make_model

__all__ = ["MODELS", "make_model"]
