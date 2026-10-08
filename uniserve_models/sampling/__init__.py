"""Numerical token selection."""

from .categorical import sample_categorical
from .greedy import greedy
from .top_k import sample_top_k

__all__ = ["greedy", "sample_categorical", "sample_top_k"]
