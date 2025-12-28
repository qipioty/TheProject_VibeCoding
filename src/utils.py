"""
Utility helpers for data loading and evaluation metrics.
"""
from typing import List
import pandas as pd


def load_materials(path: str = "data/sample_materials.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_ratings(path: str = "data/sample_ratings.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    if k == 0:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for r in recommended_k if r in relevant)
    return hits / k