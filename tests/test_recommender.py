import pandas as pd
from src.recommender import Recommender
from src.utils import load_materials, load_ratings


def test_fit_and_recommend_runs():
    materials = load_materials("data/sample_materials.csv")
    ratings = load_ratings("data/sample_ratings.csv")
    rec = Recommender(alpha=0.5, n_components=5, random_state=0)
    rec.fit(materials, ratings)
    out = rec.recommend(10, k=3)
    assert isinstance(out, list)
    assert len(out) <= 3
    assert all(isinstance(x, int) for x in out)
    mat_ids = set(materials["material_id"].tolist())
    assert all((m in mat_ids) for m in out)


def test_unknown_user_popular_fallback():
    materials = load_materials("data/sample_materials.csv")
    ratings = load_ratings("data/sample_ratings.csv")
    rec = Recommender()
    rec.fit(materials, ratings)
    out = rec.recommend(9999, k=2)
    assert len(out) == 2


def test_save_and_load(tmp_path):
    materials = load_materials("data/sample_materials.csv")
    ratings = load_ratings("data/sample_ratings.csv")
    rec = Recommender(alpha=0.4, n_components=3, random_state=1)
    rec.fit(materials, ratings)
    p = tmp_path / "model.joblib"
    rec.save(str(p))
    rec2 = Recommender.load(str(p))
    out = rec2.recommend(10, k=2)
    assert isinstance(out, list)


def test_recommender_is_deterministic():
    materials = load_materials("data/sample_materials.csv")
    ratings = load_ratings("data/sample_ratings.csv")
    rec = Recommender(alpha=0.5, n_components=5, random_state=42)
    rec.fit(materials, ratings)
    a = rec.recommend(10, k=3)
    b = rec.recommend(10, k=3)
    assert a == b