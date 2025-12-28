"""
Command-line demo for Study Material Recommender.

Usage:
    python src/main.py data/sample_materials.csv data/sample_ratings.csv
"""
from src.recommender import Recommender
from src.utils import load_materials, load_ratings


def demo(materials_path="data/sample_materials.csv", ratings_path="data/sample_ratings.csv"):
    materials = load_materials(materials_path)
    ratings = load_ratings(ratings_path)
    rec = Recommender(alpha=0.6, n_components=10)
    rec.fit(materials, ratings)
    users = sorted(ratings["user_id"].unique())[:5]
    print(f"Loaded {len(materials)} materials, {len(ratings)} ratings.")
    for u in users:
        recs = rec.recommend(u, k=5)
        print(f"User {u} -> recommended: {recs}")


if __name__ == "__main__":
    demo()