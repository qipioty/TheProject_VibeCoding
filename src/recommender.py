"""
Study Material Recommender

Hybrid recommender:
 - Content-based: TF-IDF on title + description
 - Latent item factors: TruncatedSVD on item-user matrix (simple matrix-factorization-style)
 Final score: weighted sum of content similarity and latent-item similarity.

API:
    rec = Recommender(alpha=0.6, n_components=20, random_state=42)
    rec.fit(materials_df, ratings_df)
    rec.recommend(user_id, k=5)
"""
from typing import List, Optional, Dict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class Recommender:
    def __init__(self, alpha: float = 0.6, n_components: int = 20, random_state: int = 42):
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.n_components = n_components
        self.random_state = random_state

        self.materials_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.content_sim: Optional[np.ndarray] = None
        self.item_index: Optional[List[int]] = None
        self.latent_item_vecs: Optional[np.ndarray] = None
        self.latent_sim: Optional[np.ndarray] = None

    def fit(self, materials: pd.DataFrame, ratings: pd.DataFrame):
        self.materials_df = materials.reset_index(drop=True).copy()
        self.ratings_df = ratings.copy()

        self.materials_df["material_id"] = self.materials_df["material_id"].astype(int)
        self.ratings_df["material_id"] = self.ratings_df["material_id"].astype(int)

        corpus = (
            self.materials_df.get("title", "").fillna("") + " " + self.materials_df.get("description", "").fillna("")
        )
        self.tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf_matrix = self.tfidf.fit_transform(corpus)
        self.content_sim = cosine_similarity(tfidf_matrix)

        item_user = self.ratings_df.pivot_table(index="material_id", columns="user_id", values="rating", fill_value=0)
        self.item_index = list(item_user.index)

        n_items = item_user.shape[0]
        n_components = min(self.n_components, max(1, n_items - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        latent = svd.fit_transform(item_user.values)
        norm = np.linalg.norm(latent, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        latent = latent / norm
        self.latent_item_vecs = latent
        self.latent_sim = np.dot(latent, latent.T)

    def _material_pos(self, material_id: int) -> Optional[int]:
        pos = self.materials_df.index[self.materials_df["material_id"] == int(material_id)]
        if len(pos) == 0:
            return None
        return int(pos[0])

    def _item_index_pos(self, material_id: int) -> Optional[int]:
        try:
            return self.item_index.index(int(material_id))
        except ValueError:
            return None

    def recommend(self, user_id: int, k: int = 5) -> List[int]:
        if self.materials_df is None or self.ratings_df is None:
            raise ValueError("Model not fitted. Call fit() before recommend().")

        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        if user_ratings.empty:
            pop = (
                self.ratings_df.groupby("material_id")["rating"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
            pop = [int(m) for m in pop if int(m) in set(self.materials_df["material_id"].tolist())]
            return pop[:k]

        scores: Dict[int, float] = {}
        for _, row in user_ratings.iterrows():
            mat_id = int(row["material_id"])
            rating_val = float(row["rating"])
            pos_content = self._material_pos(mat_id)
            pos_latent = self._item_index_pos(mat_id)
            if pos_content is not None and self.content_sim is not None:
                sim_vec = self.content_sim[pos_content]
                for idx, sim in enumerate(sim_vec):
                    cand_id = int(self.materials_df.loc[idx, "material_id"])
                    scores[cand_id] = scores.get(cand_id, 0.0) + self.alpha * sim * rating_val
            if pos_latent is not None and self.latent_sim is not None:
                sim_vec = self.latent_sim[pos_latent]
                for idx, sim in enumerate(sim_vec):
                    cand_id = int(self.item_index[idx])
                    scores[cand_id] = scores.get(cand_id, 0.0) + (1.0 - self.alpha) * sim * rating_val

        seen = set(user_ratings["material_id"].tolist())
        ranked = [mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True) if mid not in seen]

        if len(ranked) < k:
            pop = (
                self.ratings_df.groupby("material_id")["rating"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
            for mid in pop:
                if int(mid) not in ranked and int(mid) not in seen:
                    ranked.append(int(mid))
                if len(ranked) >= k:
                    break

        return ranked[:k]

    def save(self, path: str = "model.joblib"):
        payload = {
            "alpha": self.alpha,
            "n_components": self.n_components,
            "random_state": self.random_state,
            "materials_df": self.materials_df,
            "ratings_df": self.ratings_df,
            "tfidf": self.tfidf,
            "content_sim": self.content_sim,
            "item_index": self.item_index,
            "latent_item_vecs": self.latent_item_vecs,
            "latent_sim": self.latent_sim,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str = "model.joblib") -> "Recommender":
        payload = joblib.load(path)
        rec = cls(alpha=payload.get("alpha", 0.6), n_components=payload.get("n_components", 20),
                  random_state=payload.get("random_state", 42))
        rec.materials_df = payload.get("materials_df", None)
        rec.ratings_df = payload.get("ratings_df", None)
        rec.tfidf = payload.get("tfidf", None)
        rec.content_sim = payload.get("content_sim", None)
        rec.item_index = payload.get("item_index", None)
        rec.latent_item_vecs = payload.get("latent_item_vecs", None)
        rec.latent_sim = payload.get("latent_sim", None)
        return rec