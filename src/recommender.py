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
        """
        alpha: weight for content-based component (0..1). latent/item-based has weight (1-alpha).
        n_components: requested number of latent dimensions for TruncatedSVD (will be adapted to data).
        """
        assert 0.0 <= alpha <= 1.0
        self.alpha = alpha
        self.n_components = n_components
        self.random_state = random_state

        # Will be set in fit()
        self.materials_df: Optional[pd.DataFrame] = None
        self.ratings_df: Optional[pd.DataFrame] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.content_sim: Optional[np.ndarray] = None
        self.item_index: Optional[List[int]] = None
        self.latent_item_vecs: Optional[np.ndarray] = None
        self.latent_sim: Optional[np.ndarray] = None

    def fit(self, materials: pd.DataFrame, ratings: pd.DataFrame):
        """
        Fit both content and latent-item parts.

        materials: DataFrame with columns ['material_id', 'title', 'description']
        ratings: DataFrame with columns ['user_id', 'material_id', 'rating']
        """
        # Basic copies/validation
        self.materials_df = materials.reset_index(drop=True).copy()
        self.ratings_df = ratings.copy()

        # Ensure material_id types are consistent (attempt integer conversion when possible)
        try:
            self.materials_df["material_id"] = self.materials_df["material_id"].astype(int)
        except Exception:
            # if conversion fails, leave as-is
            pass
        try:
            self.ratings_df["material_id"] = self.ratings_df["material_id"].astype(int)
        except Exception:
            pass

        # Content: TF-IDF on title + description
        corpus = (
            self.materials_df.get("title", "").fillna("").astype(str)
            + " "
            + self.materials_df.get("description", "").fillna("").astype(str)
        )
        self.tfidf = TfidfVectorizer(max_features=2000, stop_words="english")
        tfidf_matrix = self.tfidf.fit_transform(corpus)
        # cosine similarity between materials (dense matrix ok for small/medium datasets)
        self.content_sim = cosine_similarity(tfidf_matrix)

        # Latent item vectors: build item-user matrix
        item_user = self.ratings_df.pivot_table(index="material_id", columns="user_id", values="rating", fill_value=0)
        # Keep item order consistent (this order corresponds to rows of item_user)
        self.item_index = list(item_user.index)

        # Determine safe number of components: must be <= n_features (n_users) and <= n_items-1
        n_items = item_user.shape[0]
        n_users = item_user.shape[1]

        # compute a safe maximum components
        max_comp = max(1, min(n_items - 1 if n_items > 1 else 1, n_users))
        n_components = min(self.n_components, max_comp)

        if n_users < 2 or n_items < 2 or n_components < 1:
            # Fallback for tiny datasets: create simple 1-D latent vectors to avoid SVD failure
            latent = np.ones((n_items, 1), dtype=float)
            norm = np.linalg.norm(latent, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            latent = latent / norm
            self.latent_item_vecs = latent
            self.latent_sim = np.dot(latent, latent.T)
        else:
            svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            latent = svd.fit_transform(item_user.values)  # shape (n_items, n_components)
            # Normalize rows for cosine computations
            norm = np.linalg.norm(latent, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            latent = latent / norm
            self.latent_item_vecs = latent
            self.latent_sim = np.dot(latent, latent.T)

    def _material_pos(self, material_id) -> Optional[int]:
        # return positional index in materials_df
        if self.materials_df is None:
            return None
        pos = self.materials_df.index[self.materials_df["material_id"] == material_id]
        if len(pos) == 0:
            return None
        return int(pos[0])

    def _item_index_pos(self, material_id) -> Optional[int]:
        # return position in item_index used for latent_sim
        if self.item_index is None:
            return None
        try:
            return self.item_index.index(material_id)
        except ValueError:
            return None

    def recommend(self, user_id: int, k: int = 5) -> List[int]:
        """
        Return top-k recommended material_id for user_id.

        Strategy:
         - If user has ratings: for each rated item, take its neighbors (by latent_sim and content_sim),
           accumulate weighted scores; final score is sum over neighbors (weighted by similarity and user's rating).
         - If user unknown: return top-k by average rating then fallback to materials list.
        """
        if self.materials_df is None or self.ratings_df is None:
            raise ValueError("Model not fitted. Call fit() before recommend().")

        # materials available
        material_ids_set = set(self.materials_df["material_id"].tolist())

        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]
        # If no history, return popular items
        if user_ratings.empty:
            pop = (
                self.ratings_df.groupby("material_id")["rating"]
                .mean()
                .sort_values(ascending=False)
                .index.tolist()
            )
            pop = [int(m) for m in pop if int(m) in material_ids_set]
            # if not enough popular items, fill with materials list
            if len(pop) < k:
                for mid in self.materials_df["material_id"].tolist():
                    if int(mid) not in pop:
                        pop.append(int(mid))
                    if len(pop) >= k:
                        break
            return pop[:k]

        scores: Dict[int, float] = {}
        # For each rated material, accumulate neighbor scores
        for _, row in user_ratings.iterrows():
            try:
                mat_id = int(row["material_id"])
            except Exception:
                continue
            rating_val = float(row.get("rating", 0.0))
            pos_content = self._material_pos(mat_id)
            pos_latent = self._item_index_pos(mat_id)
            # content neighbors
            if pos_content is not None and self.content_sim is not None:
                sim_vec = self.content_sim[pos_content]  # length = n_materials
                for idx, sim in enumerate(sim_vec):
                    cand_id = int(self.materials_df.loc[idx, "material_id"])
                    scores[cand_id] = scores.get(cand_id, 0.0) + self.alpha * float(sim) * rating_val
            # latent neighbors
            if pos_latent is not None and self.latent_sim is not None:
                sim_vec = self.latent_sim[pos_latent]  # length = n_items (order = item_index)
                for idx, sim in enumerate(sim_vec):
                    try:
                        cand_id = int(self.item_index[idx])
                    except Exception:
                        continue
                    scores[cand_id] = scores.get(cand_id, 0.0) + (1.0 - self.alpha) * float(sim) * rating_val

        # Remove seen items
        seen = set(user_ratings["material_id"].tolist())
        # Sort by score
        ranked = [mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True) if mid not in seen]

        # If not enough, fill with popular
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
        # Final fallback: add materials by their order
        if len(ranked) < k:
            for mid in self.materials_df["material_id"].tolist():
                mid_int = int(mid)
                if mid_int not in ranked and mid_int not in seen:
                    ranked.append(mid_int)
                if len(ranked) >= k:
                    break

        return ranked[:k]

    def save(self, path: str = "model.joblib"):
        """
        Save models and metadata to joblib.
        """
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