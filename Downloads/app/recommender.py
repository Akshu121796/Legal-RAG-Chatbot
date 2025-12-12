import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_PATH = Path("data/items.csv")

_items_df = None
_tfidf = None
_tfidf_matrix = None
_sim_matrix = None


def _ensure_model():
    global _items_df, _tfidf, _tfidf_matrix, _sim_matrix
    if _items_df is not None:
        return

    df = pd.read_csv(DATA_PATH)
    df["id"] = df["id"].astype(str)

    for col in ["title", "category", "tags", "description"]:
        if col not in df.columns:
            df[col] = ""

    df["text"] = df[["title", "category", "tags", "description"]].fillna("").agg(
        " ".join, axis=1
    )

    _items_df = df

    _tfidf = TfidfVectorizer(stop_words="english")
    _tfidf_matrix = _tfidf.fit_transform(_items_df["text"])
    _sim_matrix = linear_kernel(_tfidf_matrix, _tfidf_matrix)


def get_all_items():
    _ensure_model()
    return _items_df.copy()


def _id_to_index(item_id):
    _ensure_model()
    matches = _items_df.index[_items_df["id"] == str(item_id)].tolist()
    return matches[0] if matches else None


def get_similar_items(item_id, top_n=8):
    _ensure_model()
    idx = _id_to_index(item_id)
    if idx is None:
        return []
    scores = list(enumerate(_sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    indices = [i for i, _ in scores]
    return _items_df.iloc[indices].to_dict(orient="records")


def get_personalized_recommendations(seed_ids, top_n=12):
    _ensure_model()
    seed_ids = [str(s) for s in seed_ids]
    indices = [_id_to_index(sid) for sid in seed_ids if _id_to_index(sid) is not None]
    if not indices:
        return []

    sims = _sim_matrix[indices]
    mean_sim = sims.mean(axis=0)

    scores = list(enumerate(mean_sim))
    scores = [s for s in scores if s[0] not in indices]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    result_indices = [i for i, _ in scores]
    return _items_df.iloc[result_indices].to_dict(orient="records")


def get_trending_items(top_n=12):
    _ensure_model()
    return _items_df.sample(min(top_n, len(_items_df))).to_dict(orient="records")
