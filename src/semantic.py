from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"
_model = None


def get_model():
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(MODEL_NAME)
    return _model


def semantic_score(s1, s2):
    model = get_model()
    embeddings = model.encode([s1, s2], normalize_embeddings=True)
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return float(score)
