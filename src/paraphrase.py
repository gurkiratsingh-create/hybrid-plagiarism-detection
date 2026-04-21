from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(ngram_range=(1,2))
_is_fitted = False

def fit_vectorizer(all_texts):
    global _is_fitted
    vectorizer.fit(all_texts)
    _is_fitted = True

def paraphrase_score(s1, s2):
    if not _is_fitted:
        raise Exception("TF-IDF not fitted")

    if not s1.strip() or not s2.strip():
        return 0.0

    vec1 = vectorizer.transform([s1])
    vec2 = vectorizer.transform([s2])

    return cosine_similarity(vec1, vec2)[0][0]