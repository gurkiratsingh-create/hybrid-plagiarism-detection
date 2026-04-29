import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize once (no repeated downloads)
lemmatizer = WordNetLemmatizer()
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    stop_words = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "were", "will", "with",
    }


def _safe_sentence_tokenize(text):
    try:
        return sent_tokenize(text)
    except LookupError:
        return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _safe_word_tokenize(sentence):
    try:
        return word_tokenize(sentence)
    except LookupError:
        return re.findall(r"\b\w+\b", sentence)


def _safe_lemmatize(word):
    try:
        return lemmatizer.lemmatize(word)
    except LookupError:
        return word


def preprocess(text, remove_stopwords=True):
    if not text or not isinstance(text, str):
        return [], []

    sentences = _safe_sentence_tokenize(text)
    cleaned = []

    for s in sentences:
        words = _safe_word_tokenize(s.lower())

        processed_words = []
        for w in words:
            if w.isalnum():  # keep only valid words
                if remove_stopwords and w in stop_words:
                    continue
                processed_words.append(_safe_lemmatize(w))

        cleaned_sentence = " ".join(processed_words)

        if cleaned_sentence.strip():   
            cleaned.append(cleaned_sentence)

    return sentences, cleaned
