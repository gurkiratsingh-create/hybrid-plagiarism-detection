import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize once (no repeated downloads)
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(text, remove_stopwords=True):
    if not text or not isinstance(text, str):
        return [], []

    sentences = sent_tokenize(text)
    cleaned = []

    for s in sentences:
        words = word_tokenize(s.lower())

        processed_words = []
        for w in words:
            if w.isalnum():  # keep only valid words
                if remove_stopwords and w in stop_words:
                    continue
                processed_words.append(lemmatizer.lemmatize(w))

        cleaned_sentence = " ".join(processed_words)

        if cleaned_sentence.strip():   
            cleaned.append(cleaned_sentence)

    return sentences, cleaned