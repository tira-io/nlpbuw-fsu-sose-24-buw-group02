from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
from joblib import load
import pandas as pd
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Class for extracting additional text features
class TextFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame({
            'avg_word_length': X.apply(lambda text: sum(len(word) for word in text.split()) / len(text.split())),
            'num_sentences': X.apply(lambda text: len(re.split(r'[.!?]', text))),
            'num_words': X.apply(lambda text: len(text.split())),
            'char_count': X.apply(lambda text: len(text)),
        })

# Function to extract stopword features
def stopword_features(text_series, stopwords):
    stopword_fractions = []
    for lang_id, lang_stopwords in stopwords.items():
        counts = pd.Series(0, index=text_series.index, name=lang_id)
        for stopword in lang_stopwords:
            counts += text_series.str.contains(fr'\b{stopword}\b', regex=True, case=False).astype(int)
        stopword_fractions.append(counts / len(lang_stopwords))
    stopword_fractions = pd.concat(stopword_fractions, axis=1)
    return stopword_fractions

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"language-identification-validation-20240429-training"
    )

    lang_ids = [
        "af",
        "az",
        "bg",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "hr",
        "it",
        "ko",
        "nl",
        "no",
        "pl",
        "ru",
        "ur",
        "zh",
    ]

    stopwords = {
        lang_id: set(
            (Path(__file__).parent / "stopwords" / f"stopwords-{lang_id}.txt")
            .read_text()
            .splitlines()
        )
        - set(("(", ")", "*", "|", "+", "?"))  # remove regex special characters
        for lang_id in lang_ids
    }

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")

    # Initializing TfidfVectorizer for character n-grams and word n-grams
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000)

    # Predict on the validation data
    stopword_features_val = stopword_features(df['text'], stopwords)
    char_features_val = char_vectorizer.transform(df['text'])
    word_features_val = word_vectorizer.transform(df['text'])
    additional_features_val = TextFeatures().transform(df['text'])
    combined_features_val = hstack([char_features_val, word_features_val, additional_features_val, stopword_features_val])

    predictions = model.predict(combined_features_val)

    # Prepare the predictions DataFrame
    predictions_df = pd.DataFrame({'id': df['id'], 'lang': predictions})

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    predictions_df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
