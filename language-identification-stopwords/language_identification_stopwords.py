from pathlib import Path
import re
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack
import numpy as np
from tqdm import tqdm
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

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    # Important: please note that if this data is changed to test data, then the model will be trained on that test data and then predictions will be made
    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
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

    # Merge text data with labels on 'id'
    data = text_validation.merge(targets_validation, on='id')

    print(text_validation.head())
    print(targets_validation.head())
    print(data.head())

    # Spliting the data into training and validation sets
    # This split is not necessary but since I run this in local, I wanted to check the val_accuracy
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    # Stopword features for training and validation sets
    stopword_features_train = stopword_features(train_data['text'], stopwords)
    stopword_features_val = stopword_features(val_data['text'], stopwords)

    # Initializing TfidfVectorizer for character n-grams and word n-grams
    char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
    word_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000)

    # TF-IDF features for training and validation sets
    char_features_train = char_vectorizer.fit_transform(train_data['text'])
    word_features_train = word_vectorizer.fit_transform(train_data['text'])
    char_features_val = char_vectorizer.transform(val_data['text'])
    word_features_val = word_vectorizer.transform(val_data['text'])

    # Additional text features for training and validation sets
    additional_features_train = TextFeatures().fit_transform(train_data['text'])
    additional_features_val = TextFeatures().transform(val_data['text'])

    # Combining all features into a single DataFrame for training and validation sets
    combined_features_train = hstack([char_features_train, word_features_train, additional_features_train, stopword_features_train])
    combined_features_val = hstack([char_features_val, word_features_val, additional_features_val, stopword_features_val])

    # Training a classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(combined_features_train, train_data['lang'])

    # Prediction on the validation data
    val_predictions = clf.predict(combined_features_val)

    # Calculating validation accuracy
    val_accuracy = accuracy_score(val_data['lang'], val_predictions)
    print(f'Validation Accuracy: {val_accuracy:.4f}')

    # Save the model
    dump(clf, Path(__file__).parent / "model.joblib")
    dump(char_vectorizer, Path(__file__).parent / "char_vectorizer.joblib")
    dump(word_vectorizer, Path(__file__).parent / "word_vectorizer.joblib")
