from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from tira.rest_api_client import Client

def custom_preprocessor(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = "".join([char for char in text if char.isalnum() or char.isspace()])
    return text

def custom_tokenizer(text):
    # Split text into words
    words = text.split()
    return words

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    df = text.join(labels.set_index("id"))

    # Split the data into training, validation, and test sets
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2

    # Define and train the model pipeline
    model_pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(
            tokenizer=custom_tokenizer,
            preprocessor=custom_preprocessor,
            stop_words='english',  # Specify 'english' to use built-in English stop words
            strip_accents='unicode'
        )),
        ('classifier', LinearSVC())  # Linear Support Vector Classifier
    ])

    # Train the model
    model_pipeline.fit(train_data["text"], train_data["generated"])

    # Evaluate on validation set
    val_accuracy = model_pipeline.score(val_data["text"], val_data["generated"])
    print(f"Validation Accuracy: {val_accuracy}")

    # Save the model
    dump(model_pipeline, Path(__file__).parent / "model.joblib")
