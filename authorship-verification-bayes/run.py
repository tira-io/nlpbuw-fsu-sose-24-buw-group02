from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

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
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", f"authorship-verification-validation-20240408-training"
    )

    # Load the model and make predictions
    model = load(Path(__file__).parent / "model.joblib")
    predictions = model.predict(df["text"])
    df["generated"] = predictions
    df = df[["id", "generated"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
