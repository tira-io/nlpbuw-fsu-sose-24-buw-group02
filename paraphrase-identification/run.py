from pathlib import Path
import joblib
import pandas as pd
from levenshtein import levenshtein_distance
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def extract_features(df):
    df["len_sentence1"] = df["sentence1"].apply(len)
    df["len_sentence2"] = df["sentence2"].apply(len)
    df["word_count1"] = df["sentence1"].apply(lambda x: len(x.split()))
    df["word_count2"] = df["sentence2"].apply(lambda x: len(x.split()))
    return df

if __name__ == "__main__":

    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Compute the Levenshtein distance
    df["distance"] = levenshtein_distance(df)
    
    # Extract additional features
    df = extract_features(df)
    
    # Load the model and make predictions
    model = joblib.load(Path(__file__).parent / "model.joblib")
    
    # Predict using the model
    feature_columns = ["distance", "len_sentence1", "len_sentence2", "word_count1", "word_count2"]
    df["label"] = model.predict(df[feature_columns])
    
    # Prepare the output DataFrame
    output_df = df.drop(columns=["distance", "sentence1", "sentence2", "len_sentence1", "len_sentence2", "word_count1", "word_count2"]).reset_index()
    
    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
