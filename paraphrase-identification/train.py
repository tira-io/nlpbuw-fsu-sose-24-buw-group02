from tira.rest_api_client import Client

from levenshtein import levenshtein_distance

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import matthews_corrcoef
import joblib

def extract_features(df):
    df["len_sentence1"] = df["sentence1"].apply(len)
    df["len_sentence2"] = df["sentence2"].apply(len)
    df["word_count1"] = df["sentence1"].apply(lambda x: len(x.split()))
    df["word_count2"] = df["sentence2"].apply(lambda x: len(x.split()))
    return df

if __name__ == "__main__":

    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    
    # Compute the Levenshtein distance
    text["distance"] = levenshtein_distance(text)
    df = text.join(labels)

    # Extract additional features
    df = extract_features(df)
    
    # Prepare features and labels
    feature_columns = ["distance", "len_sentence1", "len_sentence2", "word_count1", "word_count2"]
    X = df[feature_columns]
    y = df["label"]
    
    # Split the data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model with Grid Search for hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    }
    
    model = GradientBoostingClassifier()
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)
    
    # Best model
    best_model = grid_search.best_estimator_
    
    # Validate the model
    y_pred = best_model.predict(X_val)
    mcc = matthews_corrcoef(y_val, y_pred)
    print(f"Validation MCC: {mcc}")
    
    # Save the trained model
    joblib.dump(best_model, "model.joblib")
