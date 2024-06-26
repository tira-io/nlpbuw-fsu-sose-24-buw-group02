from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import json
import spacy
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def load_data(text_path, labels_path):
    texts = text_path.to_dict(orient='records')
    labels = labels_path.to_dict(orient='records')
    return texts, labels

def predict(nlp, texts):
    predictions = []
    for text in texts:
        doc = nlp(text['sentence'])
        tags = ['O'] * len(doc)
        for ent in doc.ents:
            for idx in range(ent.start, ent.end):
                prefix = 'I-'
                if idx == ent.start:
                    prefix = 'B-'
                tags[idx] = f"{prefix}{ent.label_.lower()}"
        predictions.append({'id': text['id'], 'tags': tags})
    return predictions

def evaluate(predictions, ground_truths):
    y_true = []
    y_pred = []

    gt_dict = {item['id']: item['tags'] for item in ground_truths}

    for pred in predictions:
        pred_tags = pred['tags']
        gt_tags = gt_dict.get(pred['id'], [])
        
        min_length = min(len(pred_tags), len(gt_tags))
        y_true.extend(gt_tags[:min_length])
        y_pred.extend(pred_tags[:min_length])
    
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    texts_validation, labels_validation = load_data(text_validation, targets_validation)
    output_dir = Path(__file__).parent / "ner_model"

    # Load the trained model
    nlp = spacy.load(output_dir)

    # Make predictions
    predictions = predict(nlp, texts_validation)

    # Save predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

    # Evaluate predictions
    precision, recall, f1, accuracy = evaluate(predictions, labels_validation)

    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
