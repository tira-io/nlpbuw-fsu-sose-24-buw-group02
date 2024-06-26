from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random

def load_data(text_path, labels_path):
    texts = text_path.to_dict(orient='records')
    labels = labels_path.to_dict(orient='records')
    return texts, labels

def create_spacy_format(texts, labels):
    spacy_data = []
    for text, label in zip(texts, labels):
        doc_text = text['sentence']
        tags = label['tags']
        entities = []
        tokens = doc_text.split()
        start = 0
        for token, tag in zip(tokens, tags):
            end = start + len(token)
            if tag != 'O':
                entities.append((start, end, tag.split('-')[1].upper()))
            start = end + 1  # Account for the space
        spacy_data.append((doc_text, {'entities': entities}))
    return spacy_data

def train_spacy_model(train_data, output_dir):
    nlp = spacy.blank('en')
    ner = nlp.create_pipe('ner')
    nlp.add_pipe('ner', last=True)
    
    for _, annotations in train_data:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    
    optimizer = nlp.begin_training()
    for itn in range(10):  # Number of training iterations
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.5))
        for batch in batches:
            for text, annotations in batch:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], sgd=optimizer, drop=0.35, losses=losses)
        print(f"Iteration {itn} - Losses: {losses}")
    
    nlp.to_disk(output_dir)

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "ner-validation-20240612-training"
    )

    texts_train, labels_train = load_data(text_validation, targets_validation)
    train_data = create_spacy_format(texts_train, labels_train)

    print(text_validation.head)
    print(targets_validation.head)

    # Train the model
    output_dir = Path(__file__).parent / "ner_model"
    train_spacy_model(train_data, output_dir)
