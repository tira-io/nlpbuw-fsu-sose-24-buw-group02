from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
import json
import re
import nltk
from nltk.corpus import stopwords
import os
import io

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    text = text.lower()
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Set up Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = '/workspaces/nlpbuw-fsu-sose-24-buw-group02/summarization/credentials.json'  # Path to your service account key file

credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=credentials)

def download_from_drive(folder_name, local_folder):
    response = drive_service.files().list(q=f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder'", spaces='drive').execute()
    folder = response.get('files', [])
    
    if not folder:
        raise ValueError("Folder not found in Google Drive")
    
    folder_id = folder[0].get('id')
    response = drive_service.files().list(q=f"'{folder_id}' in parents", spaces='drive').execute()
    files = response.get('files', [])
    
    if not os.path.exists(local_folder):
        os.makedirs(local_folder)
    
    for file in files:
        file_id = file.get('id')
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.FileIO(os.path.join(local_folder, file.get('name')), 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False:
            status, done = downloader.next_chunk()

# Define Google Drive directory name and local model directory
drive_model_dir = 'T5SummarizationModel'
local_model_dir = './model'

# Download the model directory from Google Drive
download_from_drive(drive_model_dir, local_model_dir)

# Load the model and tokenizer
tokenizer = T5Tokenizer.from_pretrained(local_model_dir)
model = T5ForConditionalGeneration.from_pretrained(local_model_dir)

# Load the data
tira = Client()
df = tira.pd.inputs(
    "nlpbuw-fsu-sose-24", "summarization-validation-20240530-training"
).set_index("id")

# Grab first two sentences
df["summary"] = df["story"].str.split("\n").str[:2].str.join("\n")
df = df.drop(columns=["story"]).reset_index()

# Save the predictions
output_directory = get_output_directory(str(Path(__file__).parent))
df.to_json(
    Path(output_directory) / "predictions.jsonl", orient="records", lines=True
)
