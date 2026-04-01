import json
import time
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def load_embeddings_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        print(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        raise


def data_processing_embeddings(file_path, embd_file_path, text_data_path):
    with open(file_path, 'r') as file:
        raw_data = json.load(file)

    title_abstracts = {}
    for data in raw_data:
        title_abstracts[data.get('id', '')] = 'Title: ' + data.get('title', '') + ', Abstract: ' + data.get('abstract', '')

    id_text_mapping = {}
    idx = 0
    all_text = []

    for rid, text in title_abstracts.items():
        tokenized_text = sent_tokenize(text)
        bigrams = []
        for i in range(len(tokenized_text) - 1):
            bigrams.append("passage: " + tokenized_text[i] + " " + tokenized_text[i+1])

        # Handle edge case (single sentence)
        if len(tokenized_text) == 1:
            bigrams.append(tokenized_text[0])

        all_text.extend(bigrams)

        id_text_mapping[rid] = list(range(idx, idx + len(bigrams)))
        idx += len(bigrams)

    model = load_embeddings_model('intfloat/e5-base-v2')

    data_embeddings = model.encode(all_text, batch_size=8, normalize_embeddings=True, show_progress_bar=True)

    with open(text_data_path + "/id_text_mapping_bigrams.json", 'w', encoding='utf-8') as f:
        json.dump(id_text_mapping, f)
    with open(text_data_path + "/all_text_bigrams.json", 'w', encoding='utf-8') as f:
        json.dump(all_text, f)
    with open(embd_file_path, 'w', encoding='utf-8') as f:
        json.dump(data_embeddings.tolist(), f)

    print("Embeddings generated and saved to bigrams_embeddings.json")

if __name__ == "__main__":
    relative_path = os.getenv("FILE_PATH")
    file_path = relative_path + "/top_10000_records_ai.json"
    embd_file_path = relative_path + "/bigrams_embeddings.json"
    text_data_folder_path = relative_path + "/text_data"
    start_time = time.time()
    data_processing_embeddings(file_path, embd_file_path, text_data_folder_path)
    print(f"Total time taken: {time.time() - start_time:.2f} seconds")



"""
---------------------------------------------------------
Embeddings generated and saved to bigrams_embeddings.json
Total time taken: 680.88 seconds
---------------------------------------------------------
"""