import json, numpy as np
import os
from dotenv import load_dotenv
import faiss
from sentence_transformers import SentenceTransformer
import time

load_dotenv()

def load_embeddings(file_path):
    with open(file_path, 'r') as f:
        embeddings = json.load(f)
    return np.array(embeddings).astype("float32")

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_id_text_mapping_lookup(id_text_mapping_file_path):
    id_text_mapping = load_json(id_text_mapping_file_path)
    lookup = {}
    for rid, indices in id_text_mapping.items():
        for idx in indices:
            lookup[idx] = rid
    return lookup

def load_embeddings_model(model_name):
    try:
        model = SentenceTransformer(model_name)
        print(f"Model '{model_name}' loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model '{model_name}': {e}")
        raise

def train_model(model_path, embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    print("FAISS index ready:", index.ntotal)
    faiss.write_index(index, model_path)
    print(f"Model trained and saved to {model_save_path}")

    return index

def predict(query, model, index, all_text, id_text_mapping_lookup, top_k=5):
    query = "query: " + query
    query_embedding = model.encode([query], normalize_embeddings=True)
    D, I = index.search(query_embedding.astype("float32"), top_k)
    
    retrieved_texts = []
    for idx in I[0]:
        retrieved_texts.append(all_text[idx])
        
    return retrieved_texts

if __name__ == "__main__":
    relative_path = os.getenv("FILE_PATH")
    embd_file_path = relative_path + "/bigrams_embeddings.json"
    embeddings = load_embeddings(embd_file_path)
    print(f"Loaded {len(embeddings)} embeddings from {embd_file_path}")

    all_text = load_json(relative_path + "/text_data/all_text_bigrams.json")
    id_text_mapping_lookup = create_id_text_mapping_lookup(relative_path + "/text_data/id_text_mapping_bigrams.json")

    start_time = time.time()
    model_save_path = relative_path + "/saved_models/faiss_indexflatip.index"
    index = train_model(model_save_path, embeddings)
    print(f"Model training time: {time.time() - start_time} seconds")

    embd_model_name = 'intfloat/e5-base-v2'

    query = "What simplified model is used to represent the atom–electromagnetic field system?"

    embedding_model = load_embeddings_model(embd_model_name)
    retrieved_texts = predict(query, embedding_model, index, all_text, id_text_mapping_lookup)
    print("Retrieved Texts:", retrieved_texts)


'''
----------------------------------------------------------
FAISS index ready: 42845
Model training time: 0.054754018783569336 seconds
----------------------------------------------------------
'''