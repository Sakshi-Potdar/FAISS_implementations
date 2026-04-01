from flask import Flask, request, jsonify, render_template
import json, os, time
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

app = Flask(__name__)
load_dotenv()

################ LOAD MODELS ################
def load_models():
    global embd_model, cross_encoder_model, index
    global id_index_mapping_lookup, all_text

    relative_path = os.getenv("FILE_PATH")

    embd_model = SentenceTransformer("intfloat/e5-base-v2")
    cross_encoder_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    index = faiss.read_index(relative_path + "/saved_models/faiss_indexflatip.index")

    # Load mapping
    with open(relative_path + "/text_data/id_text_mapping_bigrams.json", 'r') as f:
        id_index_mapping = json.load(f)

    id_index_mapping_lookup = {}
    for rid, indices in id_index_mapping.items():
        for idx in indices:
            id_index_mapping_lookup[idx] = rid

    # Load content
    with open(relative_path + "/text_data/id_content_mapping.json", 'r') as f:
        all_text = json.load(f)

    print("✅ Models loaded successfully")

################ CORE FUNCTIONS ################

def predict(query, top_k=15):
    query = "query: " + query
    query_embedding = embd_model.encode([query], normalize_embeddings=True)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx in id_index_mapping_lookup:
            rid = id_index_mapping_lookup[idx]
            results.append((rid, float(distances[0][i])))

    return results


def get_retrieved_abstracts(results):
    retrieved_lookup = {}
    retrieved_texts = []

    for rid, score in results:
        record = all_text.get(rid, {})
        text = f"Title: {record.get('title','')} Abstract: {record.get('abstract','')}"
        retrieved_lookup[text] = rid
        retrieved_texts.append(text)

    return retrieved_texts, retrieved_lookup


def cross_encoder_rerank(query, retrieved_texts, top_k=5):
    pairs = [(query, text) for text in retrieved_texts]
    scores = cross_encoder_model.predict(pairs, batch_size=8)

    scored = list(zip(retrieved_texts, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


def get_final_results(reranked, lookup, all_text):
    seen = set()
    final = []

    for text, score in reranked:
        if score > 0:
            rid = lookup.get(text)
            if rid and rid not in seen:
                id_content = all_text.get(rid, {})
                final.append({
                    "id": rid,
                    "title": id_content.get("title", ""),
                    "authors": id_content.get("authors", ""),
                    "abstract": id_content.get("abstract", ""),
                    "score": float(score)
                })
                seen.add(rid)

    return final


################ ROUTES ################

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def get_prediction():
    data = request.json
    query = data.get("query", "")

    start = time.time()

    results = predict(query)
    retrieved_texts, lookup = get_retrieved_abstracts(results)
    reranked = cross_encoder_rerank(query, retrieved_texts)
    final_results = get_final_results(reranked, lookup, all_text)

    return jsonify({
        "query": query,
        "results": final_results,
        "time_taken": round(time.time() - start, 3)
    })


################ MAIN ################

if __name__ == "__main__":
    load_models()
    app.run(debug=True)