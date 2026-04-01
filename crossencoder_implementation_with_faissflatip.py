import json, os
import time
import numpy as np
import faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
load_dotenv()

def load_embdeddings_model(model_name):
    return SentenceTransformer(model_name)

def load_cross_encoder_model(model_name):
    return CrossEncoder(model_name)

def load_index_model(index_path):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        return index
    else:
        raise FileNotFoundError(f"Index file not found at {index_path}")
    
def get_retrieved_abstracts(results, all_text):
    retrieved_abstracts_lookup = {}
    retrieved_texts = []
    for rid, score in results:
        print(f"ID: {rid}, Score: {score}")
        record = all_text.get(rid, "No abstract found for this ID.")
        abstract_from_id = "Title: " + record.get("title", "") + " Abstract: " + record.get("abstract", "")
        retrieved_abstracts_lookup[abstract_from_id] = rid
        retrieved_texts.append(abstract_from_id)
    return retrieved_texts, retrieved_abstracts_lookup

def predict(index, embd_model, query, id_index_mapping_lookup, top_k=5):
    query = "query: " + query
    query_embedding = embd_model.encode([query], normalize_embeddings=True)
    
    # Search the FAISS index for the nearest neighbors
    distances, indices = index.search(query_embedding, top_k)
    
    # Retrieve the corresponding IDs from the mapping
    results = []
    for (i, idx) in enumerate(indices[0]):
        if idx < len(id_index_mapping_lookup):
            results.append((id_index_mapping_lookup[idx], distances[0][i]))

    return results

def cross_encoder_rerank(query, retrieved_texts, cross_encoder_model, top_k=5):
    # Prepare pairs for cross-encoder
    pairs = [(query, text) for text in retrieved_texts]
    
    # Get relevance scores from the cross-encoder
    scores = cross_encoder_model.predict(pairs, batch_size=8)
    
    # Combine texts with their scores and sort by score
    scored_results = list(zip(retrieved_texts, scores))
    scored_results.sort(key=lambda x: x[1], reverse=True)
    
    # take top k results after reranking
    scored_results = scored_results[:top_k]
    return scored_results

def get_final_abstracts_for_query(reranked_results, retrieved_abstracts_lookup):
    seen_ids = set()
    final_abstracts = []
    for text, score in reranked_results:
        if score > 0:
            rid = retrieved_abstracts_lookup.get(text)
            if rid and not seen_ids.__contains__(rid):
                final_abstracts.append((rid, text, score))
                seen_ids.add(rid)
    return final_abstracts


if __name__ == "__main__":

    ###### Loading models and data ######
    model_loading_time_start = time.time()
    relative_path = os.getenv("FILE_PATH")
    embd_model = load_embdeddings_model("intfloat/e5-base-v2")
    cross_encoder_reranker_model = load_cross_encoder_model("cross-encoder/ms-marco-MiniLM-L-6-v2")

    index = load_index_model(relative_path + "/saved_models/faiss_indexflatip.index")

    with open(relative_path + "/text_data/id_text_mapping_bigrams.json", 'r') as f:
        id_index_mapping = json.load(f)
    id_index_mapping_lookup = {}
    for rid, indices in id_index_mapping.items():
        for idx in indices:
            id_index_mapping_lookup[idx] = rid

    with open(relative_path + "/text_data/id_content_mapping.json", 'r') as f:
        all_text = json.load(f)

    print(f"Model and data loading time: {time.time() - model_loading_time_start} seconds")

    ###### Prediction from FAISS ######
    query = "What simplified model is used to represent the atom–electromagnetic field system?"
    prediction_time_start = time.time()
    results = predict(index, embd_model, query, id_index_mapping_lookup, 15)
    print(f"Initial prediction time: {time.time() - prediction_time_start} seconds")
    print("Top results:", results)

    ###### Abstract retrieval and cross-encoder reranking ######
    abstract_retrieval_time_start = time.time()
    retrieved_texts, retrieved_abstracts_lookup = get_retrieved_abstracts(results, all_text)
    reranked_results = cross_encoder_rerank(query, retrieved_texts, cross_encoder_reranker_model)
    print(f"Abstract retrieval and reranking time: {time.time() - abstract_retrieval_time_start} seconds")
    print("Reranked results:", reranked_results)

    final_results = get_final_abstracts_for_query(reranked_results, retrieved_abstracts_lookup)
    print("Final abstracts for the query:")
    for rid, abstract, score in final_results:
        print(f"ID: {rid}, Score: {score}\nAbstract: {abstract}\n")

'''
---------------------------------------------------------------
Model and data loading time: 7.154691934585571 seconds
Initial prediction time: 0.27833104133605957 seconds
Abstract retrieval and reranking time: 0.21354198455810547 seconds
---------------------------------------------------------------
'''