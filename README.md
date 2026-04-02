<h1>Advanced RAG System with FAISS + Cross-Encoder Reranking</h1>

<h3>Overview</h3>

This project implements an advanced Retrieval-Augmented Generation (RAG) pipeline for semantic search over scientific text data. It combines:<br>
* Dense vector retrieval using FAISS<br>
* Efficient indexing strategy IndexFlatIP<br>
* Cross-encoder reranking for improved relevance<br>

The system is designed to simulate production-grade GenAI pipelines used in real-world applications like search engines, QA systems, and AI assistants.

<h3>Architecture</h3>
<p align="center">
User Query <br> ↓ <br>
Embedding Model (E5) <br> ↓ <br>
FAISS Vector Search (Top-K Retrieval) <br> ↓ <br>
Cross-Encoder Reranking <br> ↓ <br>
Final Ranked Results
</p>

<h3>Tech Stack</h3>
* Python<br>
* FAISS (Facebook AI Similarity Search)<br>
* Sentence Transformers<br>
* Cross-Encoders (MS MARCO)<br>
* NumPy<br>

<h3>Retrieval Strategy</h3>
<h4>1. Dense Embedding Model</h4>
* Embeddings model used:
'intfloat/e5-base-v2'<br>
* Converts text into dense semantic vectors<br>
* Uses query/passage prefixing for better alignment<br>


<h4>2. FAISS Indexing</h4>
<h5>IndexFlatIP (Inner Product)</h5>
* Computes dot product similarity<br>
* Equivalent to cosine similarity if vectors are normalized<br>
* Works best with normalized embeddings<br>
* Preferred for semantic search tasks<br>
* Faster and more aligned with transformer embeddings<br>

<h4>Cross-Encoder Reranking</h4>
* Model Used - 
cross-encoder/ms-marco-MiniLM-L-6-v2<br>
Why Cross-Encoder?

- **Bi-encoder (FAISS retrieval)**
  - Fast
  - Less precise

- **Cross-encoder**
  - Slower but highly accurate
  - Evaluates query and document together

<h4>Pipeline Strategy</h4>
* Retrieve top K = 50 documents using FAISS<br>
* Rerank using cross-encoder<br>
* Select top K = 5–10

- **Benefit**
  - Improves ranking quality significantly
  - Fixes semantic mismatches from vector search

<h4>Data Processing</h4>
Input data: Scientific articles (title + abstract)<br>
Chunking: Sentence-level segmentation<br>
Mapping:<br>
 - id → text<br>
 - index → id<br>

<h4>Key Features</h4>
   - Efficient FAISS-based retrieval<br>
   - Cross-encoder reranking<br>
   - Scalable architecture<br>
   - Modular design<br>

<h4>Example Query</h4>  
"What simplified model is used to represent the atom–electromagnetic field system?"<br>
* Output:<br>
- Top relevant research abstracts<br>
- Reranked based on semantic relevance<br>

<h4>Future Improvements</h4>
- Hybrid search (BM25 + Dense)<br>
- Query expansion using LLMs<br>
- Evaluation metrics (Recall@K, MRR)<br>
- Integration with LLM for answer generation<br>
