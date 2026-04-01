#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# ===================== RAG RETRIEVAL TEAM INTERFACE =====================
# Standardized interface to load preprocessing outputs and run retrieval
# Includes Query Rewrite, BM25 Sparse Search, and Contriever Dense Search
# Path adapted to repository root structure

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ===================== 1. Environment Configuration =====================
# NOTE: Path matches repository root structure (indexes/ folder is in the same directory as this file)
# No manual path change needed for team members after cloning the repo
PREPROCESSING_OUTPUT_PATH = r"./"
INDEX_PATH = os.path.join(PREPROCESSING_OUTPUT_PATH, "indexes")

# ===================== 2. Query Rewrite Module =====================
class QueryRewriter:
    """
    Query Rewrite module using lightweight small model (no hardcoding)
    Set use_mock=True to skip model loading for quick testing
    """
    def __init__(self, model_name="google/flan-t5-small", use_mock=False):
        self.use_mock = use_mock
        self.model = None
        self.tokenizer = None
        
        if not use_mock:
            try:
                from transformers import T5Tokenizer, T5ForConditionalGeneration
                self.tokenizer = T5Tokenizer.from_pretrained(
                    model_name, 
                    cache_dir=os.environ.get("HF_HOME", "D:\\huggingface_cache")
                )
                self.model = T5ForConditionalGeneration.from_pretrained(
                    model_name, 
                    cache_dir=os.environ.get("HF_HOME", "D:\\huggingface_cache")
                )
                print(f"✅ Query Rewrite model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load Query Rewrite model, switching to mock mode: {str(e)}")
                self.use_mock = True

    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite user query to improve retrieval performance
        No manual hardcoding rules - all logic handled by model
        """
        if self.use_mock:
            return f"{original_query} (retrieval-optimized)"
        
        prompt = f"""Rewrite the following user query to make it more specific, comprehensive, and suitable for document retrieval. 
        Do NOT change the original intent or meaning of the query. Only return the rewritten query, no extra explanations.
        Original query: {original_query}
        Rewritten query:"""
        
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
        outputs = self.model.generate(input_ids, max_length=128, num_beams=1, do_sample=False)
        rewritten_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🔄 Query Rewrite: Original → {original_query} | Rewritten → {rewritten_query}")
        return rewritten_query

# ===================== 3. Preprocessing Output Loader =====================
class PreprocessingOutputLoader:
    """
    Standardized loader to access all preprocessing deliverables
    - Loads BM25 sparse index
    - Loads Contriever dense index
    - Integrates query rewrite
    """
    def __init__(self, output_base_path, enable_query_rewrite=True, use_mock_rewrite=False):
        self.base_path = output_base_path
        self.index_path = os.path.join(self.base_path, "indexes")
        self.meta_path = os.path.join(self.index_path, "chunk_metadata_final.pkl")

        # Load BM25 Sparse Index
        print("--- Loading BM25 Sparse Index ---")
        with open(os.path.join(self.index_path, "bm25_index_final.pkl"), "rb") as f:
            self.bm25_data = pickle.load(f)
            self.bm25_model = self.bm25_data["bm25_model"]
            self.all_chunks = self.bm25_data["chunks"]
            self.metadata = self.bm25_data["metadata"]
        print(f"✅ BM25 index loaded: {len(self.all_chunks)} chunks")

        # Load Contriever Dense Index
        print("--- Loading Contriever Dense Index ---")
        self.dense_index = faiss.read_index(os.path.join(self.index_path, "dense_retrieval_index_final.faiss"))
        with open(self.meta_path, "rb") as f:
            self.full_metadata = pickle.load(f)
        print(f"✅ Contriever index loaded: {self.dense_index.ntotal} vectors")

        # Initialize Query Rewrite Module
        self.query_rewriter = QueryRewriter(use_mock=use_mock_rewrite) if enable_query_rewrite else None
        print("✅ Preprocessing outputs fully loaded")

    def rewrite_query(self, original_query: str) -> str:
        return self.query_rewriter.rewrite_query(original_query) if self.query_rewriter else original_query

    def sparse_search(self, query, top_k=5, use_rewrite=True):
        """
        BM25 Sparse Retrieval
        :param query: Original user query
        :param top_k: Number of top results to return
        :param use_rewrite: Whether to apply query rewrite
        :return: List of retrieval results
        """
        if use_rewrite and self.query_rewriter:
            query = self.rewrite_query(query)
        
        scores = self.bm25_model.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "rank": idx+1,
                "chunk": self.all_chunks[i],
                "metadata": self.metadata[i],
                "bm25_score": scores[i]
            } for idx, i in enumerate(top_indices)
        ]

    def dense_search(self, query, model, top_k=5, use_rewrite=True):
        """
        Contriever Dense Retrieval
        :param query: Original user query
        :param model: Loaded Contriever SentenceTransformer model
        :param top_k: Number of top results to return
        :param use_rewrite: Whether to apply query rewrite
        :return: List of retrieval results
        """
        if use_rewrite and self.query_rewriter:
            query = self.rewrite_query(query)
        
        query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
        distances, indices = self.dense_index.search(query_embedding, top_k)
        return [
            {
                "rank": idx+1,
                "chunk": self.all_chunks[i],
                "metadata": self.full_metadata[i],
                "faiss_distance": float(distances[0][idx])
            } for idx, i in enumerate(indices[0])
        ]

# ==========================================
# Example Usage for Retrieval Team
# ==========================================
if __name__ == "__main__":
    # 1. Load preprocessing outputs
    # Set use_mock_rewrite=True to skip model loading for quick testing
    loader = PreprocessingOutputLoader(
        output_base_path=PREPROCESSING_OUTPUT_PATH,
        enable_query_rewrite=True,
        use_mock_rewrite=False
    )

    # 2. Load Contriever Model (only needed for dense retrieval)
    print("\n--- Loading Contriever Model ---")
    contriever_model = SentenceTransformer(
        "facebook/contriever-msmarco", 
        cache_folder=os.environ.get("HF_HOME", "D:\\huggingface_cache")
    )
    print("✅ Contriever model loaded")

    # 3. User Query
    user_query = "What is RAG technology and how does it work?"

    # 4. BM25 Sparse Retrieval
    print("\n" + "="*80)
    print("BM25 SPARSE RETRIEVAL RESULTS")
    print("="*80)
    sparse_results = loader.sparse_search(user_query, top_k=5)
    for res in sparse_results:
        print(f"\n[Rank {res['rank']}] (Score: {res['bm25_score']:.4f})")
        print(f"Chunk: {res['chunk'][:200]}...")
        print(f"Source: {res['metadata']['dataset']}")

    # 5. Contriever Dense Retrieval
    print("\n" + "="*80)
    print("CONTRIEVER DENSE RETRIEVAL RESULTS")
    print("="*80)
    dense_results = loader.dense_search(user_query, contriever_model, top_k=5)
    for res in dense_results:
        print(f"\n[Rank {res['rank']}] (Distance: {res['faiss_distance']:.4f})")
        print(f"Chunk: {res['chunk'][:200]}...")
        print(f"Source: {res['metadata']['dataset']}")

