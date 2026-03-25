import time
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def run_retrieval_demo():
    print("\n--- [Step 1] Initializing Embedding Model ---")
    # Must use the EXACT same model used during preprocessing
    model_name = 'BAAI/bge-small-en-v1.5'
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    print(f"✅ Embedding model '{model_name}' loaded successfully.")

    print("\n--- [Step 2] Loading Local FAISS Index ---")
    index_path = "./hotpotqa_faiss_index"
    
    # Note: allow_dangerous_deserialization=True is required in recent LangChain 
    # versions to load local .pkl files safely in your own environment.
    vector_store = FAISS.load_local(
        folder_path=index_path, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    print(f"✅ FAISS index loaded from '{index_path}'. Total vectors: {vector_store.index.ntotal}")

    print("\n--- [Step 3] Processing the Query ---")
    # Sample multi-hop query typical for HotpotQA
    query = "Who is the director of the movie Inception, and what is his nationality?"
    print(f"🔍 Query: '{query}'")
    
    start_time = time.time()
    # Embed the query using the same preprocessing as the documents
    query_embedding = embeddings.embed_query(query)
    
    # Perform similarity search to get top-K documents and their distance scores (L2 distance)
    # Lower score means higher similarity in FAISS L2 search
    k = 5
    results = vector_store.similarity_search_with_score_by_vector(query_embedding, k=k)
    print(f"✅ Retrieval completed in {time.time() - start_time:.4f} seconds.")

    print(f"\n--- [Step 4] Top-{k} Retrieval Results ---")
    for rank, (doc, score) in enumerate(results, start=1):
        print(f"\n{'='*50}")
        print(f"🏆 Rank: {rank} | L2 Distance Score: {score:.4f}")
        print(f"{'='*50}")
        
        # 1. Output Metadata
        print("📌 [Metadata]:")
        for key, value in doc.metadata.items():
            print(f"   - {key}: {value}")
            
        # 2. Output Document Text
        print("\n📄 [Document Content]:")
        # Truncating the text slightly for console readability if it's too long
        content_preview = doc.page_content if len(doc.page_content) < 300 else doc.page_content[:300] + "..."
        print(f"   {content_preview}")
        
        # 3. Output Embedding (Demonstration)
        # To avoid flooding the console with 384 dimensions, we re-embed the text 
        # (or extract from FAISS) and show the first 5 vector dimensions as proof.
        doc_embedding = embeddings.embed_query(doc.page_content)
        print(f"\n🧬 [Embedding Vector (First 5 dimensions of {len(doc_embedding)})]:")
        preview_vector = [round(val, 6) for val in doc_embedding[:5]]
        print(f"   {preview_vector} ...")

if __name__ == "__main__":
    run_retrieval_demo()