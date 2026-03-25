import time
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from tqdm import tqdm

class HotpotQAPreprocessor:
    def __init__(self, subset_size=None):
        """
        Initialize the preprocessor.
        :param subset_size: If set (e.g., 1000), it limits the number of records processed 
                            for faster local testing. If None, it processes the full dev set.
        """
        self.subset_size = subset_size
        self.raw_data = None
        self.cleaned_docs = []
        self.chunked_docs = []
        self.vector_store = None

    def fetch_data(self):
        """Module 1: Fetch the HotpotQA development set"""
        print("\n--- [Module 1] Starting to fetch HotpotQA dataset ---")
        start_time = time.time()
        
        # 'distractor' split is excellent for evaluating RAG retrieval precision
        dataset = load_dataset('hotpot_qa', 'distractor', split='validation')
        
        if self.subset_size:
            dataset = dataset.select(range(self.subset_size))
            
        self.raw_data = dataset
        
        print(f"✅ Fetching completed! Time elapsed: {time.time() - start_time:.2f} seconds")
        print(f"📊 Statistics: Fetched {len(self.raw_data)} QA records (Queries).")
        return self.raw_data

    def clean_and_extract_corpus(self):
        """Module 2: Clean data and extract unique context passages for the retrieval corpus"""
        print("\n--- [Module 2] Starting data cleaning and corpus extraction ---")
        start_time = time.time()
        
        unique_passages = {}
        
        for item in tqdm(self.raw_data, desc="Parsing Context"):
            # HotpotQA context structure: lists of [title, [sentence1, sentence2, ...]]
            titles = item['context']['title']
            sentences_lists = item['context']['sentences']
            
            for title, sentences in zip(titles, sentences_lists):
                # Only add the passage if it hasn't been added yet (deduplication)
                if title not in unique_passages:
                    # Concatenate the list of sentences into a single full string
                    full_text = " ".join(sentences)
                    # Wrap metadata and text into a LangChain Document format
                    unique_passages[title] = Document(
                        page_content=full_text,
                        metadata={"title": title, "source": "hotpot_qa"}
                    )
        
        self.cleaned_docs = list(unique_passages.values())
        
        print(f"✅ Cleaning completed! Time elapsed: {time.time() - start_time:.2f} seconds")
        print(f"📊 Statistics: Extracted {len(self.cleaned_docs)} unique candidate documents.")
        return self.cleaned_docs

    def chunk_documents(self, chunk_size=500, chunk_overlap=50):
        """Module 3: Text Chunking"""
        print("\n--- [Module 3] Starting document chunking ---")
        start_time = time.time()
        
        # Utilize LangChain's RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        self.chunked_docs = text_splitter.split_documents(self.cleaned_docs)
        
        print(f"✅ Chunking completed! Time elapsed: {time.time() - start_time:.2f} seconds")
        print(f"📊 Statistics: Split original documents into {len(self.chunked_docs)} chunks (Settings: size={chunk_size}, overlap={chunk_overlap}).")
        return self.chunked_docs

    def build_embedding_and_index(self, model_name='BAAI/bge-small-en-v1.5'):
        """Module 4: Generate Embeddings and build the FAISS vector index"""
        print(f"\n--- [Module 4] Starting vector index build (Model: {model_name}) ---")
        start_time = time.time()
        
        # Initialize the Embedding model (bge-small is lightweight and highly effective)
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}, # Change to 'cuda' if running on a GPU
            encode_kwargs={'normalize_embeddings': True} 
        )
        
        print("⏳ Computing embeddings and building FAISS index, this may take a while...")
        self.vector_store = FAISS.from_documents(self.chunked_docs, embeddings)
        
        # Save the index locally
        save_path = "./hotpotqa_faiss_index"
        self.vector_store.save_local(save_path)
        
        print(f"✅ Index built and saved! Time elapsed: {time.time() - start_time:.2f} seconds")
        print(f"📊 Statistics: Vector database saved to '{save_path}', containing {self.vector_store.index.ntotal} vectors.")
        return self.vector_store

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    # Hint: Start with a subset (e.g., 500) to test the pipeline. 
    # Change subset_size to None to process the entire dataset once verified.
    preprocessor = HotpotQAPreprocessor(subset_size=500)
    
    # Execute the pipeline
    preprocessor.fetch_data()
    preprocessor.clean_and_extract_corpus()
    preprocessor.chunk_documents(chunk_size=512, chunk_overlap=50)
    preprocessor.build_embedding_and_index(model_name='BAAI/bge-small-en-v1.5')
    
    print("\n🎉 All preprocessing steps completed successfully! Your RAG knowledge base is ready.")