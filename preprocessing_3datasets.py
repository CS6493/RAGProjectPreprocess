#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ===================== RAG PROJECT PREPROCESSING PIPELINE - 3 DATASETS FINAL VERSION =====================
# 100% Compliant with Project Requirements: HotpotQA(dev) + PubMedQA + FinanceBench | No NQ Dataset
# ===================== All Required Imports =====================
import pandas as pd
import numpy as np
import os
import pickle
import re
import time
import gc
import json
from tqdm import tqdm
from collections import defaultdict

# Hugging Face Datasets & Model Libraries
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

# ===================== Environment & Cache Configuration =====================
os.environ["HF_HOME"] = "D:\\huggingface_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ===================== 1. BM25 Sparse Retrieval Class =====================
class SimpleBM25:
    def __init__(self, corpus):
        self.corpus = corpus
        self.tokenized_corpus = [doc.lower().split() for doc in corpus]
        self.tokens = self.tokenized_corpus
        self.avgdl = sum(len(t) for t in self.tokenized_corpus) / len(self.tokenized_corpus)
        self.k1 = 1.5
        self.b = 0.75
        self.dfs = []
        self.tf = {}
        for doc in self.tokenized_corpus:
            df = {}
            for term in doc:
                df[term] = df.get(term, 0) + 1
                self.tf[term] = self.tf.get(term, 0) + 1
            self.dfs.append(df)
    
    def get_scores(self, query):
        qtok = query.lower().split()
        scores = []
        n = len(self.corpus)
        for i, doc in enumerate(self.tokens):
            s = 0
            dl = len(doc)
            for t in qtok:
                tf = self.dfs[i].get(t, 0)
                idf = np.log((n - self.tf.get(t, 0) + 0.5) / (self.tf.get(t, 0) + 0.5) + 1)
                s += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            scores.append(s)
        return scores

# ===================== 2. Query Rewrite Module (Small Model, No Hardcoding) =====================
class QueryRewriter:
    def __init__(self, model_name="google/flan-t5-small", use_mock=False):
        self.use_mock = use_mock
        self.model = None
        self.tokenizer = None
        
        if not use_mock:
            try:
                from transformers import T5Tokenizer, T5ForConditionalGeneration
                self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=os.environ["HF_HOME"])
                self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=os.environ["HF_HOME"])
                print(f"✅ Query Rewrite small model loaded successfully")
            except Exception as e:
                print(f"⚠️ Failed to load Query Rewrite model, switching to mock mode: {str(e)}")
                self.use_mock = True

    def rewrite_query(self, original_query: str) -> str:
        if self.use_mock:
            return f"{original_query} (retrieval-optimized mock version)"
        
        prompt = f"""Rewrite the following user query to make it more specific, comprehensive, and suitable for document retrieval. 
        Do NOT change the original intent or meaning of the query. Only return the rewritten query, no extra explanations.
        Original query: {original_query}
        Rewritten query:"""
        
        input_ids = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).input_ids
        outputs = self.model.generate(input_ids, max_length=128, num_beams=1, do_sample=False)
        rewritten_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"🔄 Query Rewrite: Original → {original_query} | Rewritten → {rewritten_query}")
        return rewritten_query

# ===================== 3. Standardized Retrieval Interface =====================
class PreprocessingOutputLoader:
    def __init__(self, output_base_path, enable_query_rewrite=True, use_mock_rewrite=False):
        self.base_path = output_base_path
        self.index_path = os.path.join(self.base_path, "indexes")
        self.meta_path = os.path.join(self.index_path, "chunk_metadata_final.pkl")

        with open(os.path.join(self.index_path, "bm25_index_final.pkl"), "rb") as f:
            self.bm25_data = pickle.load(f)
            self.bm25_model = self.bm25_data["bm25_model"]
            self.all_chunks = self.bm25_data["chunks"]
            self.metadata = self.bm25_data["metadata"]
        
        self.dense_index = faiss.read_index(os.path.join(self.index_path, "dense_retrieval_index_final.faiss"))
        with open(self.meta_path, "rb") as f:
            self.full_metadata = pickle.load(f)

        self.query_rewriter = QueryRewriter(use_mock=use_mock_rewrite) if enable_query_rewrite else None

    def rewrite_query(self, original_query: str) -> str:
        return self.query_rewriter.rewrite_query(original_query) if self.query_rewriter else original_query

    def sparse_search(self, query, top_k=5, use_rewrite=True):
        if use_rewrite and self.query_rewriter:
            query = self.rewrite_query(query)
        
        scores = self.bm25_model.get_scores(query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [{"chunk": self.all_chunks[i], "metadata": self.metadata[i], "score": scores[i]} for i in top_indices]

    def dense_search(self, query, model, top_k=5, use_rewrite=True):
        if use_rewrite and self.query_rewriter:
            query = self.rewrite_query(query)
        
        query_embedding = model.encode(query, convert_to_numpy=True).reshape(1, -1)
        distances, indices = self.dense_index.search(query_embedding, top_k)
        return [{"chunk": self.all_chunks[i], "metadata": self.full_metadata[i], "distance": distances[0][idx]} for idx, i in enumerate(indices[0])]

# ===================== Core Configuration =====================
BASE_PROJECT_PATH = r"D:\CS6493_project"
OUTPUT_BASE_PATH = os.path.join(BASE_PROJECT_PATH, "preprocessing_final_3datasets")

# Create organized output directories
os.makedirs(OUTPUT_BASE_PATH, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_PATH, "dataset_statistics"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_PATH, "cleaning_reports"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_PATH, "chunking_results"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE_PATH, "indexes"), exist_ok=True)

# ===================== 3 DATASETS (NO NQ) - FULLY COMPLIANT WITH PROJECT REQUIREMENTS =====================
DATASET_CONFIGS = {
    # HotpotQA: Development Dataset (validation set full 7405 records, no limit by default)
    "hotpotqa": {
        "hf_id": "hotpot_qa",
        "hf_name": "distractor",
        "split": "validation",
        "context_field": "context",
        "question_field": "question",
        "answer_field": "answer",
        "subset_size": None  # None = use full dataset, set to 1000 to limit for testing
    },
    # PubMedQA: Multi-domain Control Dataset (labeled set full 1000 records)
    "pubmedqa": {
        "hf_id": "qiaojin/PubMedQA",
        "hf_name": "pqa_labeled",
        "split": "train",
        "context_field": "context",
        "question_field": "question",
        "answer_field": "final_decision",
        "subset_size": None  # None = use full dataset
    },
    # FinanceBench: Multi-domain Control Dataset (full 150 records)
    "financebench": {
        "hf_id": "PatronusAI/financebench",
        "hf_name": None,
        "split": "train",
        "context_field": "context",
        "question_field": "question",
        "answer_field": "answer",
        "subset_size": None  # None = use full dataset
    }
}

# Chunking parameter grid
CHUNK_SIZE_GRID = [256, 512, 1024]
CHUNK_OVERLAP_GRID = [25, 51, 102]
EMBED_MODEL_NAME = "facebook/contriever-msmarco"
BATCH_SIZE = 100

# ===================== Full Preprocessing Pipeline =====================
class FullPreprocessingPipeline:
    def __init__(self, project_path=BASE_PROJECT_PATH, output_path=OUTPUT_BASE_PATH):
        self.project_path = project_path
        self.output_path = output_path
        self.raw_datasets = {}
        self.cleaned_datasets = {}
        self.cleaning_reports = {}
        self.removed_records = {}
        self.chunking_results = defaultdict(dict)
        self.chunking_stats = []
        self.final_chunks = []
        self.final_metadata = []
        self.bm25_index = None
        self.contriever_index = None

    def _generate_mock_data(self, n_samples=100):
        return pd.DataFrame({
            "context": [f"This is a mock document {i} about RAG systems, retrieval models, and preprocessing pipelines." for i in range(n_samples)],
            "question": [f"What is mock question {i}?" for i in range(n_samples)],
            "answer": [f"Mock answer {i}" for i in range(n_samples)]
        })

    def load_and_analyze_datasets(self):
        print("\n" + "="*80)
        print("MODULE 1: DATA LOADING & INITIAL STATISTICS (3 DATASETS)")
        print("="*80)

        for dataset_name, config in DATASET_CONFIGS.items():
            start_time = time.time()
            print(f"\n--- Loading {dataset_name} from Hugging Face: {config['hf_id']} ---")

            try:
                # Load dataset from Hugging Face Hub
                if config["hf_name"]:
                    dataset = load_dataset(config["hf_id"], config["hf_name"], split=config["split"], trust_remote_code=True)
                else:
                    dataset = load_dataset(config["hf_id"], split=config["split"], trust_remote_code=True)

                # Apply subset size limit if set (for testing)
                if config["subset_size"] is not None and len(dataset) > config["subset_size"]:
                    dataset = dataset.select(range(config["subset_size"]))
                    print(f"📊 Subsampled to {config['subset_size']} records for testing")
                else:
                    print(f"📊 Loaded full dataset: {len(dataset)} records")

                # Format dataset into standardized DataFrame
                if dataset_name == "hotpotqa":
                    def extract_hotpotqa_context(item):
                        titles = item[config["context_field"]]["title"]
                        sentences = item[config["context_field"]]["sentences"]
                        return " ".join([f"{title}: {' '.join(sents)}" for title, sents in zip(titles, sentences)])
                    df = pd.DataFrame({
                        "context": [extract_hotpotqa_context(item) for item in dataset],
                        "question": dataset[config["question_field"]],
                        "answer": dataset[config["answer_field"]]
                    })
                else:
                    df = pd.DataFrame({
                        "context": dataset[config["context_field"]],
                        "question": dataset[config["question_field"]],
                        "answer": dataset[config["answer_field"]]
                    })

            except Exception as e:
                print(f"⚠️ Failed to load {dataset_name}, using mock data: {str(e)}")
                df = self._generate_mock_data()

            df.columns = [str(col) for col in df.columns]
            self.raw_datasets[dataset_name] = df
            self._generate_dataset_statistics(dataset_name, df, start_time)

        gc.collect()
        return self.raw_datasets

    def _generate_dataset_statistics(self, dataset_name, df, start_time):
        context_col = "context"
        question_col = "question"
        answer_col = "answer"

        stats = {
            "dataset_name": dataset_name,
            "hf_source": DATASET_CONFIGS[dataset_name]["hf_id"],
            "total_records": len(df),
            "load_time_seconds": round(time.time() - start_time, 2),
            "columns": list(df.columns),
            "role": "development dataset" if dataset_name == "hotpotqa" else "multi-domain control dataset"
        }

        context_lengths = df[context_col].astype(str).apply(len)
        stats["context"] = {
            "avg_length_chars": round(context_lengths.mean(), 1),
            "min_length_chars": int(context_lengths.min()),
            "max_length_chars": int(context_lengths.max()),
            "std_length_chars": round(context_lengths.std(), 1)
        }

        q_lengths = df[question_col].astype(str).apply(len)
        stats["question"] = {
            "avg_length_chars": round(q_lengths.mean(), 1),
            "min_length_chars": int(q_lengths.min()),
            "max_length_chars": int(q_lengths.max())
        }

        a_lengths = df[answer_col].astype(str).apply(len)
        stats["answer"] = {
            "avg_length_chars": round(a_lengths.mean(), 1),
            "min_length_chars": int(a_lengths.min()),
            "max_length_chars": int(a_lengths.max())
        }

        print(f"✅ {dataset_name} loaded in {stats['load_time_seconds']}s")
        print(f"📊 {dataset_name} stats: {stats['total_records']} total records | Role: {stats['role']}")
        print(f"   Context: Avg {stats['context']['avg_length_chars']} chars")
        print(f"   Question: Avg {stats['question']['avg_length_chars']} chars")

        stats_save_path = os.path.join(self.output_path, "dataset_statistics", f"{dataset_name}_initial_statistics.json")
        with open(stats_save_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"📈 Statistics saved to: {os.path.basename(stats_save_path)}")

    def clean_data(self):
        print("\n" + "="*80)
        print("MODULE 2: ENHANCED DATA CLEANING")
        print("="*80)

        for dataset_name, df in self.raw_datasets.items():
            start_time = time.time()
            print(f"\n--- Cleaning {dataset_name} ---")

            context_col = "context"
            question_col = "question"
            answer_col = "answer"

            cleaning_report = {
                "dataset_name": dataset_name,
                "initial_record_count": len(df),
                "cleaning_rules": [],
                "total_removed_records": 0
            }
            removed_records_list = []

            df_clean = df.copy()
            df_clean["cleaning_removal_reason"] = ""

            # Rule 1: Remove empty/too-short context
            rule_name = "Empty/too-short context (<10 chars)"
            mask = (pd.isna(df_clean[context_col])) | (df_clean[context_col].astype(str).str.strip().str.len() < 10)
            removed = df_clean[mask].copy()
            if len(removed) > 0:
                cleaning_report["cleaning_rules"].append({"rule": rule_name, "records_removed": len(removed)})
                df_clean = df_clean[~mask].reset_index(drop=True)
                print(f"🧹 Removed {len(removed)} records: {rule_name}")

            # Rule 2: Remove duplicate context
            rule_name = "Duplicate context"
            mask = df_clean.duplicated(subset=[context_col], keep="first")
            removed = df_clean[mask].copy()
            if len(removed) > 0:
                cleaning_report["cleaning_rules"].append({"rule": rule_name, "records_removed": len(removed)})
                df_clean = df_clean[~mask].reset_index(drop=True)
                print(f"🧹 Removed {len(removed)} records: {rule_name}")

            # Rule 3: Remove too-short questions
            rule_name = "Too-short question (<5 chars)"
            mask = (pd.isna(df_clean[question_col])) | (df_clean[question_col].astype(str).str.strip().str.len() < 5)
            removed = df_clean[mask].copy()
            if len(removed) > 0:
                cleaning_report["cleaning_rules"].append({"rule": rule_name, "records_removed": len(removed)})
                df_clean = df_clean[~mask].reset_index(drop=True)
                print(f"🧹 Removed {len(removed)} records: {rule_name}")

            # Rule 4: Remove HTML tags
            rule_name = "HTML tag removal"
            df_clean[context_col] = df_clean[context_col].astype(str).apply(lambda x: re.sub(r'<.*?>', '', x))
            print(f"🧹 Applied {rule_name} to all records")

            # Rule 5: Normalize whitespace
            rule_name = "Whitespace normalization"
            df_clean[context_col] = df_clean[context_col].astype(str).apply(lambda x: re.sub(r'\s+', ' ', x.replace("\n", " ").replace("\r", " ")).strip())
            print(f"🧹 Applied {rule_name} to all records")

            # Rule 6: Remove all-whitespace context
            rule_name = "All-whitespace context"
            mask = df_clean[context_col].astype(str).str.strip().str.len() == 0
            removed = df_clean[mask].copy()
            if len(removed) > 0:
                cleaning_report["cleaning_rules"].append({"rule": rule_name, "records_removed": len(removed)})
                df_clean = df_clean[~mask].reset_index(drop=True)
                print(f"🧹 Removed {len(removed)} records: {rule_name}")

            df_clean = df_clean.drop(columns=["cleaning_removal_reason"], errors="ignore")
            cleaning_report["final_record_count"] = len(df_clean)
            cleaning_report["total_removed_records"] = cleaning_report["initial_record_count"] - cleaning_report["final_record_count"]
            cleaning_report["cleaning_time_seconds"] = round(time.time() - start_time, 2)

            self.cleaned_datasets[dataset_name] = df_clean
            self.cleaning_reports[dataset_name] = cleaning_report

            print(f"✅ {dataset_name} cleaning completed in {cleaning_report['cleaning_time_seconds']}s")
            print(f"📊 {dataset_name}: {cleaning_report['initial_record_count']} → {cleaning_report['final_record_count']} (removed {cleaning_report['total_removed_records']})")

            report_save_path = os.path.join(self.output_path, "cleaning_reports", f"{dataset_name}_cleaning_report.json")
            with open(report_save_path, "w", encoding="utf-8") as f:
                json.dump(cleaning_report, f, indent=2)
            print(f"📈 Cleaning report saved to: {os.path.basename(report_save_path)}")

        gc.collect()
        return self.cleaned_datasets, self.cleaning_reports

    def chunk_data_multi_param(self, chunk_sizes=CHUNK_SIZE_GRID, overlaps=CHUNK_OVERLAP_GRID):
        print("\n" + "="*80)
        print("MODULE 3: MULTI-PARAMETER TEXT CHUNKING")
        print("="*80)

        print("\n--- Merging cleaned datasets into final corpus ---")
        all_texts = []
        all_meta = []

        for dataset_name, df in self.cleaned_datasets.items():
            for idx, row in df.iterrows():
                text = str(row["context"])
                if len(text) > 50:
                    all_texts.append(text)
                    all_meta.append({
                        "dataset": dataset_name,
                        "original_index": int(idx),
                        "question": str(row["question"]),
                        "answer": str(row["answer"]),
                        "role": "development" if dataset_name == "hotpotqa" else "control"
                    })

        print(f"✅ Merged corpus: {len(all_texts)} unique documents from 3 datasets")

        chunking_stats = []
        final_chunks = None
        final_meta = None
        final_params = {"chunk_size": 512, "overlap": 51}

        print(f"\n--- Testing chunking parameter grid ---")
        for chunk_size in chunk_sizes:
            for overlap in overlaps:
                if overlap >= chunk_size:
                    continue
                
                start_time = time.time()
                print(f"\nTesting: chunk_size={chunk_size}, overlap={overlap}")
                
                chunks = []
                chunk_meta = []
                
                for text_idx, text in enumerate(all_texts):
                    doc_chunks = self._sentence_aware_chunking(text, chunk_size, overlap)
                    for c in doc_chunks:
                        chunks.append(c)
                        chunk_meta.append({**all_meta[text_idx], "chunk_id": f"{all_meta[text_idx]['dataset']}_{text_idx}_{len(chunks)}"})

                chunk_lengths = [len(c) for c in chunks]
                stats = {
                    "chunk_size": chunk_size,
                    "overlap": overlap,
                    "total_chunks": len(chunks),
                    "avg_chunk_length": round(np.mean(chunk_lengths), 1),
                    "chunking_time": round(time.time() - start_time, 2)
                }
                chunking_stats.append(stats)
                
                print(f"   → {stats['total_chunks']} chunks, Avg {stats['avg_chunk_length']} chars")
                chunk_save_path = os.path.join(self.output_path, "chunking_results", f"chunks_{chunk_size}_{overlap}.json")
                with open(chunk_save_path, "w", encoding="utf-8") as f:
                    json.dump([{"chunk": c, "metadata": m} for c, m in zip(chunks, chunk_meta)], f, indent=2)

                self.chunking_results[(chunk_size, overlap)] = (chunks, chunk_meta)
                if chunk_size == final_params["chunk_size"] and overlap == final_params["overlap"]:
                    final_chunks = chunks
                    final_meta = chunk_meta

        self.chunking_stats = chunking_stats
        comparison_save_path = os.path.join(self.output_path, "chunking_results", "chunking_parameter_comparison.json")
        with open(comparison_save_path, "w", encoding="utf-8") as f:
            json.dump(chunking_stats, f, indent=2)
        print(f"\n📈 Chunking comparison saved to: {os.path.basename(comparison_save_path)}")

        self.final_chunks = final_chunks
        self.final_metadata = final_meta
        print(f"\n✅ Selected final chunking: 512/51 → {len(self.final_chunks)} chunks")

        gc.collect()
        return self.chunking_results, self.chunking_stats

    def _sentence_aware_chunking(self, text, chunk_size, overlap):
        chunks = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length:
                split_point = max(text.rfind(' ', start, end), text.rfind('.', start, end))
                if split_point > start:
                    end = split_point + 1
            chunk = text[start:end].strip()
            if len(chunk) > 50:
                chunks.append(chunk)
            start = end - overlap if end - overlap > start else text_length
        return chunks

    def build_bm25_index(self):
        print("\n" + "="*80)
        print("MODULE 4: BUILD BM25 SPARSE INDEX")
        print("="*80)

        start_time = time.time()
        bm25 = SimpleBM25(self.final_chunks)
        self.bm25_index = bm25

        save_path = os.path.join(self.output_path, "indexes", "bm25_index_final.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({"bm25_model": bm25, "chunks": self.final_chunks, "metadata": self.final_metadata}, f)

        print(f"✅ BM25 index built in {round(time.time() - start_time, 2)}s, {len(self.final_chunks)} chunks")
        gc.collect()
        return self.bm25_index

    def build_contriever_index(self):
        print("\n" + "="*80)
        print("MODULE 5: BUILD CONTRIEVER DENSE INDEX")
        print("="*80)

        start_time = time.time()
        model = SentenceTransformer(EMBED_MODEL_NAME)

        print(f"\n--- Generating embeddings (batch size: {BATCH_SIZE}) ---")
        embeddings = []
        for i in tqdm(range(0, len(self.final_chunks), BATCH_SIZE), desc="Embedding batches"):
            batch = self.final_chunks[i:i+BATCH_SIZE]
            emb = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            embeddings.append(emb)

        embs = np.concatenate(embeddings, axis=0)
        print(f"✅ Embeddings generated: shape {embs.shape}")

        faiss_index = faiss.IndexFlatL2(embs.shape[1])
        faiss_index.add(embs)
        self.contriever_index = faiss_index

        index_save_path = os.path.join(self.output_path, "indexes", "dense_retrieval_index_final.faiss")
        meta_save_path = os.path.join(self.output_path, "indexes", "chunk_metadata_final.pkl")
        faiss.write_index(faiss_index, index_save_path)
        with open(meta_save_path, "wb") as f:
            pickle.dump(self.final_metadata, f)

        print(f"✅ Contriever index built in {round(time.time() - start_time, 2)}s, {faiss_index.ntotal} vectors")
        gc.collect()
        return self.contriever_index

    def run_full_pipeline(self):
        total_start = time.time()
        print("\n" + "="*100)
        print("STARTING FULL RAG PREPROCESSING PIPELINE (3 DATASETS, NO NQ)")
        print("="*100)

        self.load_and_analyze_datasets()
        self.clean_data()
        self.chunk_data_multi_param()
        self.build_bm25_index()
        self.build_contriever_index()

        print("\n" + "="*100)
        print("🎉 FULL PIPELINE COMPLETED SUCCESSFULLY")
        print("="*100)
        print(f"Total time: {round(time.time() - total_start, 2)}s")
        print(f"Outputs saved to: {self.output_path}")
        print("\n📦 Final Deliverables:")
        print("1. 3 Datasets Statistics & Cleaning Reports")
        print("2. Multi-parameter Chunking Results")
        print("3. BM25 Sparse Retrieval Index")
        print("4. Contriever Dense Retrieval Index")
        print("5. Query Rewrite Module (integrated)")

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    pipeline = FullPreprocessingPipeline()
    pipeline.run_full_pipeline()


# In[ ]:




