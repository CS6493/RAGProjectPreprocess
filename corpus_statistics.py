import os
import pandas as pd
import numpy as np
from datasets import load_dataset
import plotly.graph_objects as go
from tqdm import tqdm

class RAGDatasetProfiler:
    def __init__(self, sample_size=1000, output_dir="./rag_data"):
        """
        统一支持四种数据集的获取与统计
        :param sample_size: 默认抽取样本量
        :param output_dir: 本地保存数据集的文件夹路径
        """
        self.sample_size = sample_size
        self.output_dir = output_dir
        # 注意：这里已经修复了 FinanceBench 的大小写路径
        self.dataset_configs = {
            "Natural_Questions": {"path": "google-research-datasets/natural_questions", "split": "validation", "streaming": True},
            "PubMedQA": {"path": "qiaojin/PubMedQA", "name": "pqa_labeled", "split": "train", "streaming": False},
            "FinanceBench": {"path": "PatronusAI/financebench", "split": "train", "streaming": False},
            "HotpotQA": {"path": "hotpot_qa", "name": "distractor", "split": "validation", "streaming": False}
        }

    def fetch_and_extract(self, dataset_name, save_local=False, output_format="csv"):
        """
        统一提取 query 和 context，并提供本地保存选项
        :param dataset_name: 数据集名称
        :param save_local: 是否保存到本地 (True/False)
        :param output_format: 保存格式 ("csv" 或 "json")
        """
        config = self.dataset_configs.get(dataset_name)
        if not config:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        print(f"\nLoading {dataset_name}...")
        
        dataset = load_dataset(
            config["path"], 
            name=config.get("name"), 
            split=config["split"], 
            streaming=config["streaming"]
        )

        extracted_data = []
        iterator = iter(dataset) if config["streaming"] else dataset
        
        for i, item in enumerate(tqdm(iterator, total=self.sample_size)):
            if i >= self.sample_size:
                break
            
            # 适配不同数据集的字段格式
            if dataset_name == "Natural_Questions":
                query = item['question']['text']
                context = item['document']['tokens']['token'][:500] 
                context = " ".join(context) if isinstance(context, list) else str(context)
                
            elif dataset_name == "PubMedQA":
                query = item['question']
                context = " ".join(item['context']['contexts'])
                
            elif dataset_name == "FinanceBench":
                query = item['question']
                # 修复后的 FinanceBench 提取逻辑
                evidence_list = item.get('evidence', [])
                context = " ".join([str(ev.get('evidence_text', '')) for ev in evidence_list])
                
            elif dataset_name == "HotpotQA":
                query = item['question']
                context = " ".join([" ".join(c[1]) for c in item['context']])

            extracted_data.append({"query": query, "context": context})

        df = pd.DataFrame(extracted_data)

        # --- 新增：保存到本地的逻辑 ---
        if save_local:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                
            file_path = os.path.join(self.output_dir, f"{dataset_name}.{output_format}")
            
            if output_format == "csv":
                df.to_csv(file_path, index=False, encoding='utf-8')
            elif output_format == "json":
                df.to_json(file_path, orient="records", force_ascii=False, indent=4)
                
            print(f"✅ [{dataset_name}] 已成功保存至本地: {file_path}")

        return df

    def compute_statistics(self, df):
        """计算 RAG 强相关的统计信息"""
        df['query_length'] = df['query'].apply(lambda x: len(str(x).split()))
        df['context_length'] = df['context'].apply(lambda x: len(str(x).split()))
        
        def overlap_ratio(row):
            q_words = set(str(row['query']).lower().split())
            c_words = set(str(row['context']).lower().split())
            if not q_words: return 0
            return len(q_words.intersection(c_words)) / len(q_words)
            
        df['vocab_overlap_ratio'] = df.apply(overlap_ratio, axis=1)

        stats_summary = {
            "Query平均长度(词)": round(df['query_length'].mean(), 2),
            "Query P90长度": round(np.percentile(df['query_length'], 90), 2),
            "Context平均长度(词)": round(df['context_length'].mean(), 2),
            "Context P90长度": round(np.percentile(df['context_length'], 90), 2),
            "平均词汇重合度": f"{round(df['vocab_overlap_ratio'].mean() * 100, 2)}%"
        }
        return df, stats_summary

    # ... (此处省略 plot_interactive_distributions 的代码，与上一版本完全一致)

# --- 运行示例 ---
if __name__ == "__main__":
    # 实例化并指定保存目录
    profiler = RAGDatasetProfiler(sample_size=1000, output_dir="./rag_datasets_local")
    all_datasets_df = {}
    
    print("\n=== 开始提取并可选择保存数据集 ===")
    for db_name in profiler.dataset_configs.keys():
        # 在这里将 save_local 设为 True 即可下载到本地！
        # 可以选择 output_format="csv" 或 "json"
        df = profiler.fetch_and_extract(db_name, save_local=True, output_format="csv")
        
        df, stats = profiler.compute_statistics(df)
        all_datasets_df[db_name] = df
        
        print(f"{db_name} 统计信息:")
        for k, v in stats.items():
            print(f"  - {k}: {v}")