import json
import os
import sys
from openai import OpenAI
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, context_recall, answer_relevancy
from ragas.llms import llm_factory
# from ragas.embeddings import OpenAIEmbeddings as RagasOpenAIEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

# ============ 配置 ============
QWEN_API_KEY = "sk-8a9366aa10db4081ab2a706651ab7412"
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen-plus-latest"
EMBEDDING_MODEL = "text-embedding-v4"

# ============ 定位文件 ============
script_dir = os.path.dirname(os.path.abspath(__file__))
eval_data_path = os.path.join(script_dir, 'eval_data.json')

print(f"📁 读取文件: {eval_data_path}")

if not os.path.exists(eval_data_path):
    print(f"❌ 文件不存在: {eval_data_path}")
    sys.exit(1)

# ============ 加载数据 ============
with open(eval_data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"📊 加载 {len(data)} 条评估数据")

# ============ 构建Dataset ============
dataset = Dataset.from_dict({
    "question": [item["question"] for item in data],
    "answer": [item["answer"] for item in data],
    "contexts": [item["contexts"] for item in data],
    "reference": [item["ground_truth"] for item in data],  # ✅ 添加 reference
})

# ============ 创建客户端 ============
client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url=QWEN_BASE_URL,
)

# ============ 创建 LLM ============
ragas_llm = llm_factory(MODEL_NAME, client=client, max_tokens=8192)

# ============ 创建 Embeddings ============
lc_embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=QWEN_API_KEY,
)
ragas_embeddings = LangchainEmbeddingsWrapper(lc_embeddings)

# ============ 运行评估（三个指标）============
print("\n🔬 正在评估，请稍候...")

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness, context_recall, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
)

# ============ 输出结果 ============
print("\n" + "=" * 50)
print("📈 RAG 系统评估报告")
print("=" * 50)

def get_avg(score):
    if isinstance(score, list):
        return sum(score) / len(score) if score else 0
    return score

faith_avg = get_avg(result['faithfulness'])
recall_avg = get_avg(result['context_recall'])
relevancy_avg = get_avg(result['answer_relevancy'])

print(f"Faithfulness (忠实度):       {faith_avg:.4f}")
print(f"Context Recall (检索召回率):  {recall_avg:.4f}")
print(f"Answer Relevancy (答案相关性): {relevancy_avg:.4f}")
print("=" * 50)

# 保存报告
df = result.to_pandas()
df.to_csv('evaluation_report.csv', index=False, encoding='utf-8-sig')
print("✅ 报告已保存")