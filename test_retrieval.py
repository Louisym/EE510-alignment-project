"""
测试知识库检索功能
"""

from src.rag_pipeline import RAGPipeline

print('🔧 初始化 RAG 系统...')
rag = RAGPipeline()
rag.initialize(load_model=False)

print('\n📚 知识库信息:')
info = rag.get_system_info()
print(f'  文档数量: {info["database_info"]["document_count"]}')
print(f'  嵌入模型: {info["database_info"]["embedding_model"]}')

# 测试几个问题
test_queries = [
    "What is conditional probability?",
    "Explain random variables",
    "What is Bayes theorem?"
]

for query in test_queries:
    print(f'\n🔍 检索问题: "{query}"')
    print('=' * 80)

    contexts, results = rag.retrieve_context(query, top_k=3)

    for i, (ctx, res) in enumerate(zip(contexts, results)):
        print(f'\n结果 {i+1} (相似度距离: {res["distance"]:.4f}):')
        print(f'  来源: {res["metadata"].get("source_file", "unknown")}')
        print(f'  章节: {res["metadata"].get("chapter", "unknown")[:50]}...')
        print(f'  内容预览: {ctx[:250].replace(chr(10), " ")}...')
