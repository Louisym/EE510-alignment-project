"""
测试完整的 RAG 问答系统
"""

from src.rag_pipeline import RAGPipeline

print('🔧 初始化 RAG 系统（加载模型）...')
print('=' * 80)

rag = RAGPipeline()
rag.initialize(load_model=True)

print('\n✅ 系统初始化完成！')
print('\n📚 系统信息:')
info = rag.get_system_info()
print(f'  模型: {info["model_info"]["model_name"]}')
print(f'  量化: {info["model_info"]["quantization"]}')
print(f'  文档数量: {info["database_info"]["document_count"]}')

# 测试问题
test_question = "What is conditional probability? Please provide the definition and formula."

print(f'\n🤔 问题: {test_question}')
print('=' * 80)
print('\n💭 生成回答中...\n')

response = rag.generate_response(
    test_question,
    max_length=500,
    temperature=0.7
)

print(f'【RAG 回答】:')
print(response['answer'])

print(f'\n📖 使用的参考资料 ({len(response["contexts"])} 个):')
for i, ctx in enumerate(response['contexts'][:3]):
    print(f'\n参考 {i+1}:')
    print(f'  {ctx[:200]}...')

print('\n' + '=' * 80)
print('✅ 测试完成！')
