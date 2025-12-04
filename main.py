"""
概率论学习助手主程序
提供命令行界面和Web界面
"""

import argparse
import json
import logging
from pathlib import Path
import sys

# 添加src目录到路径
sys.path.append('./src')

from src.rag_pipeline import RAGPipeline
from src.evaluation import BenchmarkRunner
from src.document_processor import CourseDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_knowledge_base(rag_pipeline: RAGPipeline, docs_dir: str, qa_file: str = None):
    """设置知识库"""
    logger.info("开始构建知识库...")
    
    # 构建文档知识库
    if Path(docs_dir).exists():
        rag_pipeline.build_knowledge_base(docs_dir, clear_existing=True)
    else:
        logger.warning(f"文档目录不存在: {docs_dir}")
    
    # 添加问答对
    if qa_file and Path(qa_file).exists():
        rag_pipeline.add_qa_pairs(qa_file)
    elif qa_file:
        logger.warning(f"问答对文件不存在: {qa_file}")
    
    # 显示知识库信息
    info = rag_pipeline.get_system_info()
    logger.info(f"知识库构建完成: {info['database_info']}")

def interactive_chat(rag_pipeline: RAGPipeline):
    """交互式聊天模式"""
    print("\n=== 概率论学习助手 ===")
    print("输入 'quit' 退出，输入 'help' 查看帮助")
    print("你可以询问概率论、随机过程、测度论相关问题")
    print("=" * 40)
    
    while True:
        try:
            question = input("\n你的问题: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            elif question.lower() == 'help':
                print("""
帮助信息:
- 直接输入概率论相关问题
- 输入 'compare' 比较RAG和基线模型
- 输入 'info' 查看系统信息
- 输入 'quit' 退出
                """)
                continue
            elif question.lower() == 'compare':
                question = input("请输入要比较的问题: ").strip()
                if question:
                    print("\n比较RAG和基线模型效果...")
                    comparison = rag_pipeline.compare_with_without_rag(question)
                    
                    print(f"\n问题: {question}")
                    print(f"\n【RAG模型回答】:")
                    print(comparison['rag_response']['answer'])
                    print(f"\n【基线模型回答】:")
                    print(comparison['direct_response']['answer'])
                    
                    if comparison['contexts_used']:
                        print(f"\n【使用的参考资料】:")
                        for i, ctx in enumerate(comparison['contexts_used'][:2]):
                            print(f"{i+1}. {ctx[:150]}...")
                continue
            elif question.lower() == 'info':
                info = rag_pipeline.get_system_info()
                print(f"\n系统信息:")
                print(json.dumps(info, indent=2, ensure_ascii=False))
                continue
            elif not question:
                continue
            
            print("\n正在思考...")
            response = rag_pipeline.generate_response(
                question,
                max_length=400,
                temperature=0.7
            )
            
            print(f"\n【回答】:")
            print(response['answer'])
            
            if response['contexts']:
                print(f"\n【参考资料】:")
                for i, ctx in enumerate(response['contexts'][:2]):
                    print(f"{i+1}. {ctx[:100]}...")
                    
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            logger.error(f"处理问题时出错: {e}")
            print(f"抱歉，处理问题时出现错误: {e}")

def run_benchmark(rag_pipeline: RAGPipeline, test_file: str):
    """运行基准测试"""
    if not Path(test_file).exists():
        logger.error(f"测试文件不存在: {test_file}")
        return
    
    runner = BenchmarkRunner(rag_pipeline, test_file)
    results = runner.run_benchmark(compare_baseline=True, save_results=True)
    
    print(f"\n=== 基准测试结果 ===")
    print(f"总问题数: {results['test_info']['total_questions']}")
    print(f"RAG处理时间: {results['test_info']['rag_time']:.2f}秒")
    
    if 'baseline_time' in results['test_info']:
        print(f"基线处理时间: {results['test_info']['baseline_time']:.2f}秒")
    
    evaluation = results['evaluation']
    if 'improvement' in evaluation:
        improvement = evaluation['improvement']
        print(f"准确率提升: {improvement['heuristic_improvement']:.3f}")

def main():
    parser = argparse.ArgumentParser(description="概率论学习助手")
    parser.add_argument("--mode", choices=["chat", "build", "benchmark"], default="chat",
                       help="运行模式: chat(聊天), build(构建知识库), benchmark(基准测试)")
    parser.add_argument("--docs-dir", default="./data/docs", help="文档目录路径")
    parser.add_argument("--qa-file", help="问答对JSON文件路径")
    parser.add_argument("--test-file", help="测试数据文件路径")
    parser.add_argument("--model", default="deepseek-ai/deepseek-math-7b-instruct", help="模型名称")
    parser.add_argument("--no-model", action="store_true", help="不加载模型（仅构建知识库）")
    parser.add_argument("--4bit", action="store_true", default=True, help="使用4bit量化")
    
    args = parser.parse_args()
    
    # 初始化RAG管道
    rag_pipeline = RAGPipeline(
        model_name=args.model,
        use_4bit=args._4bit if hasattr(args, '_4bit') else True
    )
    
    try:
        if args.mode == "build":
            # 仅构建知识库模式
            rag_pipeline.initialize(load_model=False)
            setup_knowledge_base(rag_pipeline, args.docs_dir, args.qa_file)
            
        elif args.mode == "benchmark":
            # 基准测试模式
            if not args.test_file:
                logger.error("基准测试模式需要指定--test-file")
                return
            
            rag_pipeline.initialize(load_model=not args.no_model)
            
            if not args.no_model:
                # 构建知识库（如果需要）
                setup_knowledge_base(rag_pipeline, args.docs_dir, args.qa_file)
                
                # 运行基准测试
                run_benchmark(rag_pipeline, args.test_file)
            
        else:
            # 聊天模式（默认）
            if args.no_model:
                logger.error("聊天模式需要加载模型")
                return
            
            rag_pipeline.initialize(load_model=True)
            setup_knowledge_base(rag_pipeline, args.docs_dir, args.qa_file)
            
            # 进入交互模式
            interactive_chat(rag_pipeline)
            
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        print(f"错误: {e}")

if __name__ == "__main__":
    main()