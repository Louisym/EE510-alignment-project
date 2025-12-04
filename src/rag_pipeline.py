"""
RAG (Retrieval-Augmented Generation) 主流程
整合文档检索和模型生成功能
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .model_loader import LlamaModelLoader
from .vector_database import VectorDatabase
from .document_processor import CourseDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """RAG检索增强生成管道"""
    
    def __init__(self,
                 model_name: str = "deepseek-ai/deepseek-math-7b-instruct",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 db_path: str = "./data/chroma_db",
                 use_4bit: bool = True):
        """
        初始化RAG管道
        
        Args:
            model_name: 语言模型名称
            embedding_model: 嵌入模型名称
            db_path: 向量数据库路径
            use_4bit: 是否使用4bit量化
        """
        self.model_name = model_name
        self.use_4bit = use_4bit
        
        # 初始化组件
        self.model_loader = None
        self.vector_db = VectorDatabase(
            db_path=db_path,
            embedding_model=embedding_model
        )
        self.doc_processor = CourseDataProcessor()
        
        self.is_initialized = False
    
    def initialize(self, load_model: bool = True):
        """
        初始化所有组件
        
        Args:
            load_model: 是否加载语言模型（在只构建知识库时可设为False）
        """
        logger.info("初始化RAG管道...")
        
        # 初始化向量数据库
        self.vector_db.initialize()
        
        # 初始化语言模型
        if load_model:
            logger.info("加载语言模型（这可能需要几分钟）...")
            self.model_loader = LlamaModelLoader(
                model_name=self.model_name,
                use_4bit=self.use_4bit
            )
            self.model_loader.load_model_and_tokenizer()
        
        self.is_initialized = True
        logger.info("RAG管道初始化完成")
    
    def build_knowledge_base(self, docs_dir: str, clear_existing: bool = False):
        """
        构建知识库
        
        Args:
            docs_dir: 文档目录路径
            clear_existing: 是否清空现有数据库
        """
        if not self.is_initialized:
            self.initialize(load_model=False)
        
        logger.info(f"开始构建知识库，文档目录: {docs_dir}")
        
        # 清空现有数据库（如果需要）
        if clear_existing:
            self.vector_db.clear_collection()
            logger.info("已清空现有知识库")
        
        # 处理文档
        documents = self.doc_processor.process_directory(docs_dir)
        
        if not documents:
            logger.warning("未找到可处理的文档")
            return
        
        # 添加到向量数据库
        self.vector_db.add_documents(documents)
        
        # 显示知识库信息
        info = self.vector_db.get_collection_info()
        logger.info(f"知识库构建完成: {info}")
    
    def add_qa_pairs(self, qa_file: str):
        """
        添加问答对到知识库
        
        Args:
            qa_file: 问答对JSON文件路径
        """
        if not self.is_initialized:
            self.initialize(load_model=False)
        
        logger.info(f"添加问答对: {qa_file}")
        
        try:
            with open(qa_file, 'r', encoding='utf-8') as f:
                qa_data = json.load(f)
            
            from .vector_database import DocumentProcessor
            documents = DocumentProcessor.process_qa_pairs(qa_data)
            
            self.vector_db.add_documents(documents)
            
            logger.info(f"成功添加 {len(documents)} 个问答对")
        
        except Exception as e:
            logger.error(f"添加问答对失败: {e}")
            raise
    
    def retrieve_context(self, query: str, top_k: int = 3) -> Tuple[List[str], List[Dict]]:
        """
        检索相关上下文
        
        Args:
            query: 查询问题
            top_k: 检索结果数量
            
        Returns:
            (上下文文本列表, 检索结果详情)
        """
        if not self.is_initialized:
            raise RuntimeError("RAG管道未初始化")
        
        # 执行检索
        results = self.vector_db.search(query, n_results=top_k)
        
        # 提取上下文文本
        contexts = []
        for result in results:
            contexts.append(result['text'])
        
        return contexts, results
    
    def generate_response(self, 
                         query: str, 
                         max_length: int = 512,
                         temperature: float = 0.7,
                         use_context: bool = True,
                         top_k_retrieval: int = 3) -> Dict[str, Any]:
        """
        生成回答
        
        Args:
            query: 用户问题
            max_length: 最大生成长度
            temperature: 生成温度
            use_context: 是否使用检索到的上下文
            top_k_retrieval: 检索结果数量
            
        Returns:
            包含回答和元数据的字典
        """
        if not self.is_initialized:
            raise RuntimeError("RAG管道未初始化")
        
        if not self.model_loader:
            raise RuntimeError("语言模型未加载")
        
        response_data = {
            "query": query,
            "answer": "",
            "contexts": [],
            "retrieval_results": [],
            "used_context": use_context
        }
        
        # 构建提示词
        if use_context:
            # 检索相关上下文
            contexts, retrieval_results = self.retrieve_context(query, top_k_retrieval)
            response_data["contexts"] = contexts
            response_data["retrieval_results"] = retrieval_results
            
            # 构建包含上下文的提示词
            prompt = self._build_context_prompt(query, contexts)
        else:
            # 直接回答，不使用上下文
            prompt = self._build_direct_prompt(query)
        
        # 生成回答
        answer = self.model_loader.generate_response(
            prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        response_data["answer"] = answer
        
        return response_data
    
    def _build_context_prompt(self, query: str, contexts: List[str]) -> str:
        """Build prompt with context"""
        context_text = "\n\n".join([f"Reference {i+1}: {ctx}" for i, ctx in enumerate(contexts)])

        prompt = f"""You are a professional mathematics learning assistant specializing in probability theory, stochastic processes, and measure theory. Please answer questions based on the provided reference materials.

Reference Materials:
{context_text}

Question: {query}

Requirements:
1. Provide accurate mathematical answers based on the reference materials
2. Include necessary formula derivations and proof steps
3. Clearly indicate if you need to supplement with additional content
4. Use rigorous mathematical language and notation"""

        return prompt
    
    def _build_direct_prompt(self, query: str) -> str:
        """Build direct answer prompt"""
        prompt = f"""You are a professional mathematics assistant specializing in probability theory, stochastic processes, and measure theory.

Question: {query}

Requirements:
1. Provide accurate mathematical definitions and solutions
2. Include clear derivation processes and proof steps
3. Use standard mathematical notation and rigorous expressions
4. If theorems are involved, explain relevant conditions and application scenarios"""

        return prompt
    
    def compare_with_without_rag(self, query: str) -> Dict[str, Any]:
        """
        比较使用RAG和不使用RAG的回答效果
        
        Args:
            query: 用户问题
            
        Returns:
            包含两种回答的比较结果
        """
        logger.info(f"比较RAG效果，问题: {query}")
        
        # 使用RAG回答
        rag_response = self.generate_response(query, use_context=True)
        
        # 不使用RAG直接回答
        direct_response = self.generate_response(query, use_context=False)
        
        return {
            "query": query,
            "rag_response": rag_response,
            "direct_response": direct_response,
            "contexts_used": rag_response["contexts"]
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            "model_name": self.model_name,
            "use_4bit": self.use_4bit,
            "initialized": self.is_initialized
        }
        
        if self.model_loader:
            info["model_info"] = self.model_loader.get_model_info()
        
        if self.is_initialized:
            info["database_info"] = self.vector_db.get_collection_info()
        
        return info


class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag_pipeline = rag_pipeline
    
    def evaluate_on_qa_set(self, test_qa_file: str) -> Dict[str, Any]:
        """
        在测试问答集上评估RAG系统
        
        Args:
            test_qa_file: 测试问答对文件
            
        Returns:
            评估结果
        """
        logger.info(f"开始评估RAG系统，测试文件: {test_qa_file}")
        
        # 加载测试数据
        with open(test_qa_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        results = {
            "total_questions": len(test_data),
            "responses": [],
            "retrieval_quality": [],
            "summary": {}
        }
        
        for i, qa in enumerate(test_data):
            question = qa["question"]
            expected_answer = qa.get("answer", "")
            
            logger.info(f"评估问题 {i+1}/{len(test_data)}: {question[:50]}...")
            
            # 获取RAG回答
            response = self.rag_pipeline.generate_response(question)
            
            # 分析检索质量
            retrieval_quality = self._analyze_retrieval_quality(
                question, response["retrieval_results"]
            )
            
            result_item = {
                "question": question,
                "expected_answer": expected_answer,
                "generated_answer": response["answer"],
                "contexts": response["contexts"],
                "retrieval_quality": retrieval_quality
            }
            
            results["responses"].append(result_item)
            results["retrieval_quality"].append(retrieval_quality)
        
        # 计算汇总统计
        avg_retrieval_score = sum(rq["relevance_score"] for rq in results["retrieval_quality"]) / len(results["retrieval_quality"])
        
        results["summary"] = {
            "average_retrieval_relevance": avg_retrieval_score,
            "total_contexts_retrieved": sum(len(r["contexts"]) for r in results["responses"])
        }
        
        logger.info(f"评估完成，平均检索相关性: {avg_retrieval_score:.3f}")
        
        return results
    
    def _analyze_retrieval_quality(self, question: str, retrieval_results: List[Dict]) -> Dict[str, Any]:
        """分析检索质量"""
        if not retrieval_results:
            return {"relevance_score": 0.0, "analysis": "无检索结果"}
        
        # 简单的相关性评分（基于距离）
        distances = [result["distance"] for result in retrieval_results]
        avg_distance = sum(distances) / len(distances)
        
        # 距离越小相关性越高，转换为0-1分数
        relevance_score = max(0, 1 - avg_distance)
        
        return {
            "relevance_score": relevance_score,
            "num_results": len(retrieval_results),
            "avg_distance": avg_distance,
            "distances": distances,
            "analysis": f"检索到{len(retrieval_results)}个相关片段，平均距离{avg_distance:.3f}"
        }


if __name__ == "__main__":
    # 测试RAG管道
    
    try:
        # 初始化RAG管道（仅用于测试，不加载模型）
        rag = RAGPipeline()
        rag.initialize(load_model=False)
        
        # 显示系统信息
        info = rag.get_system_info()
        print("系统信息:", json.dumps(info, indent=2, ensure_ascii=False))
        
        # 测试知识库构建（使用测试数据）
        test_docs_dir = "./data/test_docs"
        if Path(test_docs_dir).exists():
            rag.build_knowledge_base(test_docs_dir, clear_existing=True)
        
            # 测试检索
            test_query = "什么是条件概率？"
            contexts, results = rag.retrieve_context(test_query)
            
            print(f"\n检索测试 - 查询: {test_query}")
            print(f"检索到 {len(contexts)} 个相关片段:")
            for i, (ctx, result) in enumerate(zip(contexts, results)):
                print(f"{i+1}. [距离: {result['distance']:.4f}] {ctx[:100]}...")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"错误: {e}")