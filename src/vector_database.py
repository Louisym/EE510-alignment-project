"""
向量数据库构建模块，用于构建课程资料的检索知识库
"""

import os
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorDatabase:
    def __init__(self,
                 db_path: str = "./data/chroma_db",
                 embedding_model: str = "BAAI/bge-base-en-v1.5",
                 collection_name: str = "probability_theory"):
        """
        初始化向量数据库
        
        Args:
            db_path: ChromaDB存储路径
            embedding_model: 文本嵌入模型
            collection_name: 集合名称
        """
        self.db_path = db_path
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name
        
        # 初始化嵌入模型
        self.embedding_model = None
        self.client = None
        self.collection = None
        
    def initialize(self):
        """初始化数据库和嵌入模型"""
        logger.info("初始化向量数据库...")
        
        # 创建存储目录
        os.makedirs(self.db_path, exist_ok=True)
        
        # 初始化ChromaDB客户端
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 获取或创建集合
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"已连接到现有集合: {self.collection_name}")
        except:
            self.collection = self.client.create_collection(name=self.collection_name)
            logger.info(f"创建新集合: {self.collection_name}")
        
        # 初始化嵌入模型
        logger.info(f"加载嵌入模型: {self.embedding_model_name}")
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        
        logger.info("向量数据库初始化完成")
    
    def add_documents(self, documents: List[Dict[str, Any]]):
        """
        向数据库添加文档
        
        Args:
            documents: 文档列表，每个文档包含 {'id', 'text', 'metadata'}
        """
        if not self.collection or not self.embedding_model:
            raise RuntimeError("数据库未初始化，请先调用initialize()")
        
        logger.info(f"开始添加 {len(documents)} 个文档...")
        
        # 准备数据
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        for doc in documents:
            ids.append(doc['id'])
            texts.append(doc['text'])
            metadatas.append(doc.get('metadata', {}))
        
        # 生成嵌入向量
        logger.info("生成文本嵌入向量...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True).tolist()
        
        # 添加到数据库
        logger.info("保存到向量数据库...")
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"成功添加 {len(documents)} 个文档到数据库")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        搜索相关文档
        
        Args:
            query: 搜索查询
            n_results: 返回结果数量
            
        Returns:
            搜索结果列表
        """
        if not self.collection or not self.embedding_model:
            raise RuntimeError("数据库未初始化，请先调用initialize()")
        
        # 生成查询向量
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # 执行搜索
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # 格式化结果
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return formatted_results
    
    def get_collection_info(self) -> Dict[str, Any]:
        """获取集合信息"""
        if not self.collection:
            return {"status": "数据库未初始化"}
        
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedding_model_name
        }
    
    def clear_collection(self):
        """清空集合"""
        if self.collection:
            # 获取所有ID
            all_docs = self.collection.get()
            if all_docs['ids']:
                self.collection.delete(ids=all_docs['ids'])
            logger.info("集合已清空")


class DocumentProcessor:
    """文档处理器，将原始文档转换为向量数据库可用格式"""
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        将长文本分块
        
        Args:
            text: 输入文本
            chunk_size: 块大小（字符数）
            overlap: 重叠字符数
            
        Returns:
            文本块列表
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
                
            start = end - overlap
        
        return chunks
    
    @staticmethod
    def process_textbook_content(content: str, source: str, chapter: str = None) -> List[Dict[str, Any]]:
        """
        处理教材内容
        
        Args:
            content: 文本内容
            source: 来源（如教材名称）
            chapter: 章节信息
            
        Returns:
            处理后的文档列表
        """
        chunks = DocumentProcessor.chunk_text(content)
        
        documents = []
        for i, chunk in enumerate(chunks):
            doc_id = f"{source}_{chapter}_{i}" if chapter else f"{source}_{i}"
            
            metadata = {
                "source": source,
                "chunk_index": i,
                "type": "textbook"
            }
            
            if chapter:
                metadata["chapter"] = chapter
            
            documents.append({
                "id": doc_id,
                "text": chunk,
                "metadata": metadata
            })
        
        return documents
    
    @staticmethod
    def process_qa_pairs(qa_data: List[Dict[str, str]], source: str = "exercises") -> List[Dict[str, Any]]:
        """
        处理问答对数据
        
        Args:
            qa_data: 问答对列表 [{"question": "...", "answer": "..."}]
            source: 数据来源
            
        Returns:
            处理后的文档列表
        """
        documents = []
        
        for i, qa in enumerate(qa_data):
            # 将问题和答案组合成一个文档
            combined_text = f"问题：{qa['question']}\n\n答案：{qa['answer']}"
            
            doc_id = f"{source}_qa_{i}"
            metadata = {
                "source": source,
                "type": "qa_pair",
                "question": qa['question']
            }
            
            documents.append({
                "id": doc_id,
                "text": combined_text,
                "metadata": metadata
            })
        
        return documents


if __name__ == "__main__":
    # 测试向量数据库
    
    # 创建测试数据
    test_documents = [
        {
            "id": "test_1",
            "text": "概率论是数学的一个分支，研究随机现象的数量规律。条件概率是概率论中的重要概念。",
            "metadata": {"source": "test", "type": "definition"}
        },
        {
            "id": "test_2", 
            "text": "随机过程是一族随机变量，通常用来建模随时间变化的随机现象。布朗运动是经典的随机过程例子。",
            "metadata": {"source": "test", "type": "definition"}
        }
    ]
    
    try:
        # 初始化数据库
        db = VectorDatabase()
        db.initialize()
        
        # 显示数据库信息
        info = db.get_collection_info()
        print(f"数据库信息: {info}")
        
        # 添加测试文档
        db.add_documents(test_documents)
        
        # 更新信息
        info = db.get_collection_info()
        print(f"添加文档后: {info}")
        
        # 测试搜索
        query = "什么是条件概率？"
        results = db.search(query, n_results=2)
        
        print(f"\n搜索查询: {query}")
        print("搜索结果:")
        for i, result in enumerate(results):
            print(f"{i+1}. [距离: {result['distance']:.4f}] {result['text'][:100]}...")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"错误: {e}")