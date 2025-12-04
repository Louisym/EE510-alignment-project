"""
文档处理模块，支持PDF、Word等格式的文本提取和预处理
"""

import os
import re
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

# 导入文档处理库
try:
    import PyPDF2
    import fitz  # PyMuPDF
    HAS_PDF_SUPPORT = True
except ImportError:
    HAS_PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    HAS_DOCX_SUPPORT = True
except ImportError:
    HAS_DOCX_SUPPORT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentExtractor:
    """文档文本提取器"""
    
    @staticmethod
    def extract_from_pdf(file_path: str, use_pymupdf: bool = True) -> str:
        """
        从PDF文件提取文本
        
        Args:
            file_path: PDF文件路径
            use_pymupdf: 是否使用PyMuPDF（更好的文本提取质量）
            
        Returns:
            提取的文本内容
        """
        if not HAS_PDF_SUPPORT:
            raise ImportError("PDF处理库未安装，请安装 PyPDF2 和 PyMuPDF")
        
        text = ""
        
        if use_pymupdf:
            try:
                # 使用PyMuPDF提取
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    text += page.get_text()
                    text += "\n\n"
                doc.close()
                logger.info(f"使用PyMuPDF成功提取PDF: {file_path}")
            except Exception as e:
                logger.warning(f"PyMuPDF提取失败，尝试PyPDF2: {e}")
                use_pymupdf = False
        
        if not use_pymupdf:
            # 使用PyPDF2作为备选
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text += page.extract_text()
                        text += "\n\n"
                logger.info(f"使用PyPDF2成功提取PDF: {file_path}")
            except Exception as e:
                logger.error(f"PDF提取失败: {e}")
                raise
        
        return text
    
    @staticmethod
    def extract_from_docx(file_path: str) -> str:
        """
        从Word文档提取文本
        
        Args:
            file_path: Word文档路径
            
        Returns:
            提取的文本内容
        """
        if not HAS_DOCX_SUPPORT:
            raise ImportError("Word文档处理库未安装，请安装 python-docx")
        
        try:
            doc = DocxDocument(file_path)
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            logger.info(f"成功提取Word文档: {file_path}")
            return text
        
        except Exception as e:
            logger.error(f"Word文档提取失败: {e}")
            raise
    
    @staticmethod
    def extract_from_txt(file_path: str, encoding: str = 'utf-8') -> str:
        """
        从文本文件提取内容
        
        Args:
            file_path: 文本文件路径
            encoding: 文件编码
            
        Returns:
            文件内容
        """
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                content = file.read()
            logger.info(f"成功读取文本文件: {file_path}")
            return content
        except UnicodeDecodeError:
            # 尝试其他编码
            for enc in ['gbk', 'gb2312', 'latin1']:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        content = file.read()
                    logger.info(f"使用编码 {enc} 成功读取文件: {file_path}")
                    return content
                except UnicodeDecodeError:
                    continue
            
            logger.error(f"无法解码文件: {file_path}")
            raise
        except Exception as e:
            logger.error(f"文本文件读取失败: {e}")
            raise


class TextPreprocessor:
    """文本预处理器"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        清理文本内容 (optimized for English PDFs)

        Args:
            text: 原始文本

        Returns:
            清理后的文本
        """
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
        text = re.sub(r'\n\n+', '\n\n', text)  # Replace multiple newlines with double newline

        # Remove common PDF artifacts
        text = re.sub(r'\x0c', '', text)  # Remove form feed characters

        # Keep most characters (English text, numbers, math symbols, punctuation)
        # This is more permissive than the original to preserve mathematical content

        return text.strip()
    
    @staticmethod
    def extract_chapters(text: str) -> Dict[str, str]:
        """
        尝试提取章节信息
        
        Args:
            text: 文本内容
            
        Returns:
            章节字典 {章节名: 内容}
        """
        chapters = {}
        
        # 匹配常见的章节标题模式
        chapter_patterns = [
            r'第[一二三四五六七八九十\d]+章[\s]*([^\n]+)',
            r'Chapter\s+\d+[\s]*([^\n]+)',
            r'\d+\.[\s]*([^\n]+)',
            r'§\d+[\s]*([^\n]+)'
        ]
        
        current_chapter = "前言"
        current_content = ""
        
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否为章节标题
            is_chapter = False
            for pattern in chapter_patterns:
                match = re.match(pattern, line)
                if match:
                    # 保存之前的章节
                    if current_content.strip():
                        chapters[current_chapter] = current_content.strip()
                    
                    # 开始新章节
                    current_chapter = line
                    current_content = ""
                    is_chapter = True
                    break
            
            if not is_chapter:
                current_content += line + "\n"
        
        # 保存最后一个章节
        if current_content.strip():
            chapters[current_chapter] = current_content.strip()
        
        return chapters if len(chapters) > 1 else {"全文": text}
    
    @staticmethod
    def extract_formulas(text: str) -> List[str]:
        """
        提取数学公式（简单版本）
        
        Args:
            text: 文本内容
            
        Returns:
            公式列表
        """
        # 匹配LaTeX风格的公式
        latex_formulas = re.findall(r'\$\$(.+?)\$\$', text, re.DOTALL)
        inline_formulas = re.findall(r'\$(.+?)\$', text)
        
        # 匹配其他可能的数学表达式
        math_expressions = re.findall(r'[A-Za-z]\s*[=≤≥<>]\s*[^。！？\n]+', text)
        
        formulas = []
        formulas.extend(latex_formulas)
        formulas.extend(inline_formulas)
        formulas.extend(math_expressions)
        
        return list(set(formulas))  # 去重


class CourseDataProcessor:
    """课程数据处理器，专门处理概率论课程相关文档"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.extractor = DocumentExtractor()
        self.preprocessor = TextPreprocessor()
        
    def process_directory(self, input_dir: str, output_file: str = None) -> List[Dict[str, Any]]:
        """
        处理目录下的所有文档
        
        Args:
            input_dir: 输入目录
            output_file: 输出JSON文件路径（可选）
            
        Returns:
            处理后的文档列表
        """
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"输入目录不存在: {input_dir}")
        
        documents = []
        
        # 支持的文件格式
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md'}
        
        for file_path in input_path.rglob('*'):
            if file_path.suffix.lower() in supported_extensions:
                try:
                    doc_data = self.process_single_file(str(file_path))
                    documents.extend(doc_data)
                except Exception as e:
                    logger.error(f"处理文件失败 {file_path}: {e}")
        
        # 保存到文件
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(documents, f, ensure_ascii=False, indent=2)
            logger.info(f"处理结果保存到: {output_file}")
        
        logger.info(f"成功处理 {len(documents)} 个文档")
        return documents
    
    def process_single_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        处理单个文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            处理后的文档片段列表
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        logger.info(f"处理文件: {file_path}")
        
        # 提取文本
        if extension == '.pdf':
            raw_text = self.extractor.extract_from_pdf(str(file_path))
        elif extension in ['.docx', '.doc']:
            raw_text = self.extractor.extract_from_docx(str(file_path))
        elif extension in ['.txt', '.md']:
            raw_text = self.extractor.extract_from_txt(str(file_path))
        else:
            raise ValueError(f"不支持的文件格式: {extension}")
        
        # 清理文本
        cleaned_text = self.preprocessor.clean_text(raw_text)
        
        # 提取章节
        chapters = self.preprocessor.extract_chapters(cleaned_text)
        
        # 生成文档片段
        documents = []
        for chapter_name, chapter_content in chapters.items():
            # 将长章节分块
            chunks = self._chunk_text(chapter_content, max_length=800, overlap=100)
            
            for i, chunk in enumerate(chunks):
                doc_id = f"{file_path.stem}_{chapter_name}_{i}"
                
                metadata = {
                    "source_file": file_path.name,
                    "file_type": extension[1:],
                    "chapter": chapter_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                
                documents.append({
                    "id": doc_id,
                    "text": chunk,
                    "metadata": metadata
                })
        
        return documents
    
    def _chunk_text(self, text: str, max_length: int = 1500, overlap: int = 200) -> List[str]:
        """
        智能文本分块，尽量在句子边界分割 (optimized for English text)

        Args:
            text: 输入文本
            max_length: 最大块长度（增加到1500以适应英文内容）
            overlap: 重叠长度

        Returns:
            文本块列表
        """
        if len(text) <= max_length:
            return [text]

        # Split by sentences (English punctuation)
        sentences = re.split(r'[.!?\n]+', text)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # Check if adding current sentence would exceed length limit
            if len(current_chunk) + len(sentence) + 2 <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk
                if len(sentence) > max_length:
                    # Sentence itself is too long, force split by characters
                    for i in range(0, len(sentence), max_length):
                        chunks.append(sentence[i:i+max_length])
                    current_chunk = ""
                else:
                    current_chunk = sentence + ". "

        # Add last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        # Add overlap
        overlapped_chunks = []
        for i, chunk in enumerate(chunks):
            if i == 0:
                overlapped_chunks.append(chunk)
            else:
                # Add ending of previous chunk as overlap
                prev_overlap = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
                overlapped_chunk = prev_overlap + " " + chunk
                overlapped_chunks.append(overlapped_chunk)

        return overlapped_chunks


if __name__ == "__main__":
    # 测试文档处理
    processor = CourseDataProcessor()
    
    # 创建测试目录和文件
    os.makedirs("./data/test_docs", exist_ok=True)
    
    # 创建测试文档
    test_content = """第一章 概率论基础

概率论是数学的一个分支，研究随机现象的数量规律。

条件概率的定义：设A、B是两个事件，且P(B)>0，则称P(A|B) = P(AB)/P(B)为事件B发生的条件下事件A发生的概率。

第二章 随机变量

随机变量是定义在样本空间上的实值函数。连续型随机变量具有概率密度函数。
"""
    
    with open("./data/test_docs/test_textbook.txt", "w", encoding="utf-8") as f:
        f.write(test_content)
    
    try:
        # 处理测试文档
        documents = processor.process_directory("./data/test_docs", "./data/processed_docs.json")
        
        print(f"处理完成，共 {len(documents)} 个文档片段：")
        for doc in documents:
            print(f"- ID: {doc['id']}")
            print(f"  章节: {doc['metadata']['chapter']}")
            print(f"  内容预览: {doc['text'][:100]}...")
            print()
            
    except Exception as e:
        print(f"处理失败: {e}")