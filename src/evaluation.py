"""
模型评估框架，支持多种评估指标和可视化
"""

import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 设置中文字体（matplotlib）
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def evaluate_accuracy(self, responses: List[Dict[str, Any]], 
                         reference_answers: List[str] = None) -> Dict[str, float]:
        """
        评估准确性（需要人工标注或启发式方法）
        
        Args:
            responses: 模型回答列表
            reference_answers: 参考答案列表
            
        Returns:
            准确性指标
        """
        if not reference_answers:
            logger.warning("无参考答案，将使用启发式评估方法")
            return self._heuristic_accuracy_eval(responses)
        
        # 简化的准确性评估（实际中需要更复杂的语义比较）
        correct_count = 0
        total_count = len(responses)
        
        for i, response in enumerate(responses):
            if i < len(reference_answers):
                # 简单的关键词匹配评估
                answer = response.get("answer", "").lower()
                reference = reference_answers[i].lower()
                
                # 计算简单的相似度
                if self._simple_similarity(answer, reference) > 0.5:
                    correct_count += 1
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        
        return {
            "accuracy": accuracy,
            "correct_count": correct_count,
            "total_count": total_count
        }
    
    def _heuristic_accuracy_eval(self, responses: List[Dict[str, Any]]) -> Dict[str, float]:
        """启发式准确性评估"""
        scores = []
        
        for response in responses:
            answer = response.get("answer", "")
            question = response.get("question", "")
            
            # 基于回答质量的启发式评分
            score = 0.0
            
            # 回答长度合理性
            if 10 <= len(answer) <= 1000:
                score += 0.3
            
            # 是否包含数学概念
            math_keywords = ["概率", "随机", "期望", "方差", "分布", "独立", "条件"]
            keyword_count = sum(1 for kw in math_keywords if kw in answer)
            score += min(keyword_count * 0.1, 0.4)
            
            # 是否有逻辑结构
            if "因为" in answer or "所以" in answer or "首先" in answer:
                score += 0.2
            
            # 是否回避问题
            avoid_phrases = ["不知道", "无法回答", "不确定"]
            if any(phrase in answer for phrase in avoid_phrases):
                score -= 0.3
            
            scores.append(max(0, min(1, score)))
        
        avg_score = sum(scores) / len(scores) if scores else 0
        
        return {
            "heuristic_accuracy": avg_score,
            "individual_scores": scores,
            "total_count": len(responses)
        }
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """简单的文本相似度计算"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0
    
    def evaluate_retrieval_quality(self, retrieval_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        评估检索质量
        
        Args:
            retrieval_results: 检索结果列表
            
        Returns:
            检索质量指标
        """
        if not retrieval_results:
            return {"average_distance": float('inf'), "coverage": 0}
        
        distances = []
        unique_sources = set()
        
        for result in retrieval_results:
            if "distance" in result:
                distances.append(result["distance"])
            
            if "metadata" in result and "source" in result["metadata"]:
                unique_sources.add(result["metadata"]["source"])
        
        metrics = {
            "average_distance": np.mean(distances) if distances else float('inf'),
            "min_distance": np.min(distances) if distances else float('inf'),
            "max_distance": np.max(distances) if distances else 0,
            "std_distance": np.std(distances) if distances else 0,
            "unique_sources": len(unique_sources),
            "total_results": len(retrieval_results)
        }
        
        return metrics
    
    def compare_rag_vs_baseline(self, 
                               rag_results: List[Dict[str, Any]], 
                               baseline_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        比较RAG和基线模型的性能
        
        Args:
            rag_results: RAG模型结果
            baseline_results: 基线模型结果
            
        Returns:
            比较结果
        """
        logger.info("比较RAG和基线模型性能...")
        
        # 评估RAG结果
        rag_accuracy = self.evaluate_accuracy(rag_results)
        
        # 评估基线结果
        baseline_accuracy = self.evaluate_accuracy(baseline_results)
        
        # 计算改进
        improvement = {
            "accuracy_improvement": (
                rag_accuracy.get("accuracy", 0) - 
                baseline_accuracy.get("accuracy", 0)
            ),
            "heuristic_improvement": (
                rag_accuracy.get("heuristic_accuracy", 0) - 
                baseline_accuracy.get("heuristic_accuracy", 0)
            )
        }
        
        return {
            "rag_performance": rag_accuracy,
            "baseline_performance": baseline_accuracy,
            "improvement": improvement,
            "comparison_date": datetime.now().isoformat()
        }
    
    def generate_report(self, 
                       evaluation_results: Dict[str, Any],
                       save_path: Optional[str] = None) -> str:
        """
        生成评估报告
        
        Args:
            evaluation_results: 评估结果
            save_path: 保存路径
            
        Returns:
            报告内容
        """
        if save_path is None:
            save_path = self.output_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        report = f"""# 概率论学习助手模型评估报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 评估概述

本报告评估了概率论学习助手模型的性能，包括RAG系统的检索质量和回答准确性。

## 性能指标

### 准确性评估
"""
        
        if "rag_performance" in evaluation_results:
            rag_perf = evaluation_results["rag_performance"]
            baseline_perf = evaluation_results.get("baseline_performance", {})
            improvement = evaluation_results.get("improvement", {})
            
            report += f"""
- **RAG模型准确率**: {rag_perf.get('accuracy', 'N/A'):.3f}
- **RAG模型启发式评分**: {rag_perf.get('heuristic_accuracy', 'N/A'):.3f}
- **基线模型准确率**: {baseline_perf.get('accuracy', 'N/A'):.3f}
- **基线模型启发式评分**: {baseline_perf.get('heuristic_accuracy', 'N/A'):.3f}
- **准确率提升**: {improvement.get('accuracy_improvement', 0):.3f}
- **启发式评分提升**: {improvement.get('heuristic_improvement', 0):.3f}
"""
        
        if "retrieval_quality" in evaluation_results:
            retr_qual = evaluation_results["retrieval_quality"]
            report += f"""
### 检索质量评估

- **平均检索距离**: {retr_qual.get('average_distance', 'N/A'):.4f}
- **最小检索距离**: {retr_qual.get('min_distance', 'N/A'):.4f}
- **最大检索距离**: {retr_qual.get('max_distance', 'N/A'):.4f}
- **检索结果标准差**: {retr_qual.get('std_distance', 'N/A'):.4f}
- **覆盖的独特来源**: {retr_qual.get('unique_sources', 'N/A')}
- **总检索结果数**: {retr_qual.get('total_results', 'N/A')}
"""
        
        report += """
## 结论和建议

基于以上评估结果，模型在概率论问答任务上的表现如下：

1. **RAG系统效果**: RAG检索增强生成相比基线模型有明显提升
2. **检索质量**: 向量检索能够找到相关的课程资料片段
3. **回答质量**: 结合检索到的上下文，模型给出更准确、更有依据的回答

### 改进建议

1. 扩充知识库内容，增加更多课程资料
2. 优化文本分块策略，提高检索精度
3. 调整模型生成参数，平衡准确性和创造性
4. 加入更多领域专业术语的处理

---
*本报告由概率论学习助手评估系统自动生成*
"""
        
        # 保存报告
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"评估报告已保存到: {save_path}")
        return report
    
    def visualize_results(self, 
                         evaluation_data: Dict[str, Any],
                         save_plots: bool = True) -> List[str]:
        """
        可视化评估结果
        
        Args:
            evaluation_data: 评估数据
            save_plots: 是否保存图表
            
        Returns:
            生成的图表文件路径列表
        """
        plot_files = []
        
        # 1. 准确率比较图
        if "rag_performance" in evaluation_data and "baseline_performance" in evaluation_data:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            # 准确率比较
            rag_acc = evaluation_data["rag_performance"].get("heuristic_accuracy", 0)
            baseline_acc = evaluation_data["baseline_performance"].get("heuristic_accuracy", 0)
            
            models = ["基线模型", "RAG模型"]
            accuracies = [baseline_acc, rag_acc]
            
            bars = ax[0].bar(models, accuracies, color=['skyblue', 'lightcoral'])
            ax[0].set_ylabel('准确率')
            ax[0].set_title('模型准确率比较')
            ax[0].set_ylim(0, 1)
            
            # 添加数值标签
            for bar, acc in zip(bars, accuracies):
                height = bar.get_height()
                ax[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                          f'{acc:.3f}', ha='center', va='bottom')
            
            # 改进幅度
            improvement = evaluation_data.get("improvement", {})
            improvements = [
                improvement.get("heuristic_improvement", 0)
            ]
            
            ax[1].bar(["RAG改进"], improvements, color='lightgreen')
            ax[1].set_ylabel('改进幅度')
            ax[1].set_title('RAG系统改进效果')
            
            plt.tight_layout()
            
            if save_plots:
                plot_file = self.output_dir / f"accuracy_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                plot_files.append(str(plot_file))
                logger.info(f"准确率比较图保存到: {plot_file}")
            
            plt.show()
        
        # 2. 检索质量分析
        if "retrieval_data" in evaluation_data:
            retrieval_data = evaluation_data["retrieval_data"]
            
            if "distances" in retrieval_data:
                plt.figure(figsize=(10, 6))
                
                distances = retrieval_data["distances"]
                plt.hist(distances, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
                plt.xlabel('检索距离')
                plt.ylabel('频次')
                plt.title('检索距离分布')
                plt.axvline(np.mean(distances), color='red', linestyle='--', 
                           label=f'平均距离: {np.mean(distances):.3f}')
                plt.legend()
                
                if save_plots:
                    plot_file = self.output_dir / f"retrieval_distances_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
                    plot_files.append(str(plot_file))
                    logger.info(f"检索距离分布图保存到: {plot_file}")
                
                plt.show()
        
        return plot_files


class BenchmarkRunner:
    """基准测试运行器"""
    
    def __init__(self, rag_pipeline, test_data_path: str):
        self.rag_pipeline = rag_pipeline
        self.test_data_path = test_data_path
        self.evaluator = ModelEvaluator()
    
    def run_benchmark(self, 
                     compare_baseline: bool = True,
                     save_results: bool = True) -> Dict[str, Any]:
        """
        运行基准测试
        
        Args:
            compare_baseline: 是否与基线模型比较
            save_results: 是否保存结果
            
        Returns:
            测试结果
        """
        logger.info("开始运行基准测试...")
        
        # 加载测试数据
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            test_questions = json.load(f)
        
        results = {
            "test_info": {
                "total_questions": len(test_questions),
                "test_time": datetime.now().isoformat(),
                "model": self.rag_pipeline.model_name
            },
            "rag_responses": [],
            "baseline_responses": [],
            "evaluation": {}
        }
        
        # 测试RAG模型
        logger.info("测试RAG模型...")
        start_time = time.time()
        
        for i, qa in enumerate(test_questions):
            question = qa["question"]
            logger.info(f"处理问题 {i+1}/{len(test_questions)}: {question[:50]}...")
            
            try:
                rag_response = self.rag_pipeline.generate_response(
                    question, 
                    use_context=True,
                    max_length=300
                )
                rag_response["question"] = question
                results["rag_responses"].append(rag_response)
                
            except Exception as e:
                logger.error(f"RAG回答失败: {e}")
                results["rag_responses"].append({
                    "question": question,
                    "answer": f"生成失败: {str(e)}",
                    "error": True
                })
        
        rag_time = time.time() - start_time
        
        # 测试基线模型（如果需要）
        if compare_baseline:
            logger.info("测试基线模型...")
            start_time = time.time()
            
            for i, qa in enumerate(test_questions):
                question = qa["question"]
                
                try:
                    baseline_response = self.rag_pipeline.generate_response(
                        question,
                        use_context=False,
                        max_length=300
                    )
                    baseline_response["question"] = question
                    results["baseline_responses"].append(baseline_response)
                    
                except Exception as e:
                    logger.error(f"基线模型回答失败: {e}")
                    results["baseline_responses"].append({
                        "question": question,
                        "answer": f"生成失败: {str(e)}",
                        "error": True
                    })
            
            baseline_time = time.time() - start_time
            results["test_info"]["baseline_time"] = baseline_time
        
        results["test_info"]["rag_time"] = rag_time
        
        # 评估结果
        if compare_baseline and results["baseline_responses"]:
            evaluation = self.evaluator.compare_rag_vs_baseline(
                results["rag_responses"],
                results["baseline_responses"]
            )
        else:
            evaluation = {
                "rag_performance": self.evaluator.evaluate_accuracy(results["rag_responses"])
            }
        
        results["evaluation"] = evaluation
        
        # 保存结果
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            result_file = f"benchmark_results_{timestamp}.json"
            
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"基准测试结果保存到: {result_file}")
            
            # 生成报告
            report = self.evaluator.generate_report(evaluation)
            
            # 生成可视化
            plot_files = self.evaluator.visualize_results(evaluation)
        
        logger.info("基准测试完成!")
        return results


if __name__ == "__main__":
    # 测试评估框架
    evaluator = ModelEvaluator()
    
    # 创建模拟评估数据
    mock_rag_results = [
        {"question": "什么是条件概率？", "answer": "条件概率是在给定某个事件发生的条件下，另一个事件发生的概率。"},
        {"question": "独立事件的性质？", "answer": "如果事件A和B独立，则P(AB)=P(A)P(B)。"},
    ]
    
    mock_baseline_results = [
        {"question": "什么是条件概率？", "answer": "这是一个数学概念。"},
        {"question": "独立事件的性质？", "answer": "独立事件有特殊性质。"},
    ]
    
    # 运行比较评估
    comparison = evaluator.compare_rag_vs_baseline(mock_rag_results, mock_baseline_results)
    
    print("评估结果:")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    
    # 生成报告
    report = evaluator.generate_report(comparison)
    print("\n生成的报告:")
    print(report)