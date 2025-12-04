"""
Reward Model for GRPO
Evaluates the quality of generated answers
"""

import torch
import torch.nn as nn
from typing import List, Dict
import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RewardModel:
    """Base reward model interface"""

    def __init__(self):
        pass

    def compute_reward(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards for generated answers

        Args:
            questions: List of questions
            answers: List of generated answers
            references: List of reference answers

        Returns:
            Tensor of rewards, shape (batch_size,)
        """
        raise NotImplementedError


class MathRewardModel(RewardModel):
    """
    Heuristic reward model for mathematics questions

    Uses multiple criteria:
    - Length appropriateness
    - Mathematical formula presence
    - Key concept coverage
    - Structure and formatting
    """

    def __init__(
        self,
        length_weight: float = 0.2,
        formula_weight: float = 0.3,
        concept_weight: float = 0.3,
        structure_weight: float = 0.2
    ):
        """
        Initialize math reward model

        Args:
            length_weight: Weight for length criterion
            formula_weight: Weight for formula criterion
            concept_weight: Weight for concept coverage
            structure_weight: Weight for structure criterion
        """
        super().__init__()
        self.length_weight = length_weight
        self.formula_weight = formula_weight
        self.concept_weight = concept_weight
        self.structure_weight = structure_weight

    def _compute_length_score(self, answer: str, reference: str) -> float:
        """
        Score based on length appropriateness

        Answers should be of similar length to reference
        """
        answer_len = len(answer.split())
        ref_len = len(reference.split())

        if ref_len == 0:
            return 0.5

        ratio = answer_len / ref_len

        # Optimal range: 0.7 to 1.5 times reference length
        if 0.7 <= ratio <= 1.5:
            score = 1.0
        elif 0.5 <= ratio < 0.7 or 1.5 < ratio <= 2.0:
            score = 0.7
        elif 0.3 <= ratio < 0.5 or 2.0 < ratio <= 3.0:
            score = 0.4
        else:
            score = 0.1

        return score

    def _compute_formula_score(self, answer: str) -> float:
        """
        Score based on presence of mathematical formulas

        Higher score for answers containing proper mathematical notation
        """
        score = 0.0

        # Check for common math symbols and notation
        math_patterns = [
            r'P\(.*?\)',  # Probability notation
            r'E\[.*?\]',  # Expectation
            r'[∫∑∏]',  # Integral, sum, product symbols
            r'[α-ω]',  # Greek letters
            r'\\frac',  # LaTeX fractions
            r'[=≤≥<>≠]',  # Math operators
            r'\^|\*\*',  # Exponents
            r'√',  # Square root
        ]

        matches = sum(1 for pattern in math_patterns if re.search(pattern, answer))

        # Normalize score
        score = min(matches / 4.0, 1.0)

        return score

    def _compute_concept_score(self, answer: str, reference: str) -> float:
        """
        Score based on key concept coverage

        Measures overlap of important terms with reference
        """
        # Extract key terms (words longer than 4 chars, not common words)
        common_words = {
            'the', 'and', 'for', 'that', 'this', 'with', 'from',
            'have', 'has', 'are', 'was', 'were', 'been', 'being'
        }

        def extract_key_terms(text: str) -> set:
            words = re.findall(r'\b\w{5,}\b', text.lower())
            return set(w for w in words if w not in common_words)

        answer_terms = extract_key_terms(answer)
        ref_terms = extract_key_terms(reference)

        if not ref_terms:
            return 0.5

        # Jaccard similarity
        intersection = len(answer_terms & ref_terms)
        union = len(answer_terms | ref_terms)

        if union == 0:
            return 0.0

        return intersection / union

    def _compute_structure_score(self, answer: str) -> float:
        """
        Score based on answer structure and formatting

        Well-structured answers get higher scores
        """
        score = 0.0

        # Check for definition patterns
        if re.search(r'is (defined as|the|a)', answer.lower()):
            score += 0.3

        # Check for formula presentation
        if re.search(r'formula|equation|expression', answer.lower()):
            score += 0.2

        # Check for examples
        if re.search(r'example|for instance|such as', answer.lower()):
            score += 0.2

        # Check for proper sentence structure
        sentences = re.split(r'[.!?]', answer)
        if len([s for s in sentences if len(s.strip()) > 10]) >= 2:
            score += 0.3

        return min(score, 1.0)

    def compute_reward(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str]
    ) -> torch.Tensor:
        """
        Compute rewards for a batch of answers

        Args:
            questions: List of questions
            answers: List of generated answers
            references: List of reference answers

        Returns:
            Tensor of rewards, shape (batch_size,)
        """
        batch_size = len(answers)
        rewards = []

        for i in range(batch_size):
            # Compute individual scores
            length_score = self._compute_length_score(answers[i], references[i])
            formula_score = self._compute_formula_score(answers[i])
            concept_score = self._compute_concept_score(answers[i], references[i])
            structure_score = self._compute_structure_score(answers[i])

            # Weighted combination
            total_reward = (
                self.length_weight * length_score +
                self.formula_weight * formula_score +
                self.concept_weight * concept_score +
                self.structure_weight * structure_score
            )

            rewards.append(total_reward)

        return torch.tensor(rewards, dtype=torch.float32)


class LearnedRewardModel(RewardModel, nn.Module):
    """
    Learned reward model using a classifier

    This can be trained separately on preference data
    """

    def __init__(
        self,
        model_name: str = "deepseek-ai/deepseek-math-7b-instruct",
        hidden_size: int = 768
    ):
        """
        Initialize learned reward model

        Args:
            model_name: Base model for encoding
            hidden_size: Hidden size for reward head
        """
        RewardModel.__init__(self)
        nn.Module.__init__(self)

        from transformers import AutoModel

        self.encoder = AutoModel.from_pretrained(model_name)
        self.reward_head = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # CLS token
        reward = self.reward_head(pooled)
        return reward

    def compute_reward(
        self,
        questions: List[str],
        answers: List[str],
        references: List[str]
    ) -> torch.Tensor:
        """Compute rewards using the learned model"""
        # This would require tokenization and forward pass
        # Placeholder for now
        logger.warning("LearnedRewardModel.compute_reward not fully implemented")
        return torch.zeros(len(answers), dtype=torch.float32)


if __name__ == "__main__":
    # Test reward models
    print("Testing Reward Models...")

    # Sample data
    questions = [
        "What is conditional probability?",
        "Explain Bayes theorem"
    ]

    answers = [
        "Conditional probability P(A|B) is the probability of event A given that B has occurred. The formula is P(A|B) = P(A∩B) / P(B).",
        "Bayes theorem is important"
    ]

    references = [
        "Conditional probability is the probability of an event A occurring given that another event B has already occurred. It is denoted as P(A|B) and calculated using the formula: P(A|B) = P(A ∩ B) / P(B), where P(B) > 0.",
        "Bayes' theorem states that P(A|B) = P(B|A)P(A) / P(B). It allows us to update our beliefs about the probability of an event A given new evidence B."
    ]

    # Test MathRewardModel
    print("\n1. Testing MathRewardModel...")
    reward_model = MathRewardModel()

    rewards = reward_model.compute_reward(questions, answers, references)
    print(f"   Rewards: {rewards}")
    print(f"   Shape: {rewards.shape}")

    for i, (q, a, r, reward) in enumerate(zip(questions, answers, references, rewards)):
        print(f"\n   Question {i+1}: {q[:50]}...")
        print(f"   Answer: {a[:80]}...")
        print(f"   Reward: {reward:.4f}")

    print("\n✅ Reward model test successful!")
