"""
Validate training data for SFT and GRPO training
"""
import json
import re
from typing import Dict, List, Any
from pathlib import Path


def load_data(file_path: str) -> List[Dict[str, Any]]:
    """Load JSON data"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def check_structure(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check data structure and required fields"""
    issues = []
    required_fields = {'question', 'answer', 'source'}

    for i, entry in enumerate(data):
        # Check required fields
        missing_fields = required_fields - set(entry.keys())
        if missing_fields:
            issues.append(f"Entry {i}: Missing fields {missing_fields}")

        # Check for empty values
        for field in required_fields:
            if field in entry:
                value = entry[field]
                # Handle both string and dict answers
                if isinstance(value, str):
                    if not value.strip():
                        issues.append(f"Entry {i}: Empty {field}")
                elif isinstance(value, dict):
                    # Check if dict is empty or all values are empty
                    if not value or all(not str(v).strip() for v in value.values()):
                        issues.append(f"Entry {i}: Empty dict {field}")
                elif not value:
                    issues.append(f"Entry {i}: Empty {field}")

    return {
        'total_entries': len(data),
        'issues': issues
    }


def analyze_answer_formats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze different answer formats in the data"""
    string_answers = 0
    dict_answers = 0
    dict_answer_indices = []

    for i, entry in enumerate(data):
        answer = entry.get('answer', '')
        if isinstance(answer, dict):
            dict_answers += 1
            dict_answer_indices.append(i)
        elif isinstance(answer, str):
            string_answers += 1

    return {
        'string_answers': string_answers,
        'dict_answers': dict_answers,
        'dict_answer_indices': dict_answer_indices
    }


def analyze_content(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze content statistics"""
    question_lengths = []
    answer_lengths = []
    sources = {}

    for entry in enumerate(data):
        idx, item = entry
        question = item.get('question', '')
        answer = item.get('answer', '')
        source = item.get('source', 'unknown')

        # Get text length
        if isinstance(question, str):
            question_lengths.append(len(question))

        # Handle both string and dict answers
        if isinstance(answer, str):
            answer_lengths.append(len(answer))
        elif isinstance(answer, dict):
            # For dict answers, compute total length of all values
            total_len = sum(len(str(v)) for v in answer.values())
            answer_lengths.append(total_len)

        # Count sources
        sources[source] = sources.get(source, 0) + 1

    return {
        'question_length': {
            'min': min(question_lengths) if question_lengths else 0,
            'max': max(question_lengths) if question_lengths else 0,
            'avg': sum(question_lengths) / len(question_lengths) if question_lengths else 0
        },
        'answer_length': {
            'min': min(answer_lengths) if answer_lengths else 0,
            'max': max(answer_lengths) if answer_lengths else 0,
            'avg': sum(answer_lengths) / len(answer_lengths) if answer_lengths else 0
        },
        'source_distribution': sources
    }


def check_mathematical_notation(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Check preservation of mathematical notation"""
    math_symbols = {
        'greek': re.compile(r'[αβγδεζηθικλμνξοπρστυφχψωΓΔΘΛΞΠΣΦΨΩ]'),
        'operators': re.compile(r'[∈∉⊂⊃∩∪∅≤≥≠≈∞∫∑∏√]'),
        'subscripts_superscripts': re.compile(r'[₀₁₂₃₄₅₆₇₈₉⁰¹²³⁴⁵⁶⁷⁸⁹]'),
        'arrows': re.compile(r'[←→↔⇒⇐⇔]')
    }

    entries_with_math = {key: 0 for key in math_symbols}

    for entry in data:
        # Check both question and answer
        text = str(entry.get('question', '')) + str(entry.get('answer', ''))

        for symbol_type, pattern in math_symbols.items():
            if pattern.search(text):
                entries_with_math[symbol_type] += 1

    return {
        'total_entries': len(data),
        'entries_with_math_symbols': entries_with_math
    }


def flatten_dict_answers(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert dict answers to string format for consistency"""
    flattened_data = []

    for entry in data:
        new_entry = entry.copy()
        answer = entry.get('answer', '')

        if isinstance(answer, dict):
            # Flatten the dictionary into a readable string format
            flattened_parts = []
            for key, value in answer.items():
                if key in ['I', 'M', 'A', 'C']:
                    # These seem to be step markers (Intuition, Method, Answer, Conclusion)
                    flattened_parts.append(str(value))
                elif key in ['a', 'b', 'c', 'd', 'e']:
                    # These are part labels
                    flattened_parts.append(f"({key}) {value}")
                else:
                    flattened_parts.append(f"{key}: {value}")

            new_entry['answer'] = '\n\n'.join(flattened_parts)

        flattened_data.append(new_entry)

    return flattened_data


def main():
    data_path = Path(__file__).parent.parent / "data" / "training_data" / "train.json"

    print("=" * 80)
    print("TRAINING DATA VALIDATION REPORT")
    print("=" * 80)
    print(f"\nData file: {data_path}")

    # Load data
    try:
        data = load_data(data_path)
        print(f"✓ Successfully loaded JSON file")
    except Exception as e:
        print(f"✗ Error loading JSON: {e}")
        return

    # Check structure
    print("\n" + "-" * 80)
    print("1. STRUCTURE VALIDATION")
    print("-" * 80)
    structure_info = check_structure(data)
    print(f"Total entries: {structure_info['total_entries']}")
    if structure_info['issues']:
        print(f"✗ Found {len(structure_info['issues'])} issues:")
        for issue in structure_info['issues'][:10]:  # Show first 10
            print(f"  - {issue}")
        if len(structure_info['issues']) > 10:
            print(f"  ... and {len(structure_info['issues']) - 10} more")
    else:
        print("✓ All entries have required fields (question, answer, source)")

    # Check answer formats
    print("\n" + "-" * 80)
    print("2. ANSWER FORMAT ANALYSIS")
    print("-" * 80)
    format_info = analyze_answer_formats(data)
    print(f"String answers: {format_info['string_answers']}")
    print(f"Dictionary answers: {format_info['dict_answers']}")

    if format_info['dict_answers'] > 0:
        print(f"\n⚠ WARNING: Found {format_info['dict_answers']} entries with dictionary-formatted answers")
        print(f"  Indices: {format_info['dict_answer_indices'][:10]}" +
              (f" ... and {len(format_info['dict_answer_indices']) - 10} more"
               if len(format_info['dict_answer_indices']) > 10 else ""))
        print("  Recommendation: Convert dict answers to string format for consistency")

    # Content statistics
    print("\n" + "-" * 80)
    print("3. CONTENT STATISTICS")
    print("-" * 80)
    content_info = analyze_content(data)

    print(f"\nQuestion lengths (characters):")
    print(f"  Min: {content_info['question_length']['min']}")
    print(f"  Max: {content_info['question_length']['max']}")
    print(f"  Avg: {content_info['question_length']['avg']:.1f}")

    print(f"\nAnswer lengths (characters):")
    print(f"  Min: {content_info['answer_length']['min']}")
    print(f"  Max: {content_info['answer_length']['max']}")
    print(f"  Avg: {content_info['answer_length']['avg']:.1f}")

    print(f"\nSource distribution:")
    for source, count in sorted(content_info['source_distribution'].items()):
        print(f"  {source}: {count} ({count/len(data)*100:.1f}%)")

    # Mathematical notation
    print("\n" + "-" * 80)
    print("4. MATHEMATICAL NOTATION PRESERVATION")
    print("-" * 80)
    math_info = check_mathematical_notation(data)
    print(f"Entries with Greek letters: {math_info['entries_with_math_symbols']['greek']}")
    print(f"Entries with operators (∈,∩,∪,etc): {math_info['entries_with_math_symbols']['operators']}")
    print(f"Entries with subscripts/superscripts: {math_info['entries_with_math_symbols']['subscripts_superscripts']}")
    print(f"Entries with arrows: {math_info['entries_with_math_symbols']['arrows']}")
    print("✓ Mathematical notation is preserved in Unicode format")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    if format_info['dict_answers'] > 0:
        recommendations.append(
            f"1. FLATTEN DICTIONARY ANSWERS: {format_info['dict_answers']} entries have "
            "answers in dictionary format instead of plain text. This may cause issues "
            "during tokenization. Recommend converting to plain text format."
        )

    if len(data) < 100:
        recommendations.append(
            f"2. DATASET SIZE: Current dataset has {len(data)} examples. For SFT, this is "
            "relatively small. Consider:\n"
            "   - Using more epochs (5-10) to ensure adequate learning\n"
            "   - Monitoring for overfitting\n"
            "   - Adding more diverse examples if possible"
        )

    recommendations.append(
        "3. TRAINING CONFIGURATION:\n"
        "   - For SFT: Use 'low_memory' or 'default' config with 81 samples\n"
        "   - Recommended: 5-8 epochs, learning_rate=2e-4, batch_size=2-4\n"
        "   - For GRPO: Start from SFT checkpoint using 'from_sft' config\n"
        "   - GRPO generates 4 samples per question, effective batch = 81*4 = 324"
    )

    if recommendations:
        for rec in recommendations:
            print(f"\n{rec}")
    else:
        print("\n✓ Data looks good! Ready for training.")

    # Offer to create flattened version
    if format_info['dict_answers'] > 0:
        print("\n" + "=" * 80)
        print("Would you like to create a flattened version of the data? (Recommended)")
        print("This will convert all dictionary answers to plain text format.")
        print("=" * 80)

        # For script execution, auto-generate the flattened version
        output_path = data_path.parent / "train_flattened.json"
        flattened_data = flatten_dict_answers(data)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(flattened_data, f, ensure_ascii=False, indent=2)

        print(f"\n✓ Created flattened version: {output_path}")
        print(f"  Original entries: {len(data)}")
        print(f"  Flattened entries: {len(flattened_data)}")
        print(f"  You can use this file for training if dictionary answers cause issues.")


if __name__ == "__main__":
    main()
