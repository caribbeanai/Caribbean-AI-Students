"""
=============================================================================
 CARIBBEAN AI ACADEMY - GRADUATES CAPSTONE PROJECT
 Caribbean LLM - Evaluation Script
 By Adrian Dunkley
=============================================================================

 Yuh model is only as good as yuh evaluation!
 Dis script tests di Caribbean LLM on multiple benchmarks:
 - Factual accuracy on Caribbean knowledge
 - Cultural sensitivity
 - Language understanding (Creole/Patois)
 - Response quality

 Usage:
   python evaluate.py --model_path ./output/caribbean-llm-lora
=============================================================================
"""

import argparse
import json
from typing import Dict, List, Tuple

# =============================================================================
# EVALUATION BENCHMARKS
# =============================================================================

CARIBBEAN_EVAL_BENCHMARK = [
    {
        "question": "Who is the fastest man in history and where is he from?",
        "expected_keywords": ["usain", "bolt", "jamaica"],
        "category": "sports"
    },
    {
        "question": "What happened during the Haitian Revolution?",
        "expected_keywords": ["slave", "revolt", "toussaint", "1804", "independence"],
        "category": "history"
    },
    {
        "question": "What is CARICOM and when was it established?",
        "expected_keywords": ["caribbean", "community", "1973", "integration"],
        "category": "politics"
    },
    {
        "question": "Describe the significance of cricket in West Indian culture.",
        "expected_keywords": ["cricket", "west indies", "barbados", "trinidad", "lara"],
        "category": "sports"
    },
    {
        "question": "What languages are spoken in the Caribbean?",
        "expected_keywords": ["english", "creole", "spanish", "french", "dutch", "patois"],
        "category": "language"
    },
    {
        "question": "What is carnival in Trinidad and Tobago?",
        "expected_keywords": ["carnival", "calypso", "soca", "steelpan", "mas"],
        "category": "culture"
    },
    {
        "question": "How does climate change affect Caribbean islands?",
        "expected_keywords": ["hurricane", "sea level", "coral", "vulnerable"],
        "category": "environment"
    },
    {
        "question": "Tell me about Guyana's oil discovery.",
        "expected_keywords": ["oil", "exxon", "stabroek", "economy", "growth"],
        "category": "economics"
    },
]


# =============================================================================
# EVALUATION METRICS
# =============================================================================

def keyword_accuracy(response: str, expected: List[str]) -> float:
    """Measure how many expected keywords appear in response."""
    response_lower = response.lower()
    hits = sum(1 for kw in expected if kw in response_lower)
    return hits / len(expected) if expected else 0


def cultural_sensitivity_score(response: str) -> Dict:
    """Evaluate cultural appropriateness of response."""
    positive = ["caribbean", "island", "nation", "culture", "heritage",
                "community", "tradition", "independence", "resilience", "diverse"]
    negative = ["primitive", "third world", "backward", "undeveloped",
                "poor country", "banana republic", "uncivilized"]

    response_lower = response.lower()
    pos = sum(1 for w in positive if w in response_lower)
    neg = sum(1 for w in negative if w in response_lower)

    return {
        "positive_markers": pos,
        "negative_markers": neg,
        "score": max(0, (pos - neg * 5)) / len(positive),
        "passed": neg == 0
    }


def response_quality_score(response: str) -> Dict:
    """Basic response quality metrics."""
    words = response.split()
    sentences = [s.strip() for s in response.split('.') if s.strip()]

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_sentence_length": len(words) / max(1, len(sentences)),
        "is_adequate_length": 20 < len(words) < 500,
    }


# =============================================================================
# EVALUATION RUNNER
# =============================================================================

def evaluate_model(model_path: str = None, responses: Dict[str, str] = None):
    """
    Run full evaluation suite.

    TODO: When model is trained, load it and generate actual responses.
    For now, pass in responses dict for testing.
    """
    print("=" * 60)
    print(" CARIBBEAN LLM EVALUATION")
    print("=" * 60)

    # If no responses provided, use placeholders
    if responses is None:
        responses = {}
        for item in CARIBBEAN_EVAL_BENCHMARK:
            # TODO: Replace with actual model inference
            responses[item["question"]] = (
                f"[Model response would go here for: {item['question']}]"
            )

    results = []
    for item in CARIBBEAN_EVAL_BENCHMARK:
        q = item["question"]
        response = responses.get(q, "")

        kw_acc = keyword_accuracy(response, item["expected_keywords"])
        cultural = cultural_sensitivity_score(response)
        quality = response_quality_score(response)

        result = {
            "question": q,
            "category": item["category"],
            "keyword_accuracy": kw_acc,
            "cultural_score": cultural["score"],
            "cultural_passed": cultural["passed"],
            "adequate_length": quality["is_adequate_length"],
        }
        results.append(result)

        print(f"\nQ: {q[:60]}...")
        print(f"  Keyword accuracy: {kw_acc:.1%}")
        print(f"  Cultural score:   {cultural['score']:.2f}")
        print(f"  Cultural passed:  {cultural['passed']}")

    # Summary
    print("\n" + "=" * 60)
    print(" EVALUATION SUMMARY")
    print("=" * 60)
    avg_kw = sum(r["keyword_accuracy"] for r in results) / len(results)
    avg_cultural = sum(r["cultural_score"] for r in results) / len(results)
    all_passed = all(r["cultural_passed"] for r in results)

    print(f"  Average keyword accuracy: {avg_kw:.1%}")
    print(f"  Average cultural score:   {avg_cultural:.2f}")
    print(f"  All cultural checks pass: {all_passed}")
    print(f"  Total questions evaluated: {len(results)}")

    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Caribbean LLM")
    parser.add_argument("--model_path", type=str, default="./output/caribbean-llm-lora")
    args = parser.parse_args()

    print("=" * 60)
    print(" CARIBBEAN AI ACADEMY - MODEL EVALUATION")
    print(" By Adrian Dunkley")
    print("=" * 60)

    # Run evaluation (simulation mode without actual model)
    results = evaluate_model(args.model_path)

    print("\n" + "=" * 60)
    print(" Evaluation complete! Review results and iterate.")
    print(" Remember: test wid real Caribbean people too!")
    print("=" * 60)
