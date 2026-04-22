"""
Extended Forgetting Experiment v2: 50 interactions with 64 memory slots.
Tests threshold escalation and selective filtering under pressure.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.memory_augmented_model_v2 import CultivatedMemoryModel


def run_extended_experiment(
    model_key: str = "Qwen/Qwen3.5-4B",
    memory_slots: int = 64,
):
    """Run 50-interaction extended experiment."""

    model = CultivatedMemoryModel(
        base_model_key=model_key,
        memory_slots=memory_slots,
        memory_dim=512,
        controller_mode="clean",
    )

    all_prompts = [
        # Math 20
        "What is 5 + 3?",
        "Calculate 10 * 2.",
        "What is 15 - 7?",
        "What is 8 / 2?",
        "Calculate 3 + 3.",
        "What is 100 + 200?",
        "Calculate 7 * 8.",
        "What is 50 - 25?",
        "What is 17 + 29?",
        "Calculate 6 * 7.",
        "What is 2^10?",
        "Calculate sqrt(16).",
        "What is 33 * 3?",
        "Calculate 1000 / 8.",
        "What is 99 + 1?",
        "Calculate 11 * 11.",
        "What is 45 - 18?",
        "Calculate 72 / 9.",
        "What is 13 + 27?",
        "Calculate 8 * 12.",
        # Science 15
        "What is photosynthesis?",
        "Explain gravity.",
        "What is DNA?",
        "How does battery work?",
        "What is entropy?",
        "Explain quantum superposition.",
        "What are black holes?",
        "How do vaccines work?",
        "What is climate change?",
        "Explain nuclear fission.",
        "What is relativity?",
        "Explain evolution.",
        "What is an atom?",
        "How do magnets work?",
        "What is the Big Bang?",
        # Facts 10
        "Capital of Japan?",
        "Who painted Mona Lisa?",
        "Tallest mountain?",
        "When WW2 end?",
        "Speed of light?",
        "Who invented telephone?",
        "Largest ocean?",
        "Language in Brazil?",
        "Currency of UK?",
        "Who wrote 1984?",
        # Repetitive 5
        "What is 2 + 2?",
        "What is 3 + 3?",
        "What is 4 + 4?",
        "What is 5 + 5?",
        "What is 6 + 6?",
    ]

    results = []
    total = len(all_prompts)

    for i, prompt in enumerate(all_prompts):
        result = model.generate_with_cultivated_memory(
            prompt,
            max_new_tokens=64,
            return_debug=True,
        )
        stats = model.get_garden_stats()

        threshold = result.get("dynamic_threshold")
        threshold_text = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else "N/A"

        print(
            f"[{i + 1}/{total}] "
            f"stored={result.get('stored')} "
            f"cap={stats['memory']['activation_ratio']:.1%} "
            f"thresh={threshold_text}"
        )

        results.append(
            {
                "prompt": prompt,
                "stored": result.get("stored"),
                "capacity": stats["memory"]["activation_ratio"],
                "threshold": result.get("dynamic_threshold"),
                "novelty": result.get("novelty"),
                "store_score": result.get("store_score"),
            }
        )

    final = model.get_garden_stats()
    print(f"\nFINAL: Capacity {final['memory']['activation_ratio']:.1%}")
    print(f"Active: {final['memory']['active_slots']}/{memory_slots}")

    os.makedirs("results", exist_ok=True)
    filename = f"results/extended_{memory_slots}.json"
    with open(filename, "w", encoding="utf-8") as handle:
        json.dump({"final": final, "interactions": results}, handle, indent=2, default=str)

    print(f"Results saved to {filename}")
    return final


if __name__ == "__main__":
    run_extended_experiment()
