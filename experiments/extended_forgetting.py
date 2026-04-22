import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.memory_augmented_model_v2 import CultivatedMemoryModel


def extended_test():
    model = CultivatedMemoryModel(
        base_model_key="Qwen/Qwen3.5-4B",
        memory_slots=64,
        memory_dim=512,
        controller_mode="clean",
    )

    prompts = [
        "What is 5 + 3?", "Calculate 10 * 2.", "What is 15 - 7?",
        "What is 8 / 2?", "Calculate 3 + 3.", "What is 100 + 200?",
        "Calculate 7 * 8.", "What is 50 - 25?", "What is 17 + 29?",
        "Calculate 6 * 7.",
        "What is photosynthesis?", "Explain gravity simply.",
        "What is DNA?", "How does a battery work?",
        "What is entropy?",
        "Write a Python function to reverse a string.",
        "Explain recursion.", "What is SQL JOIN?",
        "Explain Docker.", "What is REST API?",
        "Capital of Japan?", "Who painted Mona Lisa?",
        "What is the tallest mountain?", "When did WW2 end?",
        "Who invented telephone?",
        "What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?",
        "What is 5 + 5?", "What is 6 + 6?",
    ]

    results = []
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        print(f"\n[{i + 1}/{total}] {prompt[:40]}...")

        result = model.generate_with_cultivated_memory(
            prompt,
            max_new_tokens=64,
            return_debug=True,
        )

        stats = model.get_garden_stats()
        print(
            f"  Stored: {result.get('stored')} | "
            f"Capacity: {stats['memory']['activation_ratio']:.1%} | "
            f"Threshold: {result.get('dynamic_threshold', 'N/A')}"
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
    print(f"\n{'=' * 60}")
    print(f"FINAL: Capacity {final['memory']['activation_ratio']:.1%}")
    print(f"Active: {final['memory']['active_slots']}/64")
    print(f"Decay: {final['stability_decay_rate']:.4f}")

    os.makedirs("results", exist_ok=True)
    with open("results/extended_64.json", "w", encoding="utf-8") as handle:
        json.dump({"final": final, "interactions": results}, handle, indent=2, default=str)

    print("Results saved to results/extended_64.json")
    return final


if __name__ == "__main__":
    extended_test()
