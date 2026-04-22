"""
Final stability test: 100 interactions on 1024 slots with decay enabled.
Goal: validate long-run stability, threshold escalation, and repetitive rejection.
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.memory_augmented_model_v2 import CultivatedMemoryModel


def stability_test():
    model = CultivatedMemoryModel(
        base_model_key="Qwen/Qwen3.5-4B",
        memory_slots=1024,
        memory_dim=512,
        controller_mode="clean",
    )

    # Explicitly keep decay on for this final stability run.
    if hasattr(model.controller, "decay_rate"):
        model.controller.decay_rate.data = torch.tensor(0.008, device=model.controller.decay_rate.device)

    prompts = [
        # 40 Math
        "What is 5 + 3?", "Calculate 10 * 2.", "What is 15 - 7?", "What is 8 / 2?",
        "Calculate 3 + 3.", "What is 100 + 200?", "Calculate 7 * 8.", "What is 50 - 25?",
        "What is 17 + 29?", "Calculate 6 * 7.", "What is 2^10?", "Calculate sqrt(16).",
        "What is 33 * 3?", "Calculate 1000 / 8.", "What is 99 + 1?", "Calculate 11 * 11.",
        "What is 45 - 18?", "Calculate 72 / 9.", "What is 13 + 27?", "Calculate 8 * 12.",
        "Solve x^2 - 5x + 6 = 0", "What is derivative of x^3?", "Calculate 7!",
        "What is log(100)?", "Solve 2x + 3 = 11", "Area of circle radius 5?",
        "Calculate 3^5 + 2^4", "What is sin(30°)?", "Solve x + 5 = 12", "Calculate 15 * 15.",
        "What is 20% of 50?", "Calculate cube root of 27", "What is 2^8?", "Solve 3x = 21",
        "What is perimeter of square side 4?", "Calculate 1/2 + 1/3", "What is 0! ?",
        "Solve x^2 = 16", "Calculate 9 * 8 - 7",

        # 30 Science
        "What is photosynthesis?", "Explain gravity.", "What is DNA?", "How does battery work?",
        "What is entropy?", "Explain quantum superposition.", "What are black holes?",
        "How do vaccines work?", "What is climate change?", "Explain nuclear fission.",
        "What is relativity?", "Explain evolution.", "What is an atom?", "How do magnets work?",
        "What is the Big Bang?", "Explain plate tectonics.", "What is H2O?", "How do lasers work?",
        "What is electricity?", "Explain gravity waves.", "What is dark matter?", "How do planes fly?",
        "What is the immune system?", "Explain photosynthesis equation.", "What is friction?",
        "How do cameras work?", "What is sound?", "Explain DNA replication.", "What is a virus?",
        "How do muscles work?",

        # 20 Facts
        "Capital of Japan?", "Who painted Mona Lisa?", "Tallest mountain?", "When WW2 end?",
        "Speed of light?", "Who invented telephone?", "Largest ocean?", "Language in Brazil?",
        "Currency of UK?", "Who wrote 1984?", "Capital of Australia?", "Who discovered America?",
        "What is the longest river?", "Who invented internet?", "What is the smallest country?",
        "Who wrote Romeo and Juliet?", "What is the hottest planet?", "Who invented penicillin?",
        "What is the deepest ocean trench?", "Who was the first astronaut?",

        # 10 Repetitive
        "What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?", "What is 5 + 5?",
        "What is 6 + 6?", "What is 7 + 7?", "What is 8 + 8?", "What is 9 + 9?",
        "What is 10 + 10?", "What is 1 + 1?",
    ]

    results = []
    total = len(prompts)

    for i, prompt in enumerate(prompts):
        result = model.generate_with_cultivated_memory(prompt, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()

        threshold = result.get("dynamic_threshold")
        threshold_text = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else "N/A"
        decay_text = f"{model.controller.decay_rate.item():.4f}" if hasattr(model.controller, "decay_rate") else "N/A"

        print(
            f"[{i + 1}/{total}] stored={result.get('stored')} "
            f"cap={stats['memory']['activation_ratio']:.1%} "
            f"thresh={threshold_text} decay={decay_text}"
        )

        results.append(
            {
                "prompt": prompt,
                "stored": result.get("stored"),
                "capacity": stats["memory"]["activation_ratio"],
                "threshold": result.get("dynamic_threshold"),
                "novelty": result.get("novelty"),
                "decay": float(model.controller.decay_rate.item()) if hasattr(model.controller, "decay_rate") else None,
            }
        )

    final = model.get_garden_stats()
    print(f"\n{'=' * 60}")
    print(f"FINAL: Capacity {final['memory']['activation_ratio']:.1%}")
    print(f"Active: {final['memory']['active_slots']}/1024")
    print(f"Decay: {final['stability_decay_rate']:.4f}")
    print(f"Store rate: {sum(1 for r in results if r['stored'])}/100")

    os.makedirs("results", exist_ok=True)
    with open("results/stability_1024.json", "w", encoding="utf-8") as handle:
        json.dump({"final": final, "interactions": results}, handle, indent=2, default=str)

    print("Results saved to results/stability_1024.json")
    return final


if __name__ == "__main__":
    stability_test()
