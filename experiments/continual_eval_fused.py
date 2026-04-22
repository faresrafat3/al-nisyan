# Continual Learning Benchmark for Al-Nisyan v3
# Uses embedding-level memory fusion (soft prefix)

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.memory_augmented_model_v3 import CultivatedMemoryModelV3


def continual_learning_benchmark_fused():
    print("=" * 70)
    print("AL-NISYAN v3: CONTINUAL LEARNING BENCHMARK")
    print("=" * 70)

    model = CultivatedMemoryModelV3(
        base_model_key="Qwen/Qwen3.5-4B",
        memory_slots=256,
        memory_dim=512,
        controller_mode="clean",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    math_problems = [
        "What is 5 + 3?", "Calculate 10 times 2.",
        "What is 15 minus 7?", "What is 8 divided by 2?",
        "Calculate 3 plus 3.", "What is 100 plus 200?",
        "Calculate 7 times 8.", "What is 50 minus 25?",
        "What is 17 plus 29?", "Calculate 6 times 7.",
        "What is 2 to the power of 10?", "Calculate square root of 16.",
        "What is 33 times 3?", "Calculate 1000 divided by 8.",
        "What is 99 plus 1?", "Calculate 11 times 11.",
        "What is 45 minus 18?", "Calculate 72 divided by 9.",
        "What is 13 plus 27?", "Calculate 8 times 12.",
        "Solve x squared minus 5x plus 6 equals zero.",
        "What is the derivative of x cubed?",
    ]

    science_problems = [
        "What is photosynthesis?", "Explain gravity simply.",
        "What is DNA?", "How does a battery work?",
        "What is entropy?", "Explain quantum superposition simply.",
        "What are black holes?", "How do vaccines work?",
        "What is climate change?", "Explain nuclear fission.",
        "What is relativity?", "Explain evolution briefly.",
        "What is an atom?", "How do magnets work?",
    ]

    math_results = []
    for i, problem in enumerate(math_problems):
        result = model.generate_with_fused_memory(problem, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()
        print(f"[M{i+1:02d}] stored={result.get('stored')} cap={stats['memory']['activation_ratio']:.3f}")
        math_results.append({"problem": problem, "stored": result.get("stored"), "capacity": stats["memory"]["activation_ratio"]})

    science_results = []
    for i, problem in enumerate(science_problems):
        result = model.generate_with_fused_memory(problem, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()
        print(f"[S{i+1:02d}] stored={result.get('stored')} cap={stats['memory']['activation_ratio']:.3f}")
        science_results.append({"problem": problem, "stored": result.get("stored"), "capacity": stats["memory"]["activation_ratio"]})

    math_test = [
        "What is 5 + 3?",
        "Calculate 6 times 7.",
        "What is 2 to the power of 10?",
        "What is 100 plus 200?",
        "Calculate 11 times 11.",
    ]

    forward_results = []
    for i, problem in enumerate(math_test):
        result = model.generate_with_fused_memory(problem, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()
        print(f"[F{i+1}] stored={result.get('stored')} cap={stats['memory']['activation_ratio']:.3f} novelty={result.get('novelty', 0):.3f}")
        forward_results.append({"problem": problem, "stored": result.get("stored"), "capacity": stats["memory"]["activation_ratio"], "novelty": result.get("novelty")})

    science_test = [
        "What is photosynthesis?",
        "Explain gravity simply.",
        "What is DNA?",
        "How does a battery work?",
        "What is entropy?",
    ]

    backward_results = []
    for i, problem in enumerate(science_test):
        result = model.generate_with_fused_memory(problem, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()
        print(f"[B{i+1}] stored={result.get('stored')} cap={stats['memory']['activation_ratio']:.3f} novelty={result.get('novelty', 0):.3f}")
        backward_results.append({"problem": problem, "stored": result.get("stored"), "capacity": stats["memory"]["activation_ratio"], "novelty": result.get("novelty")})

    final = model.get_garden_stats()
    print(f"\nFINAL: slots={final['memory']['active_slots']}/{final['memory']['total_slots']} cap={final['memory']['activation_ratio']:.3f}")

    os.makedirs("results", exist_ok=True)
    payload = {
        "final_stats": final,
        "phase1_math": math_results,
        "phase2_science": science_results,
        "phase3_forward": forward_results,
        "phase4_backward": backward_results,
    }
    with open("results/continual_benchmark_fused.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print("Results saved: results/continual_benchmark_fused.json")
    return payload


if __name__ == "__main__":
    continual_learning_benchmark_fused()
