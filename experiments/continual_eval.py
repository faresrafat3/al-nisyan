# Continual Learning Benchmark for Al-Nisyan
# Tests: Forward Transfer (Task A -> Task B -> Task A)
#        Backward Transfer (Task B -> Task A)

import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.memory_augmented_model_v2 import CultivatedMemoryModel


def continual_learning_benchmark():
    """
    Full continual learning evaluation.

    Phase 1: Learn Math
    Phase 2: Learn Science
    Phase 3: Forward Transfer (Test Math again)
    Phase 4: Backward Transfer (Test Science again)
    """

    print("=" * 70)
    print("AL-NISYAN: CONTINUAL LEARNING BENCHMARK")
    print("=" * 70)

    model = CultivatedMemoryModel(
        base_model_key="Qwen/Qwen3.5-4B",
        memory_slots=256,
        memory_dim=512,
        controller_mode="clean",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("CPU mode")

    # ============================================================
    # PHASE 1: LEARN MATH
    # ============================================================

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

    print(f"\n{'='*70}")
    print(f"PHASE 1: Learning {len(math_problems)} Math Problems")
    print(f"{'='*70}")

    math_results = []
    for i, problem in enumerate(math_problems):
        result = model.generate_with_cultivated_memory(problem, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()

        stored = result.get("stored", False)
        capacity = stats["memory"]["activation_ratio"]

        print(
            f"[{i+1:2d}/{len(math_problems)}] {problem[:45]:45s} | "
            f"stored={stored} | Capacity={capacity:.3f}"
        )

        math_results.append(
            {
                "problem": problem,
                "stored": stored,
                "capacity": capacity,
                "novelty": result.get("novelty"),
                "threshold": result.get("dynamic_threshold"),
            }
        )

    # ============================================================
    # PHASE 2: LEARN SCIENCE
    # ============================================================

    science_problems = [
        "What is photosynthesis?", "Explain gravity simply.",
        "What is DNA?", "How does a battery work?",
        "What is entropy?", "Explain quantum superposition simply.",
        "What are black holes?", "How do vaccines work?",
        "What is climate change?", "Explain nuclear fission.",
        "What is relativity?", "Explain evolution briefly.",
        "What is an atom?", "How do magnets work?",
    ]

    print(f"\n{'='*70}")
    print(f"PHASE 2: Learning {len(science_problems)} Science Problems")
    print(f"{'='*70}")

    science_results = []
    for i, problem in enumerate(science_problems):
        result = model.generate_with_cultivated_memory(problem, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()

        stored = result.get("stored", False)
        capacity = stats["memory"]["activation_ratio"]

        print(
            f"[{i+1:2d}/{len(science_problems)}] {problem[:45]:45s} | "
            f"stored={stored} | Capacity={capacity:.3f}"
        )

        science_results.append(
            {
                "problem": problem,
                "stored": stored,
                "capacity": capacity,
                "novelty": result.get("novelty"),
                "threshold": result.get("dynamic_threshold"),
            }
        )

    # ============================================================
    # PHASE 3: FORWARD TRANSFER (Math after Science)
    # ============================================================

    print(f"\n{'='*70}")
    print("PHASE 3: Forward Transfer Test (Math after Science)")
    print("=" * 70)

    math_test = [
        "What is 5 + 3?",
        "Calculate 6 times 7.",
        "What is 2 to the power of 10?",
        "What is 100 plus 200?",
        "Calculate 11 times 11.",
    ]

    forward_results = []
    for i, problem in enumerate(math_test):
        result = model.generate_with_cultivated_memory(problem, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()

        stored = result.get("stored", False)
        capacity = stats["memory"]["activation_ratio"]
        novelty = result.get("novelty", 0.0)

        print(
            f"[{i+1}/{len(math_test)}] {problem[:45]:45s} | "
            f"stored={stored} | Capacity={capacity:.3f} | Novelty={novelty:.3f}"
        )

        forward_results.append(
            {
                "problem": problem,
                "stored": stored,
                "capacity": capacity,
                "novelty": novelty,
                "threshold": result.get("dynamic_threshold"),
            }
        )

    # ============================================================
    # PHASE 4: BACKWARD TRANSFER (Science after Math)
    # ============================================================

    print(f"\n{'='*70}")
    print("PHASE 4: Backward Transfer Test (Science after Math)")
    print("=" * 70)

    science_test = [
        "What is photosynthesis?",
        "Explain gravity simply.",
        "What is DNA?",
        "How does a battery work?",
        "What is entropy?",
    ]

    backward_results = []
    for i, problem in enumerate(science_test):
        result = model.generate_with_cultivated_memory(problem, max_new_tokens=64, return_debug=True)
        stats = model.get_garden_stats()

        stored = result.get("stored", False)
        capacity = stats["memory"]["activation_ratio"]
        novelty = result.get("novelty", 0.0)

        print(
            f"[{i+1}/{len(science_test)}] {problem[:45]:45s} | "
            f"stored={stored} | Capacity={capacity:.3f} | Novelty={novelty:.3f}"
        )

        backward_results.append(
            {
                "problem": problem,
                "stored": stored,
                "capacity": capacity,
                "novelty": novelty,
                "threshold": result.get("dynamic_threshold"),
            }
        )

    # ============================================================
    # FINAL STATS + SAVE
    # ============================================================

    final = model.get_garden_stats()

    print(f"\n{'='*70}")
    print("FINAL STATISTICS")
    print("=" * 70)
    print(f"Memory slots: {final['memory']['active_slots']}/{final['memory']['total_slots']}")
    print(f"Activation ratio: {final['memory']['activation_ratio']:.3f}")
    print(f"Step counter: {final['step_counter']}")
    print(f"Decay rate: {final['stability_decay_rate']:.4f}")

    os.makedirs("results", exist_ok=True)

    output = {
        "final_stats": final,
        "phase1_math": math_results,
        "phase2_science": science_results,
        "phase3_forward": forward_results,
        "phase4_backward": backward_results,
    }

    with open("results/continual_benchmark.json", "w", encoding="utf-8") as file_handle:
        json.dump(output, file_handle, indent=2, default=str)

    print("\nResults saved: results/continual_benchmark.json")
    print("=" * 70)

    return output


if __name__ == "__main__":
    continual_learning_benchmark()
