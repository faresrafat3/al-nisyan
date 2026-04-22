import json
import os
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.memory_augmented_model_v2 import CultivatedMemoryModel


def run_forgetting_experiment(
    model_key: str = "qwen3_0.6b",
    memory_slots: int = 256,
    save_results: bool = True,
):
    """
    Official Al-Nisyan experiment.

    Tests whether a small model with cultivated memory can:
    1) learn selectively,
    2) forget selectively,
    3) retrieve relevant traces,
    4) resolve conflicts via drift.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 75)
    print(" AL-NISYAN: The Forgetting Experiment")
    print(" Can a 0.6B model with cultivated memory learn like a human?")
    print(f" Timestamp: {timestamp}")
    print("=" * 75)

    print("\n[INIT] Cultivating the garden...")
    model = CultivatedMemoryModel(
        base_model_key=model_key,
        memory_slots=memory_slots,
        memory_dim=256,
        controller_mode="clean",
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f" Device: {device}")
    print(
        f" VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB"
        if device == "cuda"
        else " CPU mode"
    )
    print(f" Memory slots: {memory_slots}")
    print(f" Controller params: {sum(p.numel() for p in model.controller.parameters()) / 1e6:.2f}M")

    results = {
        "timestamp": timestamp,
        "model_key": model_key,
        "memory_slots": memory_slots,
        "phases": [],
    }

    print("\n" + "=" * 75)
    print("[PHASE 1] Math Flooding: 8 problems, mixed novelty")
    print(" Expectation: Store novel, discard repetitive")
    print("=" * 75)

    math_problems = [
        ("What is 5 + 3?", "simple"),
        ("Calculate 10 * 2.", "simple"),
        ("What is 15 - 7?", "simple"),
        ("What is 8 / 2?", "simple"),
        ("Calculate 3 + 3.", "repetitive"),
        ("What is 100 + 200?", "novel"),
        ("Calculate 7 * 8.", "novel"),
        ("What is 50 - 25?", "novel"),
    ]

    phase1_results = []
    for idx, (problem, expected) in enumerate(math_problems):
        print(f"\n[{idx + 1}/8] [{expected}] {problem}")

        result = model.generate_with_cultivated_memory(
            problem,
            max_new_tokens=64,
            return_debug=True,
        )

        print(f"  → {result['response'].strip()[:50]}...")
        print(
            f"  Stored: {result.get('stored')} | "
            f"Score: {result.get('store_score', 0):.3f} | "
            f"Novelty: {result.get('novelty', 0):.3f} | "
            f"Conflicts: {result.get('conflicts', 0)}"
        )
        if result.get("dynamic_threshold") is not None:
            print(f"  Dynamic threshold: {result.get('dynamic_threshold', 0):.3f}")

        phase1_results.append(
            {
                "problem": problem,
                "expected_type": expected,
                "stored": result.get("stored"),
                "store_score": result.get("store_score"),
                "novelty": result.get("novelty"),
                "dynamic_threshold": result.get("dynamic_threshold"),
                "conflicts": result.get("conflicts"),
            }
        )

    results["phases"].append(
        {
            "name": "math_flooding",
            "description": "8 math problems to test selective storage",
            "interactions": phase1_results,
        }
    )

    print("\n" + "=" * 75)
    print("[PHASE 2] Domain Switch: General Knowledge")
    print(" Expectation: High novelty (new domain), store most")
    print("=" * 75)

    facts = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What is the speed of light in vacuum?",
    ]

    phase2_results = []
    for idx, fact in enumerate(facts):
        print(f"\n[{idx + 1}/3] {fact}")

        result = model.generate_with_cultivated_memory(
            fact,
            max_new_tokens=64,
            return_debug=True,
        )

        print(f"  → {result['response'].strip()[:50]}...")
        print(
            f"  Stored: {result.get('stored')} | "
            f"Novelty: {result.get('novelty', 0):.3f} | "
            f"Conflicts: {result.get('conflicts', 0)}"
        )
        if result.get("dynamic_threshold") is not None:
            print(f"  Dynamic threshold: {result.get('dynamic_threshold', 0):.3f}")

        phase2_results.append(
            {
                "fact": fact,
                "stored": result.get("stored"),
                "novelty": result.get("novelty"),
                "dynamic_threshold": result.get("dynamic_threshold"),
                "conflicts": result.get("conflicts"),
            }
        )

    results["phases"].append(
        {
            "name": "domain_switch",
            "description": "3 general knowledge facts to test cross-domain novelty",
            "interactions": phase2_results,
        }
    )

    print("\n" + "=" * 75)
    print("[PHASE 3] The Forgetting Test: Return to Old Math")
    print(" Expectation: Retrieve from memory, low novelty (already seen)")
    print("=" * 75)

    test_problem = "What is 5 + 3?"

    print(f"\n[TEST] {test_problem}")
    print("  This was asked in Phase 1. Should retrieve, not conflict.")

    result_with = model.generate_with_cultivated_memory(
        test_problem,
        max_new_tokens=64,
        return_debug=True,
    )

    print("\n  [With Memory ON]")
    print(f"  → {result_with['response'].strip()[:50]}...")
    print(
        f"  Store Score: {result_with.get('store_score', 0):.4f} | "
        f"Novelty: {result_with.get('novelty', 0):.4f} | "
        f"Conflicts: {result_with.get('conflicts', 0)}"
    )

    model.toggle_memory(False)
    result_without = model.generate_with_cultivated_memory(
        test_problem,
        max_new_tokens=64,
    )

    print("\n  [With Memory OFF]")
    print(f"  → {result_without['response'].strip()[:50]}...")

    model.toggle_memory(True)

    same_response = result_with["response"].strip() == result_without["response"].strip()
    print("\n  [Comparison]")
    print(f"  Responses identical: {same_response}")
    print(f"  With memory - novelty was lower: {result_with.get('novelty', 1) < 0.5}")

    phase3_results = {
        "test_problem": test_problem,
        "with_memory": {
            "response": result_with["response"],
            "store_score": result_with.get("store_score"),
            "novelty": result_with.get("novelty"),
        },
        "without_memory": {
            "response": result_without["response"],
        },
        "responses_identical": same_response,
    }

    results["phases"].append(
        {
            "name": "forgetting_test",
            "description": "Return to exact previous problem to test retrieval vs re-learning",
            "test": phase3_results,
        }
    )

    print("\n" + "=" * 75)
    print("[PHASE 4] Conflict Test: Contradictory Information")
    print(" Expectation: Drift - distort both memories to preserve structure")
    print("=" * 75)

    fact1 = "The capital of Germany is Berlin."
    print(f"\n[4.1] Establish: {fact1}")
    response_1 = model.generate_with_cultivated_memory(fact1, max_new_tokens=64)
    print(f"  → {response_1['response'].strip()[:50]}...")

    fact2 = "Some people say the capital of Germany is Munich."
    print(f"\n[4.2] Contradictory: {fact2}")
    response_2 = model.generate_with_cultivated_memory(fact2, max_new_tokens=64)
    print(f"  → {response_2['response'].strip()[:50]}...")
    print(f"  Conflicts detected: {response_2.get('conflicts', 0)}")
    print(f"  Store score: {response_2.get('store_score', 0):.4f}")

    phase4_results = {
        "fact1": fact1,
        "fact2": fact2,
        "conflicts_phase2": response_2.get("conflicts", 0),
        "store_score_phase2": response_2.get("store_score"),
    }

    results["phases"].append(
        {
            "name": "conflict_test",
            "description": "Contradictory facts to test drift mechanism",
            "interactions": phase4_results,
        }
    )

    print("\n" + "=" * 75)
    print("[FINAL] Garden Statistics: The State of Memory")
    print("=" * 75)

    stats = model.get_garden_stats()

    print(
        f"\n  Memory Activation: {stats['memory']['active_slots']}/"
        f"{stats['memory']['total_slots']} slots"
    )
    print(f"  Activation Ratio: {stats['memory']['activation_ratio']:.2%}")
    print(f"  Avg Slot Similarity: {stats['memory']['avg_slot_similarity']:.4f}")
    print(f"  Mean Slot Norm: {stats['memory']['memory_mean_norm']:.4f}")
    print(f"\n  Step Counter: {stats['step_counter']}")
    print(f"  Decay Rate: {stats['stability_decay_rate']:.6f}")
    print(f"  Memory Change EMA: {stats['memory_change_ema']:.6f}")
    print(
        f"  Recent Access (mean±std): "
        f"{stats['recent_access_mean']:.1f} ± {stats['recent_access_std']:.1f}"
    )

    results["final_stats"] = stats

    if save_results:
        os.makedirs("results", exist_ok=True)
        filename = f"results/nisyan_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as file_handle:
            json.dump(results, file_handle, indent=2, default=str)
        print(f"\n  Results saved: {filename}")

    print("\n" + "=" * 75)
    print(" Experiment Complete")
    print("=" * 75)

    return results


if __name__ == "__main__":
    run_forgetting_experiment(
        model_key="qwen3_0.6b",
        memory_slots=64,
        save_results=True,
    )
