import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.memory_augmented_model_v2 import CultivatedMemoryModel


def run_quick_stress():
    model = CultivatedMemoryModel(
        base_model_key="Qwen/Qwen3.5-4B",
        memory_slots=64,
        memory_dim=512,
        controller_mode="clean",
    )

    prompts = [
        "What is 5 + 3?", "Calculate 10 * 2.", "What is 15 - 7?", "What is 8 / 2?",
        "Calculate 3 + 3.", "What is 100 + 200?", "Calculate 7 * 8.", "What is 50 - 25?",
        "What is 17 + 29?", "Calculate 6 * 7.",
        "What is photosynthesis?", "Explain gravity.", "What is DNA?", "How does battery work?",
        "What is entropy?", "Explain quantum superposition.", "What are black holes?",
        "How do vaccines work?", "What is climate change?", "Explain nuclear fission.",
        "Write Python reverse string.", "Sort list in JavaScript.", "Explain recursion.",
        "What is SQL JOIN?", "Regex for email.", "Explain Docker.", "What is REST API?",
        "How blockchain works?", "Explain ML simply.", "What is Big O?",
        "Capital of Japan?", "Who painted Mona Lisa?", "Tallest mountain?", "When WW2 end?",
        "Speed of light?", "Who invented telephone?", "Largest ocean?", "Language in Brazil?",
        "Currency of UK?", "Who wrote 1984?",
        "What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?", "What is 5 + 5?",
        "What is 6 + 6?", "What is 7 + 7?", "What is 8 + 8?", "What is 9 + 9?",
        "What is 10 + 10?", "What is 1 + 1?",
    ]

    for index, prompt in enumerate(prompts):
        result = model.generate_with_cultivated_memory(
            prompt,
            max_new_tokens=64,
            return_debug=True,
        )
        stats = model.get_garden_stats()

        threshold = result.get("dynamic_threshold")
        threshold_text = f"{threshold:.2f}" if isinstance(threshold, (int, float)) else "N/A"

        print(
            f"[{index + 1}/50] "
            f"stored={result.get('stored')} "
            f"cap={stats['memory']['activation_ratio']:.1%} "
            f"thresh={threshold_text}"
        )

    print(f"\nFINAL: {model.get_garden_stats()}")


if __name__ == "__main__":
    run_quick_stress()
