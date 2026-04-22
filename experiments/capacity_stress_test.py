"""
Stress test: intentionally fill memory to test selective filtering behavior
at high capacity (30-50%) to verify threshold scaling and rejection of repetition.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.memory_augmented_model_v2 import CultivatedMemoryModel
import json

def stress_test():
    """Run 100 interactions to push capacity to 30%+"""
    model = CultivatedMemoryModel(
        base_model_key="Qwen/Qwen3.5-4B",
        memory_slots=256,  # Smaller = faster to fill
        memory_dim=512,
        controller_mode="clean",
    )
    
    # 60 diverse interactions + 40 repetitive interactions
    base_prompts = [
        # Math
        "What is 15 + 27?", "Calculate 8 * 9.", "What is 100 / 4?",
        "Solve x^2 - 5x + 6 = 0", "What is the derivative of x^3?",
        "Calculate 7!", "What is log(100)?", "Solve 2x + 3 = 11",
        "What is the area of circle radius 5?", "Calculate 3^5 + 2^4",
        # Science
        "What is photosynthesis?", "Explain Newton's third law.",
        "What is DNA?", "How does a battery work?",
        "What is entropy?", "Explain quantum superposition simply.",
        "What are black holes?", "How do vaccines work?",
        "What is climate change?", "Explain nuclear fission.",
        # Code
        "Write a Python function to reverse a string.",
        "How do you sort a list in JavaScript?",
        "Explain recursion with an example.",
        "What is a SQL JOIN?", "Write a regex for email validation.",
        "Explain Docker containers.", "What is REST API?",
        "How does blockchain work?", "Explain machine learning simply.",
        "What is Big O notation?",
        # Facts
        "Capital of Japan?", "Who painted the Mona Lisa?",
        "What is the tallest mountain?", "When did WW2 end?",
        "What is the speed of light?", "Who invented the telephone?",
        "What is the largest ocean?", "What language is spoken in Brazil?",
        "What is the currency of UK?", "Who wrote 1984?",
    ]

    diverse_prompts = []
    while len(diverse_prompts) < 60:
        diverse_prompts.extend(base_prompts)
    diverse_prompts = diverse_prompts[:60]

    repetitive_prompts = [
        "What is 2 + 2?", "What is 3 + 3?", "What is 4 + 4?",
        "What is 5 + 5?", "What is 6 + 6?", "What is 7 + 7?",
        "What is 8 + 8?", "What is 9 + 9?", "What is 10 + 10?",
        "What is 1 + 1?",
    ] * 4

    stress_prompts = diverse_prompts + repetitive_prompts
    
    results = []
    total = len(stress_prompts)
    repetitive_start = len(diverse_prompts)

    for i, prompt in enumerate(stress_prompts):
        print(f"\n[{i+1}/{total}] {prompt[:50]}...")
        
        result = model.generate_with_cultivated_memory(
            prompt, 
            max_new_tokens=64,
            return_debug=True,
        )
        
        stats = model.get_garden_stats()
        capacity_pct = stats['memory']['activation_ratio'] * 100
        
        print(f"  Stored: {result.get('stored')} | "
              f"Capacity: {capacity_pct:.1f}% | "
              f"Threshold: {result.get('dynamic_threshold', 'N/A'):.3f}")
        
        results.append({
            "prompt": prompt,
            "stored": result.get('stored'),
            "capacity_pct": capacity_pct,
            "threshold": result.get('dynamic_threshold'),
            "novelty": result.get('novelty'),
            "store_score": result.get('store_score'),
        })
    
    # Final stats
    final = model.get_garden_stats()
    print(f"\n{'='*60}")
    print(f"FINAL CAPACITY: {final['memory']['activation_ratio']*100:.1f}%")
    print(f"Active Slots: {final['memory']['active_slots']}/{final['memory']['total_slots']}")
    print(f"Decay Rate: {final['stability_decay_rate']:.4f}")
    print(f"{'='*60}")
    
    # Analysis: count rejections in repetitive phase
    repetitive_results = results[repetitive_start:]
    rejected_count = sum(1 for r in repetitive_results if not r['stored'])
    
    print(f"\nRepetitive Phase (indices {repetitive_start}-{total - 1}):")
    print(f"  Total: {len(repetitive_results)}")
    print(f"  Stored: {len(repetitive_results) - rejected_count}")
    print(f"  Rejected: {rejected_count}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/stress_test.json", "w") as f:
        json.dump({
            "final_stats": final,
            "interactions": results,
            "repetitive_rejection_rate": rejected_count / len(repetitive_results),
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to results/stress_test.json")
    return final

if __name__ == "__main__":
    stress_test()
