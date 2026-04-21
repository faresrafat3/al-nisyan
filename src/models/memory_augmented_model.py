import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.gemma_loader import GemmaLoader
    from src.models.memory_bank import EpisodicMemoryBank
else:
    from .gemma_loader import GemmaLoader
    from .memory_bank import EpisodicMemoryBank


class MemoryAugmentedModel(nn.Module):
    """
    Wraps a base LLM with an external episodic memory bank.
    Before generating, the model reads from memory based on the query.
    After generating, it writes the interaction to memory.
    """

    def __init__(
        self,
        base_model_key: str = "qwen3_0.6b",
        memory_slots: int = 2048,
        memory_dim: int = 512,
        memory_context_length: int = 3,
    ):
        super().__init__()

        self.loader = GemmaLoader(model_key=base_model_key)
        self.model, self.tokenizer = self.loader.load(load_in_4bit=True)

        self.hidden_dim = self.model.config.hidden_size

        self.memory = EpisodicMemoryBank(
            num_slots=memory_slots,
            slot_dim=memory_dim,
        )

        self.memory_to_hidden = nn.Linear(memory_dim, self.hidden_dim)
        self.hidden_to_memory = nn.Linear(self.hidden_dim, memory_dim)

        self.memory_context_length = memory_context_length
        self.use_memory = True

        self._move_memory_modules_to_device()

    def _move_memory_modules_to_device(self) -> None:
        device = torch.device(self.loader.device)
        self.memory = self.memory.to(device)
        self.memory_to_hidden = self.memory_to_hidden.to(device)
        self.hidden_to_memory = self.hidden_to_memory.to(device)

    def encode_query(self, prompt: str) -> torch.Tensor:
        """
        Encode prompt to get hidden representation for memory query.
        Uses the base model's hidden states.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.loader.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1][:, -1, :]

        return hidden

    def retrieve_memory(self, query_hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieve relevant memories and project to model's hidden space.
        Returns: (memory_context_hidden, attention_weights, memory_query)
        """
        query_hidden = query_hidden.to(self.hidden_to_memory.weight.dtype)
        memory_query = self.hidden_to_memory(query_hidden)
        retrieved, attn_weights = self.memory.read(memory_query)
        memory_context = self.memory_to_hidden(retrieved)

        return memory_context, attn_weights, memory_query

    def generate_with_memory(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        store_in_memory: bool = True,
    ) -> dict:
        """
        Generate response using memory augmentation.
        Returns dict with response, memory stats, and attention weights.
        """
        device = self.loader.device

        query_hidden = self.encode_query(prompt)
        query_hidden = query_hidden.to(self.hidden_to_memory.weight.dtype)

        memory_context = None
        attn_weights = None
        memory_query = self.hidden_to_memory(query_hidden)

        if self.use_memory:
            memory_context, attn_weights, memory_query = self.retrieve_memory(query_hidden)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        write_gate = None
        if self.use_memory and store_in_memory:
            full_text = f"Q: {prompt}\nA: {response}"
            situation_hidden = self.encode_query(full_text)
            situation_hidden = situation_hidden.to(self.hidden_to_memory.weight.dtype)
            situation_memory = self.hidden_to_memory(situation_hidden)

            _, write_gate = self.memory(
                query=memory_query,
                content=situation_memory,
            )

        result = {
            "response": response,
            "memory_used": self.use_memory,
            "memory_stats": self.memory.get_memory_stats() if self.use_memory else None,
            "write_gate": write_gate,
        }

        if attn_weights is not None:
            result["memory_attention_max"] = attn_weights.max().item()
            result["memory_attention_entropy"] = -(
                attn_weights * torch.log(attn_weights + 1e-10)
            ).sum().item()

        if memory_context is not None:
            result["memory_context_norm"] = memory_context.norm(dim=-1).mean().item()

        return result

    def toggle_memory(self, enabled: bool = True):
        """Enable or disable memory augmentation."""
        self.use_memory = enabled
        print(f"Memory augmentation: {'ON' if enabled else 'OFF'}")

    def get_vram_usage(self) -> dict:
        """Get current VRAM usage."""
        if torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "total_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3,
            }
        return {"device": "cpu"}


if __name__ == "__main__":
    print("=" * 60)
    print("Memory-Augmented Model Test")
    print("Continual Learning Simulation")
    print("=" * 60)

    print("\n[1] Loading model with memory...")
    mam = MemoryAugmentedModel(
        base_model_key="qwen3_0.6b",
        memory_slots=512,
        memory_dim=256,
    )

    print(f"Base model: {mam.loader.get_model_info()}")
    print(f"Memory slots: {mam.memory.num_slots}")
    print(f"VRAM: {mam.get_vram_usage()}")

    print("\n" + "=" * 60)
    print("[2] Task 1: Basic Math")
    print("=" * 60)

    prompts_math = [
        "What is 15 + 27?",
        "Calculate 8 * 9.",
        "What is 100 divided by 4?",
    ]

    for prompt in prompts_math:
        print(f"\nQ: {prompt}")
        result = mam.generate_with_memory(prompt, max_new_tokens=64)
        print(f"A: {result['response'].strip()}")
        print(f"Memory write gate: {result['write_gate']:.4f}")
        print(f"Memory attention max: {result['memory_attention_max']:.4f}")

    print("\nMemory after Task 1:")
    stats = mam.memory.get_memory_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n" + "=" * 60)
    print("[3] Task 2: General Knowledge")
    print("=" * 60)

    prompts_facts = [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "What is the largest planet?",
    ]

    for prompt in prompts_facts:
        print(f"\nQ: {prompt}")
        result = mam.generate_with_memory(prompt, max_new_tokens=64)
        print(f"A: {result['response'].strip()}")
        print(f"Memory write gate: {result['write_gate']:.4f}")

    print("\n" + "=" * 60)
    print("[4] Task 3: Math Again (Testing Forgetting)")
    print("=" * 60)

    test_prompt = "What is 15 + 27?"
    print(f"\nQ: {test_prompt} (ASKED BEFORE)")

    result_with_memory = mam.generate_with_memory(test_prompt, max_new_tokens=64)
    print(f"A (with memory): {result_with_memory['response'].strip()}")
    print(f"Memory attention max: {result_with_memory['memory_attention_max']:.4f}")

    mam.toggle_memory(False)
    result_no_memory = mam.generate_with_memory(test_prompt, max_new_tokens=64)
    print(f"A (no memory): {result_no_memory['response'].strip()}")

    mam.toggle_memory(True)
    print("\n[Comparison]")
    print(f"With memory attention: {result_with_memory['memory_attention_max']:.4f}")
    print(
        "Difference in response: "
        f"{'SAME' if result_with_memory['response'].strip() == result_no_memory['response'].strip() else 'DIFFERENT'}"
    )

    print(f"\nFinal VRAM: {mam.get_vram_usage()}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
