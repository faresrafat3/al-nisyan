"""
Al-Nisyan v3: embedding-level memory fusion.

This version keeps the base LLM frozen and injects memory as a soft prefix
in the embedding space instead of prompt-level RAG.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.memory_augmented_model_v2 import CultivatedMemoryModel as BaseCultivatedMemoryModel
    from src.models.memory_fused_generation import MemoryFusedGenerator
else:
    from .memory_augmented_model_v2 import CultivatedMemoryModel as BaseCultivatedMemoryModel
    from .memory_fused_generation import MemoryFusedGenerator


class CultivatedMemoryModelV3(BaseCultivatedMemoryModel):
    """Memory-Augmented model with embedding-level fusion."""

    def __init__(
        self,
        base_model_key: str = "Qwen/Qwen3.5-4B",
        memory_slots: int = 512,
        memory_dim: int = 512,
        controller_mode: str = "clean",
    ):
        super().__init__(
            base_model_key=base_model_key,
            memory_slots=memory_slots,
            memory_dim=memory_dim,
            controller_mode=controller_mode,
        )
        self.memory_fuser = MemoryFusedGenerator(self.hidden_dim, memory_dim)
        self._move_memory_stack_to_device()

    def _move_memory_stack_to_device(self) -> None:
        super()._move_memory_stack_to_device()
        if hasattr(self, "memory_fuser"):
            self.memory_fuser = self.memory_fuser.to(torch.device(self.loader.device))

    def generate_with_fused_memory(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        return_debug: bool = False,
        store_in_memory: bool = True,
    ) -> Dict:
        """Generate with retrieved memory injected into embedding space."""
        device = self.loader.device

        query = self.encode_to_memory_space(prompt)

        memory_context = None
        attn = None
        if self.memory_active:
            memory_context, attn = self.retrieve(query)

        inputs = self.tokenizer(text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(inputs.input_ids)
            fused_embeds, attention_mask, prefix_len = self.memory_fuser.fuse(
                token_embeddings=token_embeds,
                memory_context=memory_context if self.memory_active else None,
            )

            outputs = self.model.generate(
                inputs_embeds=fused_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Handle both output conventions safely.
        if outputs.shape[1] > fused_embeds.shape[1]:
            response_ids = outputs[0][fused_embeds.shape[1]:]
        else:
            response_ids = outputs[0]

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        situation = f"Q: {prompt}\nA: {response}"
        situation_query = self.encode_to_memory_space(situation)

        cultivation_result = None
        if store_in_memory and self.memory_active:
            cultivation_result = self.cultivate(
                content=situation_query,
                query=query,
            )

        result = {
            "response": response,
            "memory_active": self.memory_active,
            "memory_used": bool(memory_context is not None),
            "fused_prefix_len": prefix_len,
            "step": self.step_counter - 1 if self.memory_active else self.step_counter,
        }

        if cultivation_result is not None:
            result.update(
                {
                    "stored": cultivation_result["stored"],
                    "store_score": cultivation_result["store_score"],
                    "novelty": cultivation_result["novelty"],
                    "dynamic_threshold": cultivation_result.get("dynamic_threshold"),
                    "relevance": cultivation_result["relevance"],
                    "conflicts": cultivation_result["conflicts"],
                    "decay_rate": cultivation_result["decay_rate"],
                    "erased_slot": cultivation_result["erased_slot"],
                    "capacity": cultivation_result.get("capacity"),
                    "emergency_erase": cultivation_result.get("emergency_erase"),
                    "erased_only": cultivation_result.get("erased_only"),
                }
            )

        if attn is not None:
            result["memory_attention_max"] = float(attn.max().item())
            result["memory_attention_entropy"] = float((-(attn * torch.log(attn + 1e-10)).sum()).item())

        if return_debug:
            result["memory_stats"] = self.memory.get_memory_stats()
            result["access_times"] = self.access_times.detach().cpu().numpy().tolist()

        return result

    # Keep the original API available, but route it through fusion.
    def generate_with_cultivated_memory(self, *args, **kwargs):
        return self.generate_with_fused_memory(*args, **kwargs)
