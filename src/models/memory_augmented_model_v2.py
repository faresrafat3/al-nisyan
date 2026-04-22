import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from src.models.gemma_loader import GemmaLoader
    from src.models.memory_bank import EpisodicMemoryBank
    from src.controllers.forgetting_gate import ForgettingController
    from src.controllers.forgetting_gate_v2 import AggressiveForgettingController
    from src.controllers.forgetting_gate_v3 import AdaptiveForgettingController
    from src.controllers.forgetting_gate_v4 import CleanForgettingController
else:
    from .gemma_loader import GemmaLoader
    from .memory_bank import EpisodicMemoryBank
    from ..controllers.forgetting_gate import ForgettingController
    from ..controllers.forgetting_gate_v2 import AggressiveForgettingController
    from ..controllers.forgetting_gate_v3 import AdaptiveForgettingController
    from ..controllers.forgetting_gate_v4 import CleanForgettingController


class CultivatedMemoryModel(nn.Module):
    """
    Memory-Augmented Model v2 with Forgetting Controller integration.

    Memory behaves as cultivated structure:
    - store what matters
    - erase what fades
    - drift conflicting traces
    """

    def __init__(
        self,
        base_model_key: str = "qwen3_0.6b",
        memory_slots: int = 512,
        memory_dim: int = 256,
        controller_mode: str = "adaptive",
    ):
        super().__init__()

        self.loader = GemmaLoader(model_key=base_model_key)
        self.model, self.tokenizer = self.loader.load(load_in_4bit=True)

        self.hidden_dim = self._resolve_hidden_dim(self.model.config)

        self.memory = EpisodicMemoryBank(
            num_slots=memory_slots,
            slot_dim=memory_dim,
            active_threshold=0.5 if controller_mode in {"adaptive", "clean", "v4", "aggressive"} else 0.1,
        )

        if controller_mode == "adaptive":
            self.controller = AdaptiveForgettingController(
                input_dim=memory_dim,
                hidden_dim=128,
                num_slots=memory_slots,
                capacity_limit=0.85,
            )
        elif controller_mode in {"clean", "v4"}:
            self.controller = CleanForgettingController(
                input_dim=memory_dim,
                hidden_dim=128,
                num_slots=memory_slots,
                capacity_limit=0.90,
            )
        elif controller_mode == "aggressive":
            self.controller = AggressiveForgettingController(
                input_dim=memory_dim,
                hidden_dim=128,
                num_slots=memory_slots,
                capacity_limit=0.85,
                min_novelty=0.4,
            )
        else:
            self.controller = ForgettingController(
                input_dim=memory_dim,
                hidden_dim=128,
                num_slots=memory_slots,
            )
        self.controller_mode = controller_mode

        self.hidden_to_memory = nn.Linear(self.hidden_dim, memory_dim)
        self.memory_to_hidden = nn.Linear(memory_dim, self.hidden_dim)

        self.register_buffer("access_times", torch.full((memory_slots,), -1.0))
        self.step_counter = 0
        self.memory_active = True

        self._move_memory_stack_to_device()

    @staticmethod
    def _resolve_hidden_dim(config) -> int:
        def _try_read(cfg):
            for key in ("hidden_size", "d_model", "dim", "model_dim", "hidden_dim"):
                value = getattr(cfg, key, None)
                if isinstance(value, int) and value > 0:
                    return value
            return None

        direct = _try_read(config)
        if direct is not None:
            return direct

        for sub_key in ("text_config", "language_config", "llm_config"):
            sub_cfg = getattr(config, sub_key, None)
            if sub_cfg is not None:
                sub_val = _try_read(sub_cfg)
                if sub_val is not None:
                    return sub_val

        text_cfg = getattr(config, "get_text_config", None)
        if callable(text_cfg):
            try:
                extracted = text_cfg()
                sub_val = _try_read(extracted)
                if sub_val is not None:
                    return sub_val
            except Exception:
                pass

        raise AttributeError(
            "Could not resolve hidden dimension from model config. "
            f"Top-level keys: {sorted(k for k in vars(config).keys() if not k.startswith('_'))}"
        )

    def _move_memory_stack_to_device(self) -> None:
        device = torch.device(self.loader.device)
        self.memory = self.memory.to(device)
        self.controller = self.controller.to(device)
        self.hidden_to_memory = self.hidden_to_memory.to(device)
        self.memory_to_hidden = self.memory_to_hidden.to(device)
        self.access_times = self.access_times.to(device)

    def encode_to_memory_space(self, text: str) -> torch.Tensor:
        """Encode text through base model then project to memory space."""
        inputs = self.tokenizer(text=text, return_tensors="pt").to(self.loader.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1][:, -1, :]

        hidden = hidden.to(self.hidden_to_memory.weight.dtype)
        memory_query = self.hidden_to_memory(hidden)
        return memory_query

    def retrieve(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Read from memory and update access-time map."""
        retrieved, attn = self.memory.read(query)

        top_slots = attn.argmax(dim=-1)
        self.access_times[top_slots] = float(self.step_counter)

        return retrieved, attn

    def cultivate(self, content: torch.Tensor, query: torch.Tensor) -> Dict:
        """
        Controller-mediated memory lifecycle:
        store / erase / drift decisions.
        """
        memory_state = self.memory.memory.detach().clone()
        access_times = self.access_times.clone()

        decision = self.controller(
            new_content=content,
            query=query,
            memory_bank=memory_state,
            access_times=access_times,
            current_step=self.step_counter,
        )

        with torch.no_grad():
            self.memory.memory.data.copy_(decision["updated_memory"])
            self.access_times.copy_(decision["updated_access_times"])

        self.step_counter += 1

        stored = bool(decision.get("stored", decision["store_score"] > 0.5))
        erased_slot = int(decision["erase_scores"].argmax().item()) if stored else None

        return {
            "stored": stored,
            "store_score": decision["store_score"],
            "novelty": decision["novelty"],
            "dynamic_threshold": decision.get("dynamic_threshold"),
            "relevance": decision.get("relevance", 0.0),
            "conflicts": decision["conflict_detected"],
            "decay_rate": decision["decay_rate"],
            "erased_slot": erased_slot,
            "capacity": decision.get("capacity"),
            "emergency_erase": decision.get("emergency_erase"),
            "erased_only": decision.get("erased_only"),
        }

    def generate_with_cultivated_memory(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        return_debug: bool = False,
    ) -> Dict:
        """Full loop: retrieve → generate → cultivate."""
        device = self.loader.device

        query = self.encode_to_memory_space(prompt)

        memory_context = None
        attn = None
        if self.memory_active:
            memory_context, attn = self.retrieve(query)

        inputs = self.tokenizer(text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response_ids = outputs[0][inputs.input_ids.shape[1] :]
        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        situation = f"Q: {prompt}\nA: {response}"
        situation_query = self.encode_to_memory_space(situation)

        cultivation_result = None
        if self.memory_active:
            cultivation_result = self.cultivate(
                content=situation_query,
                query=query,
            )

        result = {
            "response": response,
            "memory_active": self.memory_active,
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

        if memory_context is not None:
            result["memory_context_norm"] = float(memory_context.norm(dim=-1).mean().item())

        if return_debug:
            result["memory_stats"] = self.memory.get_memory_stats()
            result["access_times"] = self.access_times.detach().cpu().numpy().tolist()

        return result

    def toggle_memory(self, active: bool = True):
        self.memory_active = active
        print(f"Cultivated Memory: {'ON' if active else 'OFF'}")

    def get_garden_stats(self) -> Dict:
        """Get full state of the memory garden."""
        mem_stats = self.memory.get_memory_stats()

        recent_access = (self.step_counter - self.access_times).detach().cpu().numpy()

        controller_capacity = None
        if hasattr(self.controller, "check_capacity"):
            try:
                controller_capacity = float(self.controller.check_capacity(self.memory.memory, self.access_times))
            except TypeError:
                controller_capacity = float(self.controller.check_capacity(self.memory.memory))

        return {
            "memory": mem_stats,
            "controller_mode": self.controller_mode,
            "controller_capacity": controller_capacity,
            "step_counter": self.step_counter,
            "recent_access_mean": float(recent_access.mean()),
            "recent_access_std": float(recent_access.std()),
            "stability_decay_rate": float(getattr(self.controller, "decay_rate", torch.tensor(0.0)).item())
            if hasattr(getattr(self.controller, "decay_rate", None), "item")
            else float(getattr(self.controller, "decay_rate", 0.0)),
            "memory_change_ema": float(self.controller.memory_change_ema.item()),
        }


if __name__ == "__main__":
    print("=" * 70)
    print("AL-NISYAN: The Forgetting Experiment")
    print("Testing if memory forgets intelligently, not blindly")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[INIT] Loading Cultivated Memory Model...")
    model = CultivatedMemoryModel(
        base_model_key="qwen3_0.6b",
        memory_slots=256,
        memory_dim=256,
        controller_mode="aggressive",
    )
    print(f"Device: {device}")
    print(f"Memory slots: {model.memory.num_slots}")

    print("\n" + "=" * 70)
    print("[PHASE 1] Flooding with Math Problems")
    print("=" * 70)

    math_problems = [
        "What is 5 + 3?",
        "Calculate 10 * 2.",
        "What is 15 - 7?",
        "What is 8 / 2?",
        "Calculate 3 + 3.",
        "What is 100 + 200?",
        "Calculate 7 * 8.",
        "What is 50 - 25?",
    ]

    for idx, problem in enumerate(math_problems):
        print(f"\n[{idx + 1}] {problem}")
        result = model.generate_with_cultivated_memory(problem, max_new_tokens=64)
        print(f"  Response: {result['response'].strip()[:60]}...")
        print(
            f"  Stored: {result.get('stored', 'N/A')} | "
            f"Novelty: {result.get('novelty', 0):.3f} | "
            f"Conflicts: {result.get('conflicts', 0)}"
        )

    print("\n" + "=" * 70)
    print("[PHASE 2] Switching to General Knowledge")
    print("=" * 70)

    facts = [
        "What is the capital of France?",
        "Who wrote Hamlet?",
        "What is the speed of light?",
    ]

    for idx, fact in enumerate(facts):
        print(f"\n[{idx + 1}] {fact}")
        result = model.generate_with_cultivated_memory(fact, max_new_tokens=64)
        print(f"  Response: {result['response'].strip()[:60]}...")
        print(
            f"  Stored: {result.get('stored', 'N/A')} | "
            f"Novelty: {result.get('novelty', 0):.3f} | "
            f"Conflicts: {result.get('conflicts', 0)}"
        )

    print("\n" + "=" * 70)
    print("[PHASE 3] Return to Math (Memory Retrieval Test)")
    print("=" * 70)

    test_problem = "What is 5 + 3?"
    print(f"\n[TEST] {test_problem} (ASKED BEFORE)")

    result_with = model.generate_with_cultivated_memory(test_problem, max_new_tokens=64)
    print(f"  With Memory: {result_with['response'].strip()[:60]}...")
    print(
        f"  Store Score: {result_with.get('store_score', 0):.4f} | "
        f"Novelty: {result_with.get('novelty', 0):.4f}"
    )

    model.toggle_memory(False)
    result_without = model.generate_with_cultivated_memory(test_problem, max_new_tokens=64)
    print(f"  Without Memory: {result_without['response'].strip()[:60]}...")
    model.toggle_memory(True)

    print("\n" + "=" * 70)
    print("[PHASE 4] Garden Statistics")
    print("=" * 70)

    stats = model.get_garden_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, float):
                    print(f"  {nested_key}: {nested_value:.4f}")
                else:
                    print(f"  {nested_key}: {nested_value}")
        else:
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

    print("\n" + "=" * 70)
    print("Experiment Complete")
    print("=" * 70)
