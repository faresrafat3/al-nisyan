"""
MeKi Hybrid Memory Injection Module
----------------------------------
A lightweight hybrid memory injector that combines:
- Inference-time kNN memory retrieval
- Embedding-level fusion before generation

Designed for edge-friendly experiments on Kaggle T4.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MemoryEntry:
    key: torch.Tensor
    value: torch.Tensor
    text: str


class MeKiHybridInjector(nn.Module):
    """Hybrid memory injector with kNN retrieval + embedding fusion."""

    def __init__(
        self,
        model,
        tokenizer,
        max_entries: int = 4096,
        top_k: int = 8,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_entries = max_entries
        self.top_k = top_k

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.hidden_dim = self._resolve_hidden_dim(getattr(model, "config", None))

        self.query_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.memory_gate = nn.Parameter(torch.tensor(0.25))

        self.memory_bank: List[MemoryEntry] = []
        self.hit_counter = 0
        self.query_counter = 0

        self.to(torch.device(self.device))

    @staticmethod
    def _resolve_hidden_dim(config) -> int:
        candidates = ("hidden_size", "d_model", "dim", "model_dim", "hidden_dim")
        if config is not None:
            for key in candidates:
                value = getattr(config, key, None)
                if isinstance(value, int) and value > 0:
                    return value
            for sub_key in ("text_config", "language_config", "llm_config"):
                sub_cfg = getattr(config, sub_key, None)
                if sub_cfg is None:
                    continue
                for key in candidates:
                    value = getattr(sub_cfg, key, None)
                    if isinstance(value, int) and value > 0:
                        return value
        raise AttributeError("Could not resolve hidden dimension from model config.")

    def _encode_text_hidden(self, text: str) -> torch.Tensor:
        tokens = self.tokenizer(text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(
                **tokens,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[-1][:, -1, :]
        return hidden

    def add_memory(self, text: str, value_text: Optional[str] = None) -> None:
        key_vec = self._encode_text_hidden(text).detach().squeeze(0)
        value_source = value_text if value_text is not None else text
        value_vec = self._encode_text_hidden(value_source).detach().squeeze(0)

        self.memory_bank.append(
            MemoryEntry(key=key_vec.cpu(), value=value_vec.cpu(), text=text)
        )
        if len(self.memory_bank) > self.max_entries:
            self.memory_bank = self.memory_bank[-self.max_entries :]

    def _retrieve(self, query_hidden: torch.Tensor) -> tuple[Optional[torch.Tensor], float]:
        if not self.memory_bank:
            return None, 1.0

        self.query_counter += 1
        query = F.normalize(self.query_proj(query_hidden), dim=-1)

        keys = torch.stack([entry.key for entry in self.memory_bank], dim=0).to(self.device)
        values = torch.stack([entry.value for entry in self.memory_bank], dim=0).to(self.device)
        keys = F.normalize(keys, dim=-1)

        sims = torch.matmul(query, keys.t()).squeeze(0)
        top_k = min(self.top_k, sims.shape[0])
        top_vals, top_idx = torch.topk(sims, k=top_k)

        if float(top_vals.max().item()) > 0.5:
            self.hit_counter += 1

        weights = F.softmax(top_vals, dim=0)
        retrieved = torch.sum(weights.unsqueeze(-1) * values[top_idx], dim=0, keepdim=True)
        novelty = float((1.0 - top_vals.max()).item())

        return retrieved, novelty

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        do_store: bool = True,
    ) -> Dict:
        input_pack = self.tokenizer(text=prompt, return_tensors="pt").to(self.device)
        input_ids = input_pack.input_ids

        query_hidden = self._encode_text_hidden(prompt)
        memory_vec, novelty = self._retrieve(query_hidden)

        with torch.no_grad():
            token_embeds = self.model.get_input_embeddings()(input_ids)

            if memory_vec is not None:
                memory_hidden = self.value_proj(memory_vec).to(token_embeds.dtype)
                gate = torch.sigmoid(self.memory_gate)
                token_embeds[:, 0, :] = (1.0 - gate) * token_embeds[:, 0, :] + gate * memory_hidden

            outputs = self.model.generate(
                inputs_embeds=token_embeds,
                attention_mask=torch.ones(
                    (input_ids.shape[0], input_ids.shape[1]),
                    dtype=torch.long,
                    device=self.device,
                ),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )

        prompt_len = input_ids.shape[1]
        if outputs.shape[1] > prompt_len:
            response_ids = outputs[0][prompt_len:]
        else:
            response_ids = outputs[0]

        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)

        if do_store:
            self.add_memory(prompt, value_text=f"Q: {prompt}\nA: {response}")

        return {
            "response": response,
            "novelty": novelty,
            "entries": len(self.memory_bank),
            "memory_gate": float(torch.sigmoid(self.memory_gate).item()),
        }

    def base_generate(self, prompt: str, max_new_tokens: int = 64, temperature: float = 0.7) -> str:
        input_pack = self.tokenizer(text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **input_pack,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
            )
        response_ids = outputs[0][input_pack.input_ids.shape[1] :]
        return self.tokenizer.decode(response_ids, skip_special_tokens=True)

    def stats(self) -> Dict:
        vram = 0.0
        if torch.cuda.is_available():
            vram = torch.cuda.memory_allocated() / (1024**3)
        hit_rate = (self.hit_counter / self.query_counter) if self.query_counter else 0.0
        return {
            "entries": len(self.memory_bank),
            "max_entries": self.max_entries,
            "k": self.top_k,
            "hit_rate": hit_rate,
            "vram_gb": vram,
            "memory_gate": float(torch.sigmoid(self.memory_gate).item()),
        }


class MeKiBenchmark:
    """Quick benchmark wrapper for learning + recall phases."""

    def __init__(self, injector: MeKiHybridInjector):
        self.injector = injector
        self.phase1: List[Dict] = []
        self.phase2: List[Dict] = []
        self.base_latency: List[float] = []
        self.hybrid_latency: List[float] = []

    def run_phase1_learning(self, questions: Sequence[str], max_new_tokens: int = 64) -> None:
        for idx, question in enumerate(questions, start=1):
            start = time.perf_counter()
            result = self.injector.generate(question, max_new_tokens=max_new_tokens, do_store=True)
            elapsed = time.perf_counter() - start
            self.hybrid_latency.append(elapsed)

            self.phase1.append(
                {
                    "idx": idx,
                    "question": question,
                    "novelty": result.get("novelty"),
                    "entries": result.get("entries"),
                }
            )
            print(
                f"[L{idx}] novelty={result.get('novelty', 0):.3f} "
                f"entries={result.get('entries', 0)} latency={elapsed:.3f}s"
            )

    def run_phase2_recall(self, questions: Sequence[str], max_new_tokens: int = 64) -> None:
        for idx, question in enumerate(questions, start=1):
            base_start = time.perf_counter()
            _ = self.injector.base_generate(question, max_new_tokens=max_new_tokens)
            self.base_latency.append(time.perf_counter() - base_start)

            hybrid_start = time.perf_counter()
            result = self.injector.generate(question, max_new_tokens=max_new_tokens, do_store=False)
            hybrid_elapsed = time.perf_counter() - hybrid_start
            self.hybrid_latency.append(hybrid_elapsed)

            self.phase2.append(
                {
                    "idx": idx,
                    "question": question,
                    "novelty": result.get("novelty"),
                    "entries": result.get("entries"),
                }
            )
            print(
                f"[R{idx}] novelty={result.get('novelty', 0):.3f} "
                f"entries={result.get('entries', 0)} latency={hybrid_elapsed:.3f}s"
            )

    def report(self) -> Dict:
        base_avg = sum(self.base_latency) / len(self.base_latency) if self.base_latency else 0.0
        hybrid_avg = sum(self.hybrid_latency) / len(self.hybrid_latency) if self.hybrid_latency else 0.0
        overhead = ((hybrid_avg - base_avg) / base_avg) if base_avg > 0 else 0.0

        repeat_novelties = [x["novelty"] for x in self.phase2 if x.get("novelty") is not None]
        repeat_novelty_avg = sum(repeat_novelties) / len(repeat_novelties) if repeat_novelties else 0.0

        stats = self.injector.stats()
        payload = {
            "latency_base_s": base_avg,
            "latency_hybrid_s": hybrid_avg,
            "latency_overhead_ratio": overhead,
            "repeat_novelty_avg": repeat_novelty_avg,
            "kNN_entries": stats["entries"],
            "kNN_hit_rate": stats["hit_rate"],
            "vram_gb": stats["vram_gb"],
            "gate": stats["memory_gate"],
        }

        print("\n=== MeKi Benchmark Report ===")
        for key, value in payload.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")

        return payload
