import torch
import torch.nn as nn
import torch.nn.functional as F


class CleanForgettingController(nn.Module):
    """
    v4: dynamic threshold only, no percentile, no top-k.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_slots: int = 2048,
        capacity_limit: float = 0.90,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_slots = num_slots
        self.capacity_limit = capacity_limit

        self.store_relevance = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.erase_gate = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.drift_strength = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.decay_rate = nn.Parameter(torch.tensor(0.03))
        self.register_buffer("memory_change_ema", torch.zeros(1))

    def check_capacity(self, memory_bank: torch.Tensor) -> float:
        mem_norms = torch.norm(memory_bank, dim=-1)
        meaningful = (mem_norms > 0.5).sum().item()
        return float(meaningful / self.num_slots)

    def compute_threshold(self, capacity: float) -> float:
        if capacity < 0.30:
            return 0.03
        elif capacity < 0.60:
            return 0.03 + (capacity - 0.30) * 0.9
        else:
            return 0.30 + (capacity - 0.60) * 1.333

    def forward(self, new_content, query, memory_bank, access_times, current_step):
        capacity = self.check_capacity(memory_bank)
        threshold = self.compute_threshold(capacity)

        combined = torch.cat([new_content, query], dim=-1)
        relevance = self.store_relevance(combined).squeeze(-1)

        new_norm = F.normalize(new_content, dim=-1)
        mem_norm = F.normalize(memory_bank, dim=-1)
        similarities = torch.matmul(new_norm, mem_norm.t())
        max_sim = similarities.max(dim=-1).values
        novelty = 1.0 - max_sim

        should_store = (relevance > 0.35) & (novelty > threshold)

        victim = None
        if capacity > self.capacity_limit:
            age = current_step - access_times
            confidence = torch.norm(memory_bank, dim=-1)
            erase_scores = self.erase_gate(
                torch.stack([confidence, age.float()], dim=-1)
            ).squeeze(-1)
            victim = int(erase_scores.argmax().item())
            should_store = torch.ones_like(should_store, dtype=torch.bool)

        written = False
        write_idx = None
        conflict = (max_sim > 0.7) & (max_sim < 0.99)
        erase_scores = None

        if should_store.any():
            age = (current_step - access_times).float()
            confidence = torch.norm(memory_bank, dim=-1)
            erase_scores = self.erase_gate(
                torch.stack([confidence, age], dim=-1)
            ).squeeze(-1)

            cooldown = (age < 3).float() * -1e9
            erase_scores = erase_scores + cooldown

            write_idx = victim if victim is not None else int(erase_scores.argmax().item())

            content_to_write = new_content[0].clone()
            if conflict.any():
                nearest_idx = int(similarities.argmax(dim=-1).item())
                pair = torch.cat([new_content[0], memory_bank[nearest_idx]], dim=-1).unsqueeze(0)
                strength = self.drift_strength(pair).squeeze()
                avg = (new_content[0] + memory_bank[nearest_idx]) / 2.0
                content_to_write = (1.0 - strength) * new_content[0] + strength * avg

            with torch.no_grad():
                memory_bank[write_idx] = content_to_write
                access_times[write_idx] = float(current_step)

            written = True

        with torch.no_grad():
            memory_bank *= (1.0 - self.decay_rate)

        return {
            "stored": written,
            "store_score": float(relevance.mean().item()),
            "novelty": float(novelty.mean().item()),
            "dynamic_threshold": float(threshold),
            "threshold": float(threshold),
            "capacity": float(capacity),
            "victim": victim,
            "emergency_erase": victim,
            "decay_rate": float(self.decay_rate.item()),
            "conflict_detected": float(conflict.sum().item()),
            "updated_memory": memory_bank,
            "updated_access_times": access_times,
            "write_index": write_idx,
            "erase_scores": erase_scores if erase_scores is not None else torch.zeros(memory_bank.shape[0], device=memory_bank.device),
        }