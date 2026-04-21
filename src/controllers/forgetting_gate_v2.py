import torch
import torch.nn as nn
import torch.nn.functional as F


class AggressiveForgettingController(nn.Module):
    """
    v2: aggressive forgetting to prevent memory saturation.

    Key policy:
    1) higher base decay,
    2) hard capacity limit with emergency erase,
    3) novelty threshold before storing.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_slots: int = 2048,
        capacity_limit: float = 0.85,
        min_novelty: float = 0.4,
        store_threshold: float = 0.35,
        activation_threshold: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_slots = num_slots
        self.capacity_limit = capacity_limit
        self.min_novelty = min_novelty
        self.store_threshold = store_threshold
        self.activation_threshold = activation_threshold

        self.store_relevance = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )

        self.store_novelty = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.Sigmoid(),
        )

        self.store_gate = nn.Sequential(
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

        self.erase_lru = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
        )

        self.erase_confidence = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Sigmoid(),
        )

        self.erase_gate = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.drift_detector = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.drift_strength = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.decay_rate = nn.Parameter(torch.tensor(0.04))
        self.register_buffer("memory_change_ema", torch.zeros(1))
        self.stability_target = 0.1

    def check_capacity(self, memory_bank: torch.Tensor) -> float:
        mem_norms = torch.norm(memory_bank, dim=-1)
        active = (mem_norms > self.activation_threshold).sum().item()
        return float(active / self.num_slots)

    def emergency_erase(
        self,
        memory_bank: torch.Tensor,
        access_times: torch.Tensor,
        current_step: int,
    ) -> int:
        age = (float(current_step) - access_times).clamp(min=0)
        age_score = age / (age.max() + 1e-6)

        confidence = torch.norm(memory_bank, dim=-1)
        confidence_score = 1.0 - torch.sigmoid(confidence)

        erase_score = age_score + confidence_score
        victim_idx = int(erase_score.argmax().item())

        return victim_idx

    def _detect_conflict(self, new_content: torch.Tensor, memory_bank: torch.Tensor):
        norm_new = F.normalize(new_content, dim=-1)
        norm_mem = F.normalize(memory_bank, dim=-1)
        similarities = torch.matmul(norm_new, norm_mem.t())
        topk_vals, topk_idx = similarities.topk(3, dim=-1)
        conflict_mask = ((topk_vals > 0.7) & (topk_vals < 0.99)).float()

        drifted_new = new_content.clone()
        drifted_old = memory_bank.clone()

        if conflict_mask.sum() > 0:
            for batch_idx in range(new_content.shape[0]):
                for k in range(3):
                    if conflict_mask[batch_idx, k] <= 0.5:
                        continue

                    slot_idx = topk_idx[batch_idx, k]
                    pair = torch.cat([new_content[batch_idx], memory_bank[slot_idx]], dim=-1).unsqueeze(0)
                    conflict_prob = self.drift_detector(pair).squeeze()
                    if conflict_prob <= 0.5:
                        continue

                    strength = self.drift_strength(pair).squeeze()
                    avg = (new_content[batch_idx] + memory_bank[slot_idx]) / 2.0

                    drifted_new[batch_idx] = (1.0 - strength) * new_content[batch_idx] + strength * avg
                    drifted_old[slot_idx] = (1.0 - strength) * memory_bank[slot_idx] + strength * avg

        return conflict_mask, drifted_new, drifted_old

    def update_stability_plasticity(self, recent_changes: torch.Tensor) -> float:
        change_magnitude = float(recent_changes.abs().mean().item())
        self.memory_change_ema = 0.9 * self.memory_change_ema + 0.1 * change_magnitude

        if self.memory_change_ema.item() > self.stability_target:
            adjustment = -0.005
        else:
            adjustment = 0.005

        with torch.no_grad():
            self.decay_rate.data = torch.clamp(self.decay_rate + adjustment, 0.01, 0.2)

        return float(self.decay_rate.item())

    def forward(
        self,
        new_content: torch.Tensor,
        query: torch.Tensor,
        memory_bank: torch.Tensor,
        access_times: torch.Tensor,
        current_step: int,
    ) -> dict:
        memory_before = memory_bank.clone()
        updated_memory = memory_bank.clone()
        updated_access_times = access_times.clone()

        capacity = self.check_capacity(updated_memory)

        combined = torch.cat([new_content, query], dim=-1)
        relevance_features = self.store_relevance(combined)

        norm_new = F.normalize(new_content, dim=-1)
        norm_mem = F.normalize(updated_memory, dim=-1)
        similarities = torch.matmul(norm_new, norm_mem.t())
        max_sim = similarities.max(dim=-1, keepdim=True).values
        novelty = (1.0 - max_sim) / 2.0

        novelty_features = self.store_novelty(new_content)
        combined_features = relevance_features + novelty_features
        store_score = self.store_gate(combined_features)

        base_store = bool((store_score.mean() > self.store_threshold).item())
        novelty_ok = bool((novelty.mean() > self.min_novelty).item())
        should_store = base_store and novelty_ok

        victim = None
        erased_only = False
        if capacity > self.capacity_limit:
            victim = self.emergency_erase(updated_memory, updated_access_times, current_step)
            with torch.no_grad():
                updated_memory[victim] = torch.zeros_like(updated_memory[victim])
                updated_access_times[victim] = -99999.0

            if not novelty_ok:
                should_store = False
                erased_only = True

        erase_scores = self.erase_gate(
            torch.cat(
                [
                    self.erase_lru(((float(current_step) - updated_access_times) / 1000.0).unsqueeze(-1)),
                    self.erase_confidence(updated_memory),
                ],
                dim=-1,
            )
        ).squeeze(-1)

        conflict_mask, drifted_new, drifted_old = self._detect_conflict(new_content, updated_memory)
        updated_memory = drifted_old

        write_idx = None
        if should_store:
            write_idx = victim if victim is not None else int(erase_scores.argmax().item())
            with torch.no_grad():
                updated_memory[write_idx] = drifted_new[0]
                updated_access_times[write_idx] = float(current_step)

        recent_changes = (updated_memory - memory_before).abs().mean()
        new_decay = self.update_stability_plasticity(recent_changes)

        return {
            "store_score": float(store_score.mean().item()),
            "novelty": float(novelty.mean().item()),
            "relevance": float(torch.sigmoid(max_sim).mean().item()),
            "capacity": capacity,
            "emergency_erase": victim,
            "stored": should_store,
            "erased_only": erased_only,
            "write_index": write_idx,
            "conflict_detected": float(conflict_mask.sum().item()),
            "erase_scores": erase_scores,
            "decay_rate": new_decay,
            "updated_memory": updated_memory,
            "updated_access_times": updated_access_times,
        }
