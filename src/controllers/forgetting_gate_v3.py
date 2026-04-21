import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveForgettingController(nn.Module):
    """
    v3: dynamic novelty threshold based on memory pressure.

    When memory has more free space, threshold is lower (store more).
    When memory gets crowded, threshold rises (store selectively).
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_slots: int = 2048,
        capacity_limit: float = 0.85,
        store_threshold: float = 0.40,
        activation_threshold: float = 0.5,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_slots = num_slots
        self.capacity_limit = capacity_limit
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

        self.decay_rate = nn.Parameter(torch.tensor(0.02))
        self.register_buffer("memory_change_ema", torch.zeros(1))
        self.stability_target = 0.1

    def compute_dynamic_threshold(self, capacity: float) -> float:
        if capacity < 0.3:
            threshold = 0.08
        elif capacity < 0.6:
            threshold = 0.08 + (capacity - 0.3) * 0.733
        else:
            threshold = 0.30 + (capacity - 0.6) * 1.0
        return float(max(0.0, min(0.7, threshold)))

    def compute_topk_threshold(self, capacity: float) -> float:
        if capacity < 0.3:
            return 0.10
        elif capacity < 0.6:
            return 0.20
        else:
            return 0.40

    def check_capacity(self, memory_bank: torch.Tensor, access_times: torch.Tensor | None = None) -> float:
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
        return int(erase_score.argmax().item())

    def _detect_conflict(self, new_content: torch.Tensor, memory_bank: torch.Tensor):
        new_norm = F.normalize(new_content, dim=-1)
        mem_norm = F.normalize(memory_bank, dim=-1)
        similarities = torch.matmul(new_norm, mem_norm.t())
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

        adjustment = -0.002 if self.memory_change_ema.item() > self.stability_target else 0.002
        with torch.no_grad():
            self.decay_rate.data = torch.clamp(self.decay_rate + adjustment, 0.01, 0.03)

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

        capacity = self.check_capacity(updated_memory, updated_access_times)
        dynamic_threshold = self.compute_dynamic_threshold(capacity)

        combined = torch.cat([new_content, query], dim=-1)
        relevance_features = self.store_relevance(combined)

        new_norm = F.normalize(new_content, dim=-1)
        mem_norm = F.normalize(updated_memory, dim=-1)
        similarities = torch.matmul(new_norm, mem_norm.t())
        max_sim = similarities.max(dim=-1, keepdim=True).values
        novelty = (1.0 - max_sim) / 2.0

        novelty_features = self.store_novelty(new_content)
        combined_features = relevance_features + novelty_features
        raw_store_score = self.store_gate(combined_features)

        active_mask = torch.norm(updated_memory, dim=-1) > self.activation_threshold
        active_memory = updated_memory[active_mask]
        if active_memory.shape[0] > 0:
            active_norms = F.normalize(active_memory, dim=-1)
            slot_similarities = torch.matmul(new_norm, active_norms.t()).squeeze(0)
            slot_novelties = 1.0 - slot_similarities
            novelty_percentile = (novelty.mean() > slot_novelties).float().mean()
        else:
            novelty_percentile = torch.tensor(1.0, device=new_content.device)

        topk_threshold = self.compute_topk_threshold(capacity)
        raw_threshold = 0.3 if capacity < 0.3 else 0.5

        base_store = bool((raw_store_score.mean() > raw_threshold).item())
        novelty_ok = bool((novelty.mean() > dynamic_threshold).item())
        topk_ok = bool((novelty_percentile > topk_threshold).item())
        should_store = base_store and novelty_ok and topk_ok

        print(
            f"  [DEBUG] capacity={capacity:.3f} threshold={dynamic_threshold:.3f} "
            f"novelty={novelty.mean().item():.3f} store_score={raw_store_score.mean().item():.3f} "
            f"raw_threshold={raw_threshold:.3f} percentile={novelty_percentile.item():.3f} topk_threshold={topk_threshold:.3f} "
            f"base_store={base_store} novelty_ok={novelty_ok} topk_ok={topk_ok} should_store={should_store}",
            flush=True,
        )

        victim = None
        erased_only = False
        if capacity > self.capacity_limit:
            victim = self.emergency_erase(updated_memory, updated_access_times, current_step)
            print(f"  [DEBUG] EMERGENCY ERASE slot {victim}", flush=True)
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
            if victim is None:
                cooldown_steps = 3
                slot_age = float(current_step) - updated_access_times
                recent_writes = (updated_access_times >= 0) & (slot_age < cooldown_steps)
                cooldown_mask = (~recent_writes).float()
                masked_erase_scores = erase_scores * cooldown_mask

                print(
                    f"  [DEBUG] cooldown_steps={cooldown_steps} "
                    f"recent_slots={int(recent_writes.sum().item())}",
                    flush=True,
                )

                if float(masked_erase_scores.max().item()) <= 0.0:
                    masked_erase_scores = slot_age.float()
                    print("  [DEBUG] cooldown fallback: pure LRU", flush=True)

                write_idx = int(masked_erase_scores.argmax().item())
                erase_scores = masked_erase_scores
            else:
                write_idx = victim
            old_norm = torch.norm(updated_memory[write_idx]).item()
            print(f"  [DEBUG] WRITING to slot {write_idx}", flush=True)
            with torch.no_grad():
                updated_memory[write_idx] = drifted_new[0]
                updated_access_times[write_idx] = float(current_step)
            new_norm = torch.norm(updated_memory[write_idx]).item()
            print(f"  [DEBUG] slot norm: {old_norm:.4f} -> {new_norm:.4f}", flush=True)
        else:
            print("  [DEBUG] SKIP WRITE", flush=True)

        recent_changes = (updated_memory - memory_before).abs().mean()
        new_decay = self.update_stability_plasticity(recent_changes)
        print(f"  [DEBUG] decay_rate={new_decay:.4f}", flush=True)

        return {
            "store_score": float(raw_store_score.mean().item()),
            "novelty": float(novelty.mean().item()),
            "dynamic_threshold": dynamic_threshold,
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
