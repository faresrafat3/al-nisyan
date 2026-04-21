import torch
import torch.nn as nn
from typing import Optional, Tuple


class ForgettingController(nn.Module):
    """
    5M-aimed controller that manages memory lifecycle.

    Core idea:
    1) STORE novel+relevant experiences
    2) ERASE stale/low-confidence memories
    3) DRIFT conflicting memories to preserve structure over exact detail
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        num_slots: int = 2048,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_slots = num_slots

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

        self.decay_rate = nn.Parameter(torch.tensor(0.01))
        self.register_buffer("memory_change_ema", torch.zeros(1))
        self.stability_target = 0.1

    def compute_store_score(
        self,
        content: torch.Tensor,
        query: torch.Tensor,
        memory_bank: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        combined = torch.cat([content, query], dim=-1)
        relevance_features = self.store_relevance(combined)

        similarities = torch.matmul(content, memory_bank.t())
        max_sim = similarities.max(dim=-1, keepdim=True).values
        novelty = 1.0 - torch.sigmoid(max_sim * 5.0)

        novelty_features = self.store_novelty(content)
        combined_features = relevance_features + novelty_features
        store_score = self.store_gate(combined_features)

        relevance = torch.sigmoid(max_sim)
        return store_score, novelty, relevance

    def compute_erase_mask(
        self,
        memory_bank: torch.Tensor,
        access_times: torch.Tensor,
        current_step: int,
    ) -> torch.Tensor:
        time_since_access = (current_step - access_times).unsqueeze(-1).float()
        time_since_access = time_since_access / 1000.0

        lru_features = self.erase_lru(time_since_access)
        confidence_features = self.erase_confidence(memory_bank)

        combined = torch.cat([lru_features, confidence_features], dim=-1)
        erase_scores = self.erase_gate(combined).squeeze(-1)

        decay = torch.exp(-self.decay_rate * time_since_access.squeeze(-1))
        erase_scores = erase_scores * (1.0 - decay)

        return erase_scores

    def detect_and_resolve_conflict(
        self,
        new_content: torch.Tensor,
        memory_bank: torch.Tensor,
        top_k: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        similarities = torch.matmul(new_content, memory_bank.t())
        topk_vals, topk_idx = similarities.topk(top_k, dim=-1)

        conflict_threshold = 0.7
        identical_threshold = 0.99
        candidate_conflicts = (topk_vals > conflict_threshold) & (topk_vals < identical_threshold)

        drifted_new = new_content.clone()
        drifted_old = memory_bank.clone()
        full_conflict = torch.zeros(memory_bank.shape[0], device=memory_bank.device)

        for batch_idx in range(new_content.shape[0]):
            for top_idx in range(top_k):
                if not candidate_conflicts[batch_idx, top_idx]:
                    continue

                slot_idx = topk_idx[batch_idx, top_idx]
                old_mem = memory_bank[slot_idx]
                new_mem = new_content[batch_idx]

                pair = torch.cat([new_mem, old_mem], dim=-1).unsqueeze(0)
                conflict_prob = self.drift_detector(pair).squeeze()
                if conflict_prob <= 0.5:
                    continue

                strength = self.drift_strength(pair).squeeze()
                avg = (new_mem + old_mem) / 2.0

                drifted_new[batch_idx] = (1.0 - strength) * new_mem + strength * avg
                drifted_old[slot_idx] = (1.0 - strength) * old_mem + strength * avg
                full_conflict[slot_idx] = 1.0

        return full_conflict, drifted_new, drifted_old

    def update_stability_plasticity(self, recent_changes: torch.Tensor) -> float:
        if isinstance(recent_changes, torch.Tensor):
            change_magnitude = float(recent_changes.abs().mean().item())
        else:
            change_magnitude = float(abs(recent_changes))

        self.memory_change_ema = 0.9 * self.memory_change_ema + 0.1 * change_magnitude

        adjustment = -0.001 if self.memory_change_ema.item() > self.stability_target else 0.001

        with torch.no_grad():
            self.decay_rate.data = torch.clamp(self.decay_rate + adjustment, 0.001, 0.1)

        return float(self.decay_rate.item())

    def forward(
        self,
        new_content: torch.Tensor,
        query: torch.Tensor,
        memory_bank: torch.Tensor,
        access_times: torch.Tensor,
        current_step: Optional[int] = None,
        step: Optional[int] = None,
    ) -> dict:
        resolved_step = current_step if current_step is not None else step
        if resolved_step is None:
            raise ValueError("Either current_step or step must be provided.")

        store_score, novelty, relevance = self.compute_store_score(new_content, query, memory_bank)
        erase_scores = self.compute_erase_mask(memory_bank, access_times, resolved_step)

        conflict_mask, drifted_new, drifted_old = self.detect_and_resolve_conflict(
            new_content,
            memory_bank,
        )

        updated_memory = drifted_old.clone()
        updated_access_times = access_times.clone()

        if store_score.mean() > 0.5:
            erase_idx = torch.argmax(erase_scores)
            updated_memory[erase_idx] = drifted_new[0]
            updated_access_times[erase_idx] = float(resolved_step)

        recent_changes = (updated_memory - memory_bank).abs().mean()
        new_decay = self.update_stability_plasticity(recent_changes)

        return {
            "store_score": float(store_score.mean().item()),
            "novelty": float(novelty.mean().item()),
            "relevance": float(relevance.mean().item()),
            "erase_scores": erase_scores,
            "conflict_detected": float(conflict_mask.sum().item()),
            "decay_rate": new_decay,
            "updated_memory": updated_memory,
            "updated_access_times": updated_access_times,
        }


class ForgettingGate(nn.Module):
    """Backward-compatible gate used by earlier trainer code."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, query: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        gate_input = torch.cat([query, content], dim=-1)
        return self.net(gate_input).squeeze(-1)


if __name__ == "__main__":
    print("=" * 60)
    print("Forgetting Controller Test")
    print("The Core of 'Al-Nisyan'")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    controller = ForgettingController(
        input_dim=256,
        hidden_dim=128,
        num_slots=512,
    ).to(device)

    params_m = sum(p.numel() for p in controller.parameters()) / 1e6
    print(f"\nParameters: {params_m:.2f}M")

    memory_bank = torch.randn(512, 256, device=device) * 0.1
    access_times = torch.zeros(512, device=device)

    print("\n[Test 1] Novel Content")
    new_content = torch.randn(1, 256, device=device)
    query = torch.randn(1, 256, device=device)

    result = controller(new_content, query, memory_bank, access_times, step=100)
    memory_bank = result["updated_memory"]
    access_times = result["updated_access_times"]
    print(f"Store score: {result['store_score']:.4f}")
    print(f"Novelty: {result['novelty']:.4f}")
    print(f"Conflicts: {result['conflict_detected']:.0f}")
    print(f"Decay rate: {result['decay_rate']:.6f}")

    print("\n[Test 2] Conflicting Content")
    similar_content = memory_bank[0].unsqueeze(0).clone() + torch.randn(1, 256, device=device) * 0.01

    result2 = controller(similar_content, query, memory_bank, access_times, step=101)
    memory_bank = result2["updated_memory"]
    access_times = result2["updated_access_times"]
    print(f"Store score: {result2['store_score']:.4f}")
    print(f"Conflicts: {result2['conflict_detected']:.0f}")
    print(f"Decay rate: {result2['decay_rate']:.6f}")

    print("\n[Test 3] Stability Adaptation")
    for i in range(10):
        random_content = torch.randn(1, 256, device=device)
        result3 = controller(random_content, query, memory_bank, access_times, step=200 + i)
        memory_bank = result3["updated_memory"]
        access_times = result3["updated_access_times"]
        if i % 3 == 0:
            print(f"Step {200 + i}: decay={result3['decay_rate']:.6f}, store={result3['store_score']:.4f}")

    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)
