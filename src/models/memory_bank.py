import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class EpisodicMemoryBank(nn.Module):
    """
    Differentiable memory bank with soft read/write.
    Each slot stores a "situation" (latent representation), not a fact.
    """

    def __init__(
        self,
        num_slots: int = 2048,
        slot_dim: int = 512,
        num_heads: int = 4,
        temperature: float = 0.1,
        active_threshold: float = 0.1,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_heads = num_heads
        self.temperature = temperature
        self.active_threshold = active_threshold

        self.memory = nn.Parameter(torch.randn(num_slots, slot_dim) * 0.02)
        self.query_proj = nn.Linear(slot_dim, slot_dim)

        self.write_gate = nn.Sequential(
            nn.Linear(slot_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.addressing = nn.Linear(slot_dim, num_slots)

    def read(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_proj(query)
        similarities = torch.matmul(q, self.memory.t())
        similarities = similarities / self.temperature

        attn_weights = F.softmax(similarities, dim=-1)
        retrieved = torch.matmul(attn_weights, self.memory)

        return retrieved, attn_weights

    def write(
        self,
        content: torch.Tensor,
        query: torch.Tensor,
        strength: Optional[torch.Tensor] = None,
    ) -> float:
        gate_input = torch.cat([content, query], dim=-1)
        write_gate = self.write_gate(gate_input)

        if strength is not None:
            write_gate = write_gate * strength.unsqueeze(-1)

        _, attn_weights = self.read(query)
        least_used = 1.0 - attn_weights

        write_gate_expanded = write_gate.unsqueeze(-1)
        least_used = least_used.unsqueeze(-1)
        content_expanded = content.unsqueeze(1)

        update_mask = least_used * write_gate_expanded
        update = (update_mask * content_expanded).mean(dim=0)

        with torch.no_grad():
            self.memory.data = 0.9 * self.memory.data + 0.1 * update

        return float(write_gate.mean().item())

    def forward(
        self,
        query: torch.Tensor,
        content: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[float]]:
        retrieved, _ = self.read(query)

        write_val = None
        if content is not None:
            write_val = self.write(content, query)

        return retrieved, write_val

    def get_memory_stats(self) -> dict:
        with torch.no_grad():
            mem_norms = torch.norm(self.memory, dim=-1)
            active_slots = (mem_norms > self.active_threshold).sum().item()

            mem_normalized = F.normalize(self.memory, dim=-1)
            similarity = torch.matmul(mem_normalized, mem_normalized.t())
            mask = 1 - torch.eye(self.num_slots, device=similarity.device)
            avg_similarity = (similarity * mask).sum() / mask.sum()

            return {
                "active_slots": active_slots,
                "total_slots": self.num_slots,
                "activation_ratio": active_slots / self.num_slots,
                "avg_slot_similarity": avg_similarity.item(),
                "memory_mean_norm": mem_norms.mean().item(),
            }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    mem = EpisodicMemoryBank(num_slots=2048, slot_dim=512).to(device)

    query = torch.randn(2, 512).to(device)
    content = torch.randn(2, 512).to(device)

    retrieved, _ = mem.read(query)
    print(f"Retrieved shape: {retrieved.shape}")
    print(f"Retrieved norm: {torch.norm(retrieved, dim=-1).mean().item():.4f}")

    retrieved, write_gate = mem(query, content)
    print(f"Write gate: {write_gate:.4f}")

    retrieved2, _ = mem.read(query)
    print(f"Retrieved after write norm: {torch.norm(retrieved2, dim=-1).mean().item():.4f}")

    stats = mem.get_memory_stats()
    print("\nMemory Stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    if device == "cuda":
        print(f"\nVRAM used: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
