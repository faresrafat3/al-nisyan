import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.memory_bank import EpisodicMemoryBank


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    memory = EpisodicMemoryBank(num_slots=2048, slot_dim=512).to(device)

    query = torch.randn(4, 512, device=device)
    content = torch.randn(4, 512, device=device)

    retrieved_before, _ = memory.read(query)
    _, write_gate = memory(query, content)
    retrieved_after, _ = memory.read(query)

    delta = (retrieved_after - retrieved_before).norm(dim=-1).mean().item()

    print(f"device: {device}")
    print(f"write_gate: {write_gate:.4f}")
    print(f"retrieval_delta: {delta:.4f}")
    print(memory.get_memory_stats())


if __name__ == "__main__":
    main()
