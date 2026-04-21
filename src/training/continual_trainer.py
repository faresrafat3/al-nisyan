from dataclasses import dataclass
from typing import Dict

import torch

from src.controllers.forgetting_gate import ForgettingGate
from src.models.memory_bank import EpisodicMemoryBank


@dataclass
class TrainerConfig:
    slot_dim: int = 512
    lr: float = 1e-4


class ContinualTrainer:
    def __init__(self, config: TrainerConfig):
        self.config = config
        self.memory = EpisodicMemoryBank(slot_dim=config.slot_dim)
        self.controller = ForgettingGate(input_dim=config.slot_dim)
        self.optimizer = torch.optim.AdamW(self.controller.parameters(), lr=config.lr)

    def train_step(self, query: torch.Tensor, content: torch.Tensor) -> Dict[str, float]:
        strength = self.controller(query, content)
        retrieved, write_gate = self.memory(query, content=content)

        loss = torch.mean((retrieved - content) ** 2)
        loss = loss + 0.01 * (1.0 - strength).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "write_gate": float(write_gate if write_gate is not None else 0.0),
            "strength_mean": float(strength.mean().item()),
        }
