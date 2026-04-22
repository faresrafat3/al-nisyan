"""
Embedding-level memory fusion for Al-Nisyan.

This module injects retrieved memory as a soft prefix in hidden space.
It avoids prompt-level RAG and keeps the base model frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MemoryFusedGenerator(nn.Module):
    """Build fused input embeddings from token embeddings + retrieved memory."""

    def __init__(self, hidden_dim: int, memory_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_projection = nn.Linear(memory_dim, hidden_dim)
        self.memory_gate = nn.Parameter(torch.tensor(0.5))

    def fuse(
        self,
        token_embeddings: torch.Tensor,
        memory_context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Args:
            token_embeddings: [batch, seq_len, hidden_dim]
            memory_context: [batch, memory_dim] or [batch, 1, memory_dim]

        Returns:
            fused_embeddings: [batch, seq_len + prefix_len, hidden_dim]
            attention_mask: [batch, seq_len + prefix_len]
            prefix_len: number of memory prefix tokens added
        """
        batch_size, seq_len, hidden_dim = token_embeddings.shape
        device = token_embeddings.device

        if memory_context is None:
            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
            return token_embeddings, attention_mask, 0

        if memory_context.dim() == 3:
            memory_context = memory_context.squeeze(1)

        memory_prefix = self.memory_projection(memory_context).unsqueeze(1)
        gate = torch.sigmoid(self.memory_gate)
        memory_prefix = gate * memory_prefix

        fused_embeddings = torch.cat([memory_prefix, token_embeddings], dim=1)
        attention_mask = torch.ones(
            (batch_size, seq_len + 1),
            dtype=torch.long,
            device=device,
        )

        return fused_embeddings, attention_mask, 1
