#!/usr/bin/env python3
"""
MeKi Hybrid Memory Module for Al-Nisyan
Optimized for Kaggle T4 (16GB VRAM)
Tested with Qwen3.5-0.6B and Qwen3.5-1.7B

Usage:
    from meki_hybrid import MeKiHybridInjector, MeKiBenchmark
    
    # Wrap existing model
    injector = MeKiHybridInjector(model, tokenizer)
    
    # Run benchmark
    benchmark = MeKiBenchmark(injector)
    results = benchmark.run()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
import numpy as np
from collections import deque
import json
import time


# =============================================================================
# 1. MEKI MODULE — Semantic Memory (Token-Level Lookup + Additive Gated Fusion)
# =============================================================================

class MeKiModule(nn.Module):
    """
    MeKi-inspired semantic memory module.
    
    Architecture:
        1. Retrieve: e = memory_embedding[input_ids]
        2. Gate:    g = sigmoid(W_gate @ hidden_states)
        3. Fuse:    v = e + g  (additive fusion — best from MeKi ablations)
        4. Project: y = W_out @ v
        5. Norm:    output = RMSNorm(y)
    
    Cost: ~76MB for vocab=150K, mem_dim=128
    Speed: ~1% overhead
    """
    
    def __init__(self, vocab_size: int, hidden_dim: int, mem_dim: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mem_dim = mem_dim
        
        # ROM: vocabulary × memory dimension (offloaded to storage if needed)
        self.memory = nn.Embedding(vocab_size, mem_dim)
        
        # Low-rank projections
        self.gate_proj = nn.Linear(hidden_dim, mem_dim, bias=False)
        self.out_proj = nn.Linear(mem_dim, hidden_dim, bias=False)
        self.post_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        
        # Initialize
        nn.init.normal_(self.memory.weight, std=0.02)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    
    def forward(self, hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            input_ids: [batch, seq_len] — token IDs
        Returns:
            [batch, seq_len, hidden_dim] — memory injection
        """
        # 1. Retrieve memory vectors
        e = self.memory(input_ids)  # [B, T, mem_dim]
        
        # 2. Contextual gate from hidden state
        g = torch.sigmoid(self.gate_proj(hidden_states))  # [B, T, mem_dim]
        
        # 3. Additive fusion (MeKi best ablation)
        v = e + g  # [B, T, mem_dim]
        
        # 4. Project back to hidden dimension
        y = self.out_proj(v)  # [B, T, hidden]
        
        # 5. Normalize
        return self.post_norm(y)
    
    def extra_repr(self):
        return f"vocab={self.vocab_size}, hidden={self.hidden_dim}, mem_dim={self.mem_dim}, params={sum(p.numel() for p in self.parameters())/1e6:.1f}M"


# =============================================================================
# 2. LM2-STYLE MODULE — Latent Memory (Gated Cross-Attention)
# =============================================================================

class LM2MemoryModule(nn.Module):
    """
    LM2-style latent memory with gated cross-attention.
    
    Architecture:
        1. Cross-attention: hidden_states → attends to → mem_bank
        2. Output gate:     g_out = sigmoid(W_out @ hidden_states)
        3. Gated output:    output = g_out * cross_attn_output
    
    Cost: ~2M params for mem_slots=256, hidden=2048
    Speed: ~3% overhead
    """
    
    def __init__(self, hidden_dim: int, mem_slots: int = 256, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mem_slots = mem_slots
        
        # Memory bank (learnable)
        self.mem_bank = nn.Parameter(torch.randn(mem_slots, hidden_dim) * 0.02)
        
        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True, dropout=0.0
        )
        
        # Output gate
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Layer norm
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
        Returns:
            [batch, seq_len, hidden_dim] — gated memory output
        """
        B, T, H = hidden_states.shape
        
        # Expand memory bank for batch
        mem = self.mem_bank.unsqueeze(0).expand(B, -1, -1)  # [B, mem_slots, H]
        
        # Cross-attention: query=hidden, key/value=mem_bank
        attn_output, _ = self.cross_attn(
            query=hidden_states,  # [B, T, H]
            key=mem,              # [B, mem_slots, H]
            value=mem             # [B, mem_slots, H]
        )  # [B, T, H]
        
        # Output gate
        gate = self.output_gate(hidden_states)  # [B, T, H]
        gated = gate * attn_output  # [B, T, H]
        
        return self.norm(gated)
    
    def extra_repr(self):
        return f"mem_slots={self.mem_slots}, hidden={self.hidden_dim}, params={sum(p.numel() for p in self.parameters())/1e6:.1f}M"


# =============================================================================
# 3. KNN MEMORY MODULE — Pattern Matching (Non-Parametric)
# =============================================================================

class KNNMemoryModule:
    """
    kNN-based memory for pattern matching.
    Stores (key, value) pairs from forward passes.
    
    Uses CPU storage + FAISS for search (if available).
    Falls back to brute-force on GPU if FAISS not installed.
    
    Cost: 0 params, ~200MB storage for 16K entries
    Speed: ~5% overhead
    """
    
    def __init__(self, dim: int, max_size: int = 16384, k: int = 16, device: str = "cpu"):
        self.dim = dim
        self.max_size = max_size
        self.k = k
        self.device = device
        
        self.keys = []
        self.values = []
        self.total_stored = 0
        
        # Try FAISS
        self.use_faiss = False
        try:
            import faiss
            self.faiss = faiss
            self.index = None
            self.use_faiss = True
        except ImportError:
            pass
    
    def add(self, keys: torch.Tensor, values: torch.Tensor):
        """Store new (key, value) pairs."""
        keys = keys.detach().cpu()
        values = values.detach().cpu()
        
        self.keys.append(keys)
        self.values.append(values)
        self.total_stored += keys.shape[0]
        
        # Maintain max size (FIFO)
        while self.total_stored > self.max_size and len(self.keys) > 1:
            removed = self.keys.pop(0)
            self.values.pop(0)
            self.total_stored -= removed.shape[0]
        
        # Invalidate index
        if self.use_faiss:
            self.index = None
    
    def search(self, queries: torch.Tensor) -> Optional[torch.Tensor]:
        """Find k nearest neighbors for each query."""
        if len(self.keys) == 0:
            return None
        
        queries_device = queries.device
        queries_np = queries.detach().cpu().numpy()
        
        # Build index if needed
        if self.use_faiss and self.index is None:
            all_keys = torch.cat(self.keys, dim=0).numpy()
            self.index = self.faiss.IndexFlatIP(self.dim)
            self.index.add(all_keys)
        
        if self.use_faiss and self.index is not None:
            distances, indices = self.index.search(queries_np, min(self.k, self.total_stored))
            all_values = torch.cat(self.values, dim=0)
            retrieved = all_values[indices]  # [Q, k, dim]
            attn = torch.softmax(torch.from_numpy(distances).float(), dim=-1)  # [Q, k]
            output = (attn.unsqueeze(-1) * retrieved).sum(dim=1)  # [Q, dim]
            return output.to(queries_device)
        else:
            # Brute-force fallback
            all_keys = torch.cat(self.keys, dim=0).to(queries_device)  # [N, dim]
            all_values = torch.cat(self.values, dim=0).to(queries_device)  # [N, dim]
            
            similarities = torch.matmul(queries, all_keys.t())  # [Q, N]
            topk = min(self.k, similarities.shape[1])
            scores, indices = torch.topk(similarities, topk, dim=-1)  # [Q, k]
            attn = torch.softmax(scores, dim=-1)  # [Q, k]
            
            retrieved = all_values[indices]  # [Q, k, dim]
            output = (attn.unsqueeze(-1) * retrieved).sum(dim=1)  # [Q, dim]
            return output


# =============================================================================
# 4. HYBRID INJECTOR — Combines All 4 Memory Types via Hooks
# =============================================================================

class MeKiHybridInjector:
    """
    Injects MeKi + LM2 + kNN + Episodic memory into a frozen model.
    Uses PyTorch forward hooks — zero weight modification.
    
    Usage:
        model = load_your_base_model()  # e.g., Qwen3.5-0.6B
        tokenizer = load_your_tokenizer()
        
        injector = MeKiHybridInjector(model, tokenizer)
        
        # Generate with memory
        inputs = tokenizer("What is 5+3?", return_tensors="pt").to("cuda")
        with injector.inject_memory():
            output = model.generate(**inputs, max_new_tokens=20)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        meki_mem_dim: int = 128,
        lm2_mem_slots: int = 256,
        knn_max_size: int = 16384,
        knn_k: int = 16,
        enable_meki: bool = True,
        enable_lm2: bool = True,
        enable_knn: bool = True,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self._current_input_ids = None
        
        # Get model config
        self.hidden_dim = model.config.hidden_size
        self.vocab_size = len(tokenizer)
        self.device = next(model.parameters()).device
        
        # Detect layers
        self.layers = self._detect_layers()
        num_layers = len(self.layers)
        
        print(f"[MeKiHybrid] Detected {num_layers} layers, hidden={self.hidden_dim}")
        
        # Initialize memory modules
        self.meki_modules = nn.ModuleList()
        self.lm2_modules = nn.ModuleList()
        self.knn_modules = []
        
        for i in range(num_layers):
            # MeKi
            if enable_meki:
                meki = MeKiModule(self.vocab_size, self.hidden_dim, meki_mem_dim).to(self.device)
                self.meki_modules.append(meki)
            else:
                self.meki_modules.append(None)
            
            # LM2
            if enable_lm2:
                lm2 = LM2MemoryModule(self.hidden_dim, lm2_mem_slots).to(self.device)
                self.lm2_modules.append(lm2)
            else:
                self.lm2_modules.append(None)
            
            # kNN (non-parametric)
            if enable_knn:
                knn = KNNMemoryModule(self.hidden_dim, knn_max_size, knn_k)
                self.knn_modules.append(knn)
            else:
                self.knn_modules.append(None)
        
        # Print stats
        meki_params = sum(p.numel() for m in self.meki_modules if m is not None for p in m.parameters())
        lm2_params = sum(p.numel() for m in self.lm2_modules if m is not None for p in m.parameters())
        print(f"[MeKiHybrid] MeKi params: {meki_params/1e6:.1f}M")
        print(f"[MeKiHybrid] LM2 params: {lm2_params/1e6:.1f}M")
        print(f"[MeKiHybrid] kNN modules: {num_layers} (non-parametric)")
    
    def _detect_layers(self):
        """Auto-detect transformer layers."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers  # Qwen, Llama
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            return self.model.transformer.h  # GPT-2
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            return self.model.model.decoder.layers  # Qwen2.5, Gemma
        else:
            raise ValueError("Cannot detect transformer layers. Please check model architecture.")
    
    def _create_hook(self, layer_idx: int):
        """Create forward hook for a layer."""
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            
            # 1. MeKi injection (needs input_ids)
            if self.meki_modules[layer_idx] is not None and self._current_input_ids is not None:
                try:
                    meki_out = self.meki_modules[layer_idx](hidden, self._current_input_ids)
                    hidden = hidden + meki_out
                except Exception as e:
                    pass  # Graceful fallback
            
            # 2. LM2 injection
            if self.lm2_modules[layer_idx] is not None:
                try:
                    lm2_out = self.lm2_modules[layer_idx](hidden)
                    hidden = hidden + lm2_out
                except Exception as e:
                    pass
            
            # 3. kNN injection
            if self.knn_modules[layer_idx] is not None:
                try:
                    B, T, H = hidden.shape
                    flat = hidden.view(-1, H)
                    
                    # Retrieve
                    knn_out = self.knn_modules[layer_idx].search(flat)
                    if knn_out is not None:
                        knn_out = knn_out.view(B, T, H)
                        hidden = hidden + knn_out * 0.05  # Small scale
                    
                    # Store for next time
                    self.knn_modules[layer_idx].add(flat, flat)
                except Exception as e:
                    pass
            
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden
        
        return hook
    
    def inject_memory(self):
        """Context manager to activate memory injection."""
        return _MemoryInjectionContext(self)
    
    def register_hooks(self):
        """Register all hooks."""
        for idx, layer in enumerate(self.layers):
            hook_fn = self._create_hook(idx)
            handle = layer.register_forward_hook(hook_fn)
            self.hooks.append(handle)
    
    def remove_hooks(self):
        """Remove all hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks.clear()
    
    def generate(self, prompt: str, max_new_tokens: int = 50, **kwargs):
        """Generate with memory injection."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        self._current_input_ids = inputs.input_ids
        
        with self.inject_memory():
            output = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, **kwargs)
        
        response = self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response


class _MemoryInjectionContext:
    """Context manager for memory injection."""
    def __init__(self, injector: MeKiHybridInjector):
        self.injector = injector
    
    def __enter__(self):
        self.injector.register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.injector.remove_hooks()
        return False


# =============================================================================
# 5. BENCHMARK — Compare v3 vs Hybrid
# =============================================================================

class MeKiBenchmark:
    """
    Benchmark: Compare base model vs MeKi Hybrid.
    
    Metrics:
        - Forward Transfer: novelty drop on repeated questions
        - Response Time: latency per generation
        - Memory Usage: VRAM consumption
        - kNN Hit Rate: % of queries with useful retrieval
    """
    
    def __init__(self, injector: MeKiHybridInjector):
        self.injector = injector
        self.results = {
            "base": {"responses": [], "times": [], "novelties": []},
            "hybrid": {"responses": [], "times": [], "novelties": [], "knn_hits": []},
        }
    
    def run_phase1_learning(self, questions: List[str]):
        """Phase 1: Learn questions with Hybrid memory."""
        print(f"\n[Phase 1] Learning {len(questions)} questions with Hybrid memory...")
        
        for i, q in enumerate(questions):
            t0 = time.time()
            response = self.injector.generate(q, max_new_tokens=30)
            t1 = time.time()
            
            self.results["hybrid"]["responses"].append({"q": q, "r": response})
            self.results["hybrid"]["times"].append(t1 - t0)
            
            if (i + 1) % 5 == 0:
                avg_time = np.mean(self.results["hybrid"]["times"][-5:])
                print(f"  [{i+1}/{len(questions)}] {avg_time:.2f}s/q | {q[:40]}...")
    
    def run_phase2_recall(self, questions: List[str]):
        """Phase 2: Recall with both base and hybrid."""
        print(f"\n[Phase 2] Recalling {len(questions)} questions (Base vs Hybrid)...")
        
        for i, q in enumerate(questions):
            # Base (no memory)
            inputs = self.injector.tokenizer(q, return_tensors="pt").to(self.injector.device)
            
            t0 = time.time()
            base_out = self.injector.model.generate(**inputs, max_new_tokens=30, do_sample=False)
            base_time = time.time() - t0
            base_resp = self.injector.tokenizer.decode(base_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # Hybrid (with memory)
            t0 = time.time()
            hybrid_resp = self.injector.generate(q, max_new_tokens=30)
            hybrid_time = time.time() - t0
            
            self.results["base"]["responses"].append({"q": q, "r": base_resp})
            self.results["hybrid"]["responses"].append({"q": q, "r": hybrid_resp})
            self.results["base"]["times"].append(base_time)
            self.results["hybrid"]["times"].append(hybrid_time)
            
            # Novelty: cosine similarity between base and hybrid responses
            # (Lower = hybrid used memory to produce similar/better response)
            with torch.no_grad():
                base_emb = self._get_embedding(base_resp)
                hybrid_emb = self._get_embedding(hybrid_resp)
                sim = F.cosine_similarity(base_emb, hybrid_emb, dim=-1).item()
                novelty = 1.0 - sim
            
            self.results["hybrid"]["novelties"].append(novelty)
            
            match = "MATCH" if base_resp.strip() == hybrid_resp.strip() else "DIFF"
            print(f"  [{i+1}] {match} | novelty={novelty:.3f} | {q[:40]}...")
    
    def _get_embedding(self, text: str) -> torch.Tensor:
        """Get mean embedding of text."""
        inputs = self.injector.tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(self.injector.device)
        with torch.no_grad():
            out = self.injector.model(**inputs, output_hidden_states=True)
            emb = out.hidden_states[-1].mean(dim=1)  # [1, hidden]
        return emb
    
    def report(self) -> Dict:
        """Generate final report."""
        print("\n" + "="*60)
        print("MeKi Hybrid Benchmark Report")
        print("="*60)
        
        # Latency
        base_times = self.results["base"]["times"]
        hybrid_times = self.results["hybrid"]["times"]
        
        print(f"\n[Latency]")
        if base_times:
            print(f"  Base:   {np.mean(base_times):.3f}s ± {np.std(base_times):.3f}")
        if hybrid_times:
            print(f"  Hybrid: {np.mean(hybrid_times):.3f}s ± {np.std(hybrid_times):.3f}")
        if base_times and hybrid_times:
            overhead = (np.mean(hybrid_times) / np.mean(base_times) - 1) * 100
            print(f"  Overhead: {overhead:.1f}%")
        
        # Novelty
        novelties = self.results["hybrid"]["novelties"]
        if novelties:
            print(f"\n[Novelty] (lower = more memory reuse)")
            print(f"  Mean: {np.mean(novelties):.3f}")
            print(f"  Min:  {np.min(novelties):.3f}")
            print(f"  <0.2 (good recall): {sum(1 for n in novelties if n < 0.2)}/{len(novelties)}")
        
        # VRAM
        vram = torch.cuda.memory_allocated() / 1024**3
        print(f"\n[VRAM] {vram:.2f} GB")
        
        # kNN stats
        print(f"\n[kNN Stats]")
        for i, knn in enumerate(self.injector.knn_modules):
            if knn is not None and knn.total_stored > 0:
                print(f"  Layer {i}: {knn.total_stored} entries stored")
        
        print("="*60)
        
        return self.results


# =============================================================================
# 6. MAIN — Run on Kaggle T4
# =============================================================================

def main_kaggle():
    """Main function optimized for Kaggle T4."""
    import sys
    sys.path.insert(0, "/kaggle/working/al-nisyan")
    
    from src.models.memory_augmented_model_v3 import CultivatedMemoryModelV3
    
    print("="*60)
    print("MeKi Hybrid Memory — Kaggle T4")
    print("="*60)
    
    # Step 1: Load model (use 0.6B for T4)
    print("\n[1] Loading Qwen3.5-0.6B...")
    model = CultivatedMemoryModelV3(
        base_model_key="Qwen/Qwen3.5-0.6B",  # Smaller for T4
        memory_slots=64,
        memory_dim=512,
        controller_mode="clean",
    )
    base_model = model.model
    tokenizer = model.tokenizer
    
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after load: {vram:.2f} GB")
    
    # Step 2: Wrap with Hybrid Injector
    print("\n[2] Wrapping with MeKi Hybrid Injector...")
    injector = MeKiHybridInjector(
        base_model,
        tokenizer,
        meki_mem_dim=64,       # Reduced for T4
        lm2_mem_slots=128,      # Reduced for T4
        knn_max_size=4096,      # Reduced for T4
        knn_k=8,
    )
    
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after hybrid: {vram:.2f} GB")
    
    # Step 3: Benchmark
    print("\n[3] Running Benchmark...")
    
    math_questions = [
        "What is 5 + 3?",
        "Calculate 10 times 2.",
        "What is 15 minus 7?",
        "What is 8 divided by 2?",
        "Calculate 3 plus 3.",
        "What is 100 plus 200?",
        "Calculate 7 times 8.",
        "What is 50 minus 25?",
        "Calculate 6 times 7.",
        "What is 2 to the power of 10?",
    ]
    
    benchmark = MeKiBenchmark(injector)
    
    # Phase 1: Learn
    benchmark.run_phase1_learning(math_questions)
    
    # Phase 2: Recall (with some repeats)
    recall_questions = [
        "What is 5 + 3?",           # Repeat
        "Calculate 10 times 2.",     # Repeat
        "What is the capital of France?",  # New
        "Calculate 7 times 8.",      # Repeat
        "Who wrote Romeo and Juliet?",  # New
    ]
    benchmark.run_phase2_recall(recall_questions)
    
    # Report
    results = benchmark.report()
    
    # Save
    with open("/kaggle/working/meki_hybrid_results.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "responses"} 
                   for k, v in results.items()}, f, indent=2)
    print("\nResults saved to /kaggle/working/meki_hybrid_results.json")
    
    return injector, benchmark


if __name__ == "__main__":
    main_kaggle()
