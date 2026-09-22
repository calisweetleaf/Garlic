"""
LoRA-MoE: Complete System Integration
======================================
Full LoRA-MoE system bringing together all components.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import math
from collections import defaultdict

from .core_components import LoRAMoEConfig, GraphMemoryConfig, DynamicRankLoRAAdapter, HierarchicalLoRAExpert
from .router_memory import EntropyRegularizedRouter, HSGMMemorySystem


# ============================================================
# SEMANTIC SIGNAL PREDICTOR
# ============================================================

class SemanticSignal(Enum):
    GENERAL = "general"
    PYTHON_CODE = "python_code"
    ALGORITHM = "algorithm"
    DEBUGGING = "debugging"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    DOCUMENTATION = "documentation"
    TESTING = "testing"
    REASONING = "reasoning"
    QUESTION = "question"


class SemanticSignalPredictor(nn.Module):
    """Predicts semantic signals from hidden states for expert routing."""

    def __init__(self, hidden_dim: int = 768, num_signals: int = 10, embedding_dim: int = 384):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_signals = num_signals
        self.embedding_dim = embedding_dim

        self.signal_embeddings = nn.Parameter(torch.randn(num_signals, embedding_dim))
        self._init_signal_embeddings()

        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU()
        )

        self.signal_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 8, num_signals)
        )

        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 8),
            nn.GELU(),
            nn.Linear(hidden_dim // 8, 1),
            nn.Sigmoid()
        )

        self.signal_names = [s.value for s in SemanticSignal]

    def _init_signal_embeddings(self):
        q, _ = torch.linalg.qr(self.signal_embeddings.T)
        self.signal_embeddings.data = q.T[:self.num_signals] * math.sqrt(self.embedding_dim)

    def forward(self, hidden_states: torch.Tensor):
        pooled = hidden_states.mean(dim=1)
        features = self.feature_extractor(pooled)

        signal_logits = self.signal_classifier(features)
        signal_probs = F.softmax(signal_logits, dim=-1)

        top_probs, top_indices = torch.max(signal_probs, dim=-1)
        confidence = self.confidence_estimator(features).squeeze(-1)

        batch_size = hidden_states.shape[0]
        predicted_embeddings = self.signal_embeddings[top_indices]

        seq_len = hidden_states.shape[1]
        signal_embeddings_expanded = predicted_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        if self.embedding_dim < self.hidden_dim:
            padding = torch.zeros(batch_size, seq_len, self.hidden_dim - self.embedding_dim,
                                 device=signal_embeddings_expanded.device)
            signal_embeddings_expanded = torch.cat([signal_embeddings_expanded, padding], dim=-1)

        return {
            'signal_logits': signal_logits,
            'signal_probs': signal_probs,
            'predicted_signal_idx': top_indices,
            'predicted_signal': [self.signal_names[i] for i in top_indices.tolist()],
            'confidence': confidence,
            'signal_embedding': signal_embeddings_expanded,
            'top_probability': top_probs
        }


# ============================================================
# STEP ENTROPY CALCULATOR
# ============================================================

class EntropyLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class StepEntropyCalculator:
    """Calculates step-level entropy with adaptive thresholds."""

    def __init__(self, threshold_low: float = 2.0, threshold_high: float = 4.0,
                 base: float = 2.0, adaptive_thresholds: bool = True):
        self.threshold_low = threshold_low
        self.threshold_high = threshold_high
        self.base = base
        self.adaptive_thresholds = adaptive_thresholds

        self.entropy_history: List[float] = []
        self.max_history = 100

        self.stats = {
            'steps_analyzed': 0,
            'steps_low': 0,
            'steps_medium': 0,
            'steps_high': 0,
            'compression_ratio': 0.0
        }

    def compute_token_entropy(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -torch.sum(probs * log_probs, dim=-1)

        if self.base == 2.0:
            entropy = entropy / math.log(2)

        return entropy

    def compute_step_entropy(self, token_entropies: torch.Tensor):
        if token_entropies.dim() == 2:
            return token_entropies.mean(dim=-1)
        else:
            return token_entropies.mean()

    def classify_entropy(self, step_entropy: float):
        if self.adaptive_thresholds and len(self.entropy_history) > 10:
            hist_mean = np.mean(self.entropy_history)
            hist_std = np.std(self.entropy_history)
            adaptive_low = max(self.threshold_low, hist_mean - 0.5 * hist_std)
            adaptive_high = min(self.threshold_high, hist_mean + 0.5 * hist_std)
        else:
            adaptive_low = self.threshold_low
            adaptive_high = self.threshold_high

        if step_entropy < adaptive_low:
            return EntropyLevel.LOW
        elif step_entropy > adaptive_high:
            return EntropyLevel.HIGH
        else:
            return EntropyLevel.MEDIUM

    def __call__(self, logits: torch.Tensor):
        """Make the calculator callable."""
        token_entropies = self.compute_token_entropy(logits)
        step_entropy = self.compute_step_entropy(token_entropies)

        avg_entropy = step_entropy.mean().item()
        entropy_level = self.classify_entropy(avg_entropy)

        self.entropy_history.append(avg_entropy)
        if len(self.entropy_history) > self.max_history:
            self.entropy_history.pop(0)

        self.stats['steps_analyzed'] += 1
        if entropy_level == EntropyLevel.LOW:
            self.stats['steps_low'] += 1
        elif entropy_level == EntropyLevel.MEDIUM:
            self.stats['steps_medium'] += 1
        else:
            self.stats['steps_high'] += 1

        if entropy_level == EntropyLevel.LOW:
            routing_path, action, use_experts = 'fast', 'skip_or_compress', False
        elif entropy_level == EntropyLevel.MEDIUM:
            routing_path, action, use_experts = 'normal', 'continue', False
        else:
            routing_path, action, use_experts = 'slow', 'retrieve_and_reason', True

        return {
            'token_entropies': token_entropies,
            'step_entropy': step_entropy,
            'avg_entropy': avg_entropy,
            'entropy_level': entropy_level.value,
            'routing_path': routing_path,
            'action': action,
            'use_experts': use_experts
        }

    def forward(self, logits: torch.Tensor):
        """Alias for __call__ for compatibility."""
        return self(logits)

    def get_stats(self):
        total = self.stats['steps_analyzed']
        if total > 0:
            return {
                **self.stats,
                'pct_low': self.stats['steps_low'] / total * 100,
                'pct_medium': self.stats['steps_medium'] / total * 100,
                'pct_high': self.stats['steps_high'] / total * 100,
                'avg_entropy_recent': np.mean(self.entropy_history) if self.entropy_history else 0
            }
        return self.stats


# ============================================================
# COMPLETE LoRA-MoE SYSTEM
# ============================================================

class LoRAMoESystem(nn.Module):
    """Complete LoRA-MoE System with all innovations."""

    def __init__(self, config: LoRAMoEConfig):
        super().__init__()
        self.config = config

        self.router = EntropyRegularizedRouter(
            hidden_dim=config.hidden_dim,
            graph_context_dim=config.hidden_dim,
            num_experts=config.num_experts,
            top_k=config.top_k,
            entropy_threshold_low=config.entropy_threshold_low,
            entropy_threshold_high=config.entropy_threshold_high,
            temperature=config.routing_temperature
        )

        self.experts = nn.ModuleList([
            HierarchicalLoRAExpert(
                input_dim=config.hidden_dim,
                num_sub_experts=config.num_sub_experts,
                rank_base=config.lora_rank_base,
                rank_max=config.lora_rank_max,
                alpha=config.lora_alpha,
                expert_type=expert_type
            )
            for expert_type in config.expert_types
        ])

        hsgm_config = GraphMemoryConfig(hidden_dim=config.hidden_dim, max_nodes=1024)
        self.memory = HSGMMemorySystem(hsgm_config)

        self.signal_predictor = SemanticSignalPredictor(
            hidden_dim=config.hidden_dim,
            num_signals=len(config.expert_types)
        )

        self.entropy_calc = StepEntropyCalculator(
            threshold_low=config.entropy_threshold_low,
            threshold_high=config.entropy_threshold_high
        )

        self.output_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU()
        )

        self.call_count = 0
        self.expert_usage = defaultdict(int)

    def forward(self, hidden_states: torch.Tensor, logits: Optional[torch.Tensor] = None,
                use_memory: bool = True):
        batch_size, seq_len, hidden_dim = hidden_states.shape

        entropy_info = None
        if logits is not None:
            entropy_info = self.entropy_calc(logits)

        memory_output = None
        if use_memory:
            memory_output = self.memory.compress(hidden_states)
            graph_nodes = memory_output['summary_embeddings']
            graph_context = graph_nodes.mean(dim=1, keepdim=True).expand(batch_size, seq_len, -1)
        else:
            pooled = hidden_states.mean(dim=1, keepdim=True)
            graph_context = pooled.expand(batch_size, seq_len, -1)

        signal_output = self.signal_predictor(hidden_states)
        semantic_signals = signal_output['signal_embedding']

        router_output = self.router(
            hidden_states=hidden_states,
            graph_context=graph_context,
            semantic_signals=semantic_signals
        )

        top_k_weights = router_output['top_k_weights']
        top_k_indices = router_output['top_k_indices']

        x_flat = hidden_states.reshape(-1, hidden_dim)
        g_flat = top_k_weights.reshape(-1, self.config.top_k)
        idx_flat = top_k_indices.reshape(-1, self.config.top_k)

        expert_outputs = torch.zeros_like(x_flat)

        for k in range(self.config.top_k):
            weights = g_flat[:, k].unsqueeze(-1)
            indices = idx_flat[:, k]

            for expert_idx in range(self.config.num_experts):
                mask = (indices == expert_idx)
                if mask.any():
                    x_expert = x_flat[mask]
                    expert_result = self.experts[expert_idx](x_expert, return_metadata=True)
                    expert_out = expert_result['output']
                    expert_outputs[mask] += weights[mask] * expert_out
                    self.expert_usage[self.config.expert_types[expert_idx]] += mask.sum().item()

        expert_outputs = expert_outputs.view(batch_size, seq_len, hidden_dim)
        combined = torch.cat([hidden_states, expert_outputs], dim=-1)
        output = self.output_proj(combined)
        output = hidden_states + output

        self.call_count += 1

        return {
            'output': output,
            'router_output': router_output,
            'signal_output': signal_output,
            'entropy_info': entropy_info,
            'memory_output': memory_output,
            'expert_usage': dict(self.expert_usage)
        }

    def add_to_memory(self, hidden_states: torch.Tensor):
        compressed = self.memory.compress(hidden_states)
        self.memory.add_to_memory(compressed)

 # Only clear when end of conversation
    def reset_memory(self):
        self.memory.reset_memory()

    def get_stats(self):
        total_calls = sum(self.expert_usage.values())
        expert_distribution = {}
        if total_calls > 0:
            expert_distribution = {
                name: count / total_calls * 100
                for name, count in self.expert_usage.items()
            }

        return {
            'call_count': self.call_count,
            'total_expert_calls': total_calls,
            'expert_distribution': expert_distribution,
            'memory_nodes': len(self.memory.salience_scores) if self.memory.salience_scores else 0,
            'entropy_stats': self.entropy_calc.get_stats()
        }
