"""
GARLIC v3.0: LoRA Experts and Dynamic Rank Adapters
Integrated from LoRA-MoE system with production enhancements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import math
import logging

logger = logging.getLogger(__name__)

class DynamicRankLoRAAdapter(nn.Module):
    """
    LoRA adapter with dynamic rank adaptation based on input complexity.
    """
    def __init__(self, input_dim: int, rank_base: int, rank_max: int, 
                 alpha: float, dropout: float = 0.05):
        super().__init__()
        self.input_dim = input_dim
        self.rank_base = rank_base
        self.rank_max = rank_max
        self.alpha = alpha

        # Maximum rank matrices
        self.lora_A = nn.Parameter(torch.randn(rank_max, input_dim) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(input_dim, rank_max))

        # Complexity estimator
        self.complexity_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.LayerNorm(input_dim // 4),
            nn.GELU(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def compute_effective_rank(self, x: torch.Tensor) -> int:
        """Compute effective rank based on input complexity."""
        # Use mean across batch/seq for complexity
        complexity = self.complexity_encoder(x.mean(dim=0, keepdim=True)).item()
        effective_rank = int(self.rank_base + complexity * (self.rank_max - self.rank_base))
        return min(max(effective_rank, self.rank_base), self.rank_max)

    def forward(self, x: torch.Tensor, return_metadata: bool = False):
        original = x
        effective_rank = self.compute_effective_rank(x)

        A_active = self.lora_A[:effective_rank, :]
        B_active = self.lora_B[:, :effective_rank]

        scaling = self.alpha / effective_rank
        
        # Standard LoRA: (x @ A.T) @ B.T
        lora_out = (x @ A_active.T) @ B_active.T
        lora_out = scaling * self.dropout(lora_out)

        output = original + lora_out

        if return_metadata:
            complexity = self.complexity_encoder(x.mean(dim=0, keepdim=True)).item()
            return {
                'output': output,
                'complexity': complexity,
                'effective_rank': effective_rank
            }
        return output

class HierarchicalLoRAExpert(nn.Module):
    """
    Hierarchical expert with sub-expert routing based on difficulty.
    """
    def __init__(self, input_dim: int, num_sub_experts: int = 4,
                 rank_base: int = 8, rank_max: int = 32,
                 alpha: float = 16.0, expert_type: str = "general"):
        super().__init__()
        self.input_dim = input_dim
        self.num_sub_experts = num_sub_experts
        self.expert_type = expert_type

        self.main_adapter = DynamicRankLoRAAdapter(input_dim, rank_base, rank_max, alpha)

        self.sub_experts = nn.ModuleList([
            DynamicRankLoRAAdapter(input_dim, rank_base // 2, rank_max // 2, alpha / 2)
            for _ in range(num_sub_experts)
        ])

        self.sub_router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LayerNorm(input_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_sub_experts)
        )

        self.difficulty_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )

        self.easy_threshold = 0.3
        self.hard_threshold = 0.7

    def forward(self, x: torch.Tensor, return_metadata: bool = False):
        # Handle both [batch, seq, dim] and [N, dim]
        original_shape = x.shape
        if len(original_shape) == 3:
            batch_size, seq_len, input_dim = x.shape
            x_flat = x.reshape(-1, input_dim)
        else:
            x_flat = x
            input_dim = x.shape[-1]

        main_output = self.main_adapter(x_flat)
        difficulty = self.difficulty_encoder(x_flat).squeeze(-1)
        sub_logits = self.sub_router(x_flat)
        sub_weights = F.softmax(sub_logits, dim=-1)

        output = torch.zeros_like(x_flat)
        
        # Vectorized routing for efficiency
        # 1. Easy path: just main adapter
        easy_mask = difficulty < self.easy_threshold
        if easy_mask.any():
            output[easy_mask] = main_output[easy_mask]
            
        # 2. Medium path: main + top-1 sub-expert
        medium_mask = (difficulty >= self.easy_threshold) & (difficulty < self.hard_threshold)
        if medium_mask.any():
            x_medium = x_flat[medium_mask]
            top_sub_indices = sub_weights[medium_mask].argmax(dim=-1)
            
            medium_out = torch.zeros_like(x_medium)
            for i in range(self.num_sub_experts):
                expert_mask = (top_sub_indices == i)
                if expert_mask.any():
                    medium_out[expert_mask] = self.sub_experts[i](x_medium[expert_mask])
            
            output[medium_mask] = main_output[medium_mask] + 0.3 * medium_out
            
        # 3. Hard path: main + weighted sum of all sub-experts
        hard_mask = difficulty >= self.hard_threshold
        if hard_mask.any():
            x_hard = x_flat[hard_mask]
            weights_hard = sub_weights[hard_mask]
            
            hard_out = torch.zeros_like(x_hard)
            for i in range(self.num_sub_experts):
                hard_out += weights_hard[:, i:i+1] * self.sub_experts[i](x_hard)
                
            output[hard_mask] = main_output[hard_mask] + 0.5 * hard_out

        if len(original_shape) == 3:
            output = output.reshape(original_shape)

        if return_metadata:
            return {
                'output': output,
                'difficulty': difficulty.mean().item(),
                'expert_type': self.expert_type
            }
        return output
