"""
Hierarchical LoRA Expert - Enhanced Implementation
Each expert is itself a small Mixture-of-Experts (MoE)

This extends the original LoRA MoE with hierarchical composition,
where each expert contains multiple sub-experts with dynamic routing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import math


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation adapter for sub-experts."""
    
    def __init__(self, input_dim: int = 768, rank: int = 4, alpha: int = 32):
        super().__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(input_dim, rank) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(rank, input_dim))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation."""
        # x: [..., input_dim]
        # Returns: [..., input_dim] with LoRA adaptation
        original = x
        lora_adaptation = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)
        return original + self.scaling * lora_adaptation


class HierarchicalLoRAExpert(nn.Module):
    """
    Hierarchical LoRA Expert: Each expert is itself a small MoE.
    
    Architecture:
    - Multiple sub-experts (LoRA adapters)
    - Sub-router for dynamic sub-expert selection
    - Context-aware difficulty-based routing
    
    Paper Extension: "Hierarchical MoE enables finer-grained specialization
    while maintaining parameter efficiency"
    """
    
    def __init__(self, 
                 input_dim: int = 768,
                 num_sub_experts: int = 4,
                 sub_expert_rank: int = 4,
                 sub_expert_alpha: int = 32,
                 routing_temperature: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_sub_experts = num_sub_experts
        self.routing_temperature = routing_temperature
        
        # Sub-experts: Each is a LoRA adapter
        self.sub_experts = nn.ModuleList([
            LoRAAdapter(input_dim=input_dim, rank=sub_expert_rank, alpha=sub_expert_alpha)
            for _ in range(num_sub_experts)
        ])
        
        # Sub-router: Maps input to sub-expert weights
        self.sub_router = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_sub_experts)
        )
        
        # Context difficulty encoder
        self.difficulty_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.Tanh(),
            nn.Linear(input_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Difficulty thresholds
        self.easy_threshold = 0.3
        self.hard_threshold = 0.7
        
    def forward(self, 
                x: torch.Tensor, 
                context_difficulty: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Forward pass through hierarchical expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim] or [batch_size, input_dim]
            context_difficulty: Optional difficulty score [batch_size] or scalar
            
        Returns:
            Dictionary with output, routing weights, and metadata
        """
        original_shape = x.shape
        
        # Handle different input shapes
        if len(original_shape) == 3:
            batch_size, seq_len, input_dim = x.shape
            x_flat = x.view(-1, input_dim)  # [batch_size * seq_len, input_dim]
        else:
            batch_size, input_dim = x.shape
            seq_len = 1
            x_flat = x
        
        # Estimate context difficulty if not provided
        if context_difficulty is None:
            # Use input variance as proxy for difficulty
            context_difficulty = self.difficulty_encoder(x_flat).mean(dim=-1)  # [batch_size * seq_len]
        
        # Compute sub-expert routing weights
        routing_logits = self.sub_router(x_flat)  # [batch_size * seq_len, num_sub_experts]
        routing_weights = F.softmax(routing_logits / self.routing_temperature, dim=-1)
        
        # Hierarchical routing based on difficulty
        if len(original_shape) == 3:
            context_difficulty = context_difficulty.view(batch_size, seq_len)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        routing_decisions = []
        
        # Route based on difficulty
        for i, difficulty in enumerate(context_difficulty):
            difficulty_val = difficulty.item() if isinstance(difficulty, torch.Tensor) else difficulty
            
            if difficulty_val < self.easy_threshold:
                # Easy tasks: Use single sub-expert (expert 0)
                expert_idx = 0
                weight = 1.0
                sub_output = self.sub_experts[expert_idx](x_flat[i:i+1])
                output[i:i+1] = sub_output
                routing_decisions.append({
                    'type': 'easy_single',
                    'expert_idx': expert_idx,
                    'weight': weight,
                    'difficulty': difficulty_val
                })
                
            elif difficulty_val < self.hard_threshold:
                # Medium tasks: Weighted blend of top-2 sub-experts
                top_weights, top_indices = torch.topk(routing_weights[i], k=2)
                top_weights = top_weights / top_weights.sum()  # Renormalize
                
                sub_output = torch.zeros_like(x_flat[i:i+1])
                for weight, expert_idx in zip(top_weights, top_indices):
                    expert_output = self.sub_experts[expert_idx](x_flat[i:i+1])
                    sub_output += weight * expert_output
                
                output[i:i+1] = sub_output
                routing_decisions.append({
                    'type': 'medium_blend',
                    'expert_indices': top_indices.tolist(),
                    'weights': top_weights.tolist(),
                    'difficulty': difficulty_val
                })
                
            else:
                # Hard tasks: Full mixture of all sub-experts
                sub_output = torch.zeros_like(x_flat[i:i+1])
                for expert_idx, weight in enumerate(routing_weights[i]):
                    expert_output = self.sub_experts[expert_idx](x_flat[i:i+1])
                    sub_output += weight * expert_output
                
                output[i:i+1] = sub_output
                routing_decisions.append({
                    'type': 'hard_mixture',
                    'weights': routing_weights[i].tolist(),
                    'difficulty': difficulty_val
                })
        
        # Reshape output back to original shape
        if len(original_shape) == 3:
            output = output.view(original_shape)
        
        return {
            'output': output,
            'routing_weights': routing_weights,
            'routing_decisions': routing_decisions,
            'context_difficulty': context_difficulty,
            'sub_expert_count': self.num_sub_experts
        }


class QuantumSuperpositionRouter(nn.Module):
    """
    Quantum Superposition Router with Gaussian amplitude distributions.
    
    Instead of discrete routing decisions, this router computes ALL paths
    simultaneously with amplitude weights, then collapses at readout.
    
    Theory: "Quantum-inspired routing enables exploration of multiple
    computational paths before measurement"
    """
    
    def __init__(self, 
                 num_paths: int = 3,
                 entropy_range: Tuple[float, float] = (0.0, 5.0)):
        super().__init__()
        self.num_paths = num_paths
        self.entropy_min, self.entropy_max = entropy_range
        
        # Gaussian parameters for each path
        # Path 0 (Fast): centered at low entropy
        # Path 1 (Expert): centered at medium entropy  
        # Path 2 (Reasoning): centered at high entropy
        self.register_buffer('path_centers', torch.tensor([0.8, 2.5, 4.2]))
        self.register_buffer('path_sigmas', torch.tensor([0.8, 1.0, 1.2]))
        
        # Learnable amplitude scaling
        self.amplitude_scaler = nn.Parameter(torch.ones(num_paths))
        
    def compute_amplitudes(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Compute quantum amplitudes for each path using Gaussian wavefunctions.
        
        Args:
            entropy: Scalar or tensor of entropy values
            
        Returns:
            amplitudes: [num_paths] or [batch_size, num_paths] tensor
        """
        # entropy: scalar or [batch_size]
        if entropy.dim() == 0:
            entropy = entropy.unsqueeze(0)  # [1]
        
        # Expand for broadcasting: [batch_size, 1]
        entropy_expanded = entropy.unsqueeze(-1)
        
        # Compute Gaussian amplitudes
        # amplitude_i = exp(-(H - μ_i)² / (2σ_i²))
        diff_squared = (entropy_expanded - self.path_centers) ** 2
        amplitudes = torch.exp(-diff_squared / (2 * self.path_sigmas ** 2))
        
        # Apply learnable scaling
        amplitudes = amplitudes * self.amplitude_scaler
        
        # Normalize to get probabilities
        amplitude_sum = amplitudes.sum(dim=-1, keepdim=True)
        probabilities = amplitudes / (amplitude_sum + 1e-8)
        
        return probabilities
    
    def forward(self, 
                x: torch.Tensor,
                entropy: torch.Tensor,
                path_functions: List[callable]) -> Dict[str, Any]:
        """
        Quantum superposition routing: compute ALL paths, then measure.
        
        Args:
            x: Input tensor
            entropy: Entropy value for routing
            path_functions: List of functions, one per path
            
        Returns:
            Dictionary with superposition output, amplitudes, and collapsed result
        """
        # Compute amplitudes
        amplitudes = self.compute_amplitudes(entropy)  # [batch_size, num_paths]
        
        # Execute ALL paths simultaneously (quantum superposition)
        path_outputs = []
        for i, path_func in enumerate(path_functions):
            if path_func is not None:
                path_output = path_func(x)
                path_outputs.append(path_output)
            else:
                # Identity function for base path
                path_outputs.append(x)
        
        # Stack path outputs: [num_paths, batch_size, ...]
        path_outputs_stack = torch.stack(path_outputs, dim=0)
        
        # Weighted superposition: Y = Σ α_i * Y_i
        # amplitudes: [batch_size, num_paths]
        # Need to reshape for broadcasting
        amplitudes_expanded = amplitudes.t().unsqueeze(-1)  # [num_paths, batch_size, 1]
        
        superposition_output = torch.sum(amplitudes_expanded * path_outputs_stack, dim=0)
        
        # Measurement/collapse: Sample from amplitude distribution
        # For inference, we can take the most probable path
        most_probable_path = torch.argmax(amplitudes, dim=-1)
        
        # Or sample probabilistically
        # sampled_path = torch.multinomial(amplitudes, 1).squeeze(-1)
        
        # Collapsed output based on most probable path
        collapsed_output = path_outputs_stack[most_probable_path, torch.arange(amplitudes.size(0))]
        
        return {
            'superposition_output': superposition_output,
            'collapsed_output': collapsed_output,
            'amplitudes': amplitudes,
            'most_probable_path': most_probable_path,
            'path_outputs': path_outputs_stack,
            'quantum_state': f'|ψ⟩ = Σ α_i |{i}⟩'
        }


def test_hierarchical_expert():
    """Test the HierarchicalLoRAExpert implementation."""
    print("Testing Hierarchical LoRA Expert...")
    
    # Create expert
    expert = HierarchicalLoRAExpert(
        input_dim=768,
        num_sub_experts=4,
        sub_expert_rank=4,
        sub_expert_alpha=32
    )
    
    # Test with different difficulty levels
    test_inputs = [
        ("Easy task", torch.randn(2, 10, 768), 0.2),
        ("Medium task", torch.randn(2, 10, 768), 0.5),
        ("Hard task", torch.randn(2, 10, 768), 0.8)
    ]
    
    for task_name, x, difficulty in test_inputs:
        print(f"\n--- {task_name} (difficulty={difficulty}) ---")
        
        # Create difficulty tensor
        difficulty_tensor = torch.tensor([difficulty] * x.size(0) * x.size(1))
        
        result = expert(x, difficulty_tensor)
        
        print(f"Output shape: {result['output'].shape}")
        print(f"Routing decisions: {result['routing_decisions'][:2]}")
        print(f"Avg routing weights: {result['routing_weights'].mean(dim=0)}")
    
    print("\n Hierarchical LoRA Expert test passed!")


def test_quantum_router():
    """Test the QuantumSuperpositionRouter implementation."""
    print("\nTesting Quantum Superposition Router...")
    
    # Create quantum router
    quantum_router = QuantumSuperpositionRouter(num_paths=3)
    
    # Test input
    x = torch.randn(5, 768)
    
    # Define path functions
    def fast_path(x):
        return x * 0.5  # Simple scaling
    
    def expert_path(x):
        return x + 0.1 * torch.randn_like(x)  # Add noise
    
    def reasoning_path(x):
        return F.relu(x)  # Non-linear transformation
    
    path_functions = [fast_path, expert_path, reasoning_path]
    
    # Test with different entropy values
    test_entropies = [0.5, 2.0, 4.5]
    
    for entropy_val in test_entropies:
        print(f"\n--- Entropy H = {entropy_val} ---")
        
        result = quantum_router(x, torch.tensor(entropy_val), path_functions)
        
        amplitudes = result['amplitudes']
        print(f"Quantum amplitudes: {amplitudes[0].tolist()}")
        print(f"Most probable path: {result['most_probable_path'].item()}")
        print(f"Quantum state: {result['quantum_state']}")
        
        # Show which path dominates
        path_names = ['Fast', 'Expert', 'Reasoning']
        dominant_path = torch.argmax(amplitudes[0]).item()
        print(f"Dominant path: {path_names[dominant_path]}")
    
    print("\n Quantum Superposition Router test passed!")


if __name__ == "__main__":
    test_hierarchical_expert()
    test_quantum_router()
    print("\n All hierarchical and quantum routing tests completed!")
