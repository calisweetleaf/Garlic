import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class RoutingDecision:
    """Production routing decision metadata"""
    path: str  # 'fast', 'normal', 'slow'
    expert_indices: torch.Tensor
    gating_weights: torch.Tensor
    entropy: torch.Tensor
    is_ood: torch.Tensor
    retrieve_memory: torch.Tensor
    signal_influence: Optional[torch.Tensor] = None

class EntropyRegularizedRouter(nn.Module):
    """
    Production-ready Entropy-Regularized Router for Garlic/HSGM.
    
    Implements:
    1. Routing Entropy Regularization (from Pangu/GNNMoE)
    2. Out-of-Distribution (OOD) Detection via Entropy Spikes
    3. Dynamic Expert Selection (Top-K)
    4. Integration with Semantic Signals (Reasoning Scaffolding)
    5. Load Balancing via Batch-Wise Entropy Maximization
    """
    def __init__(
        self, 
        hidden_dim: int, 
        graph_context_dim: int, 
        num_experts: int, 
        top_k: int = 2,
        temperature: float = 1.0,
        entropy_threshold_high: float = 0.8,  # Threshold for 'slow' path / OOD
        entropy_threshold_low: float = 0.2,   # Threshold for 'fast' path
        lambda_entropy: float = 0.01,         # Regularization strength
        use_gnn: bool = False,                # Toggle GNN-based routing
        enable_entropy_coupling: bool = True, # v2.0: Entropy-Cross-Coupling
        coupling_alpha: float = 0.5           # v2.0: Coupling coefficient
    ):
        super(EntropyRegularizedRouter, self).__init__()
        self.hidden_dim = hidden_dim
        self.graph_context_dim = graph_context_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        self.entropy_threshold_high = entropy_threshold_high
        self.entropy_threshold_low = entropy_threshold_low
        self.lambda_entropy = lambda_entropy
        self.enable_entropy_coupling = enable_entropy_coupling
        self.coupling_alpha = coupling_alpha

        # Projection layers to a common dimension
        self.hidden_proj = nn.Linear(hidden_dim, hidden_dim)
        self.graph_proj = nn.Linear(graph_context_dim, hidden_dim)
        
        # Semantic Signal Influence (Reasoning Scaffolder integration)
        self.signal_influence_layer = nn.Linear(hidden_dim, hidden_dim)

        # Router MLP
        # Input is concatenated [hidden_proj, graph_proj]
        self.mlp_router = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_experts)
        )
        
        # Optional GNN components (if torch_geometric is available)
        self.use_gnn = use_gnn
        if use_gnn:
            try:
                from torch_geometric.nn import GATConv
                self.gnn = GATConv(hidden_dim * 2, hidden_dim)
                logger.info("GNN-based routing enabled.")
            except ImportError:
                logger.warning("torch_geometric not found. Falling back to MLP routing.")
                self.use_gnn = False

    def calculate_routing_entropy(self, gating_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate Shannon entropy of the routing distribution.
        H = -sum(p * log(p))
        """
        epsilon = 1e-8
        entropy = -torch.sum(gating_weights * torch.log(gating_weights + epsilon), dim=-1)
        # Normalize by log(num_experts) to keep it in [0, 1]
        return entropy / torch.log(torch.tensor(float(self.num_experts)))

    def forward(
        self, 
        hidden_states: torch.Tensor, 
        graph_context: torch.Tensor, 
        edge_index: Optional[torch.Tensor] = None,
        semantic_signals: Optional[torch.Tensor] = None,
        step_entropy: Optional[torch.Tensor] = None  # v2.0: Step Entropy from Pass 1
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for the router.
        
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            graph_context: [batch, seq_len, graph_context_dim]
            edge_index: [2, num_edges] (optional, for GNN)
            semantic_signals: [batch, seq_len, hidden_dim] (optional, from Scaffolder)
            
        Returns:
            gating_weights: [batch, seq_len, num_experts]
            expert_indices: [batch, seq_len, top_k]
            metadata: Dictionary with entropy, OOD flags, and routing path
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Feature Fusion
        # Expect graph_context to already be [batch, seq_len, graph_context_dim]
        # (Orchestrator handles shape alignment before calling this)
        h_flat = hidden_states.reshape(-1, self.hidden_dim)
        g_flat = graph_context.reshape(-1, self.graph_context_dim)
        
        h_proj = self.hidden_proj(h_flat)
        g_proj = self.graph_proj(g_flat)
        
        # Incorporate Semantic Signals if provided
        if semantic_signals is not None:
            s_flat = semantic_signals.reshape(-1, self.hidden_dim)
            signal_influence = self.signal_influence_layer(s_flat)
            h_proj = h_proj + signal_influence
            
        fused_features = torch.cat([h_proj, g_proj], dim=-1) # [batch*seq, hidden*2]
        
        # 2. Routing Logic (GNN or MLP)
        if self.use_gnn and edge_index is not None:
            # GNN requires a graph structure over the tokens
            x = F.relu(self.gnn(fused_features, edge_index))
            # Map back to expert logits via the last part of MLP
            expert_logits = self.mlp_router[4:](x) 
        else:
            expert_logits = self.mlp_router(fused_features)
            
        # 3. Gating and Entropy (v2.0: Entropy-Cross-Coupling)
        # Modulate temperature based on Step Entropy if provided
        if self.enable_entropy_coupling and step_entropy is not None:
            # High step_entropy -> higher temperature -> more exploratory routing
            # Low step_entropy -> lower temperature -> more confident routing
            step_entropy_mean = step_entropy.mean() if step_entropy.numel() > 1 else step_entropy
            temperature_modulated = self.temperature * torch.exp(self.coupling_alpha * step_entropy_mean)
        else:
            temperature_modulated = self.temperature
        
        gating_weights_all = F.softmax(expert_logits / temperature_modulated, dim=-1)
        entropy = self.calculate_routing_entropy(gating_weights_all)
        
        # 4. Top-K Selection
        top_k_weights, top_k_indices = torch.topk(gating_weights_all, self.top_k, dim=-1)
        # Re-normalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        # 5. Path Decision & OOD Detection
        # OOD if entropy is very high (router is uncertain)
        is_ood = entropy > self.entropy_threshold_high
        
        # Path classification
        # Low entropy -> Fast (System 1)
        # High entropy -> Slow (System 2 / Memory Retrieval)
        path_indices = torch.zeros_like(entropy, dtype=torch.long)
        path_indices[entropy < self.entropy_threshold_low] = 0 # Fast
        path_indices[(entropy >= self.entropy_threshold_low) & (entropy <= self.entropy_threshold_high)] = 1 # Normal
        path_indices[entropy > self.entropy_threshold_high] = 2 # Slow / OOD
        
        # 6. Reshape and Package
        gating_weights_all = gating_weights_all.view(batch_size, seq_len, self.num_experts)
        top_k_indices = top_k_indices.view(batch_size, seq_len, self.top_k)
        top_k_weights = top_k_weights.view(batch_size, seq_len, self.top_k)
        entropy = entropy.view(batch_size, seq_len)
        is_ood = is_ood.view(batch_size, seq_len)
        path_indices = path_indices.view(batch_size, seq_len)
        
        metadata = {
            "entropy": entropy,
            "is_ood": is_ood,
            "path_indices": path_indices, # 0: fast, 1: normal, 2: slow
            "retrieve_memory": path_indices == 2,
            "expert_logits": expert_logits.view(batch_size, seq_len, self.num_experts)
        }
        
        return top_k_weights, top_k_indices, metadata

    def get_loss(self, gating_weights: torch.Tensor, graph_metrics: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate routing loss:
        1. Sample-wise entropy minimization (Confidence)
        2. Batch-wise entropy maximization (Load Balancing)
        3. v2.0: Topological Routing Loss (Graph-Aware Expert Specialization)
        """
        # gating_weights can be either:
        # - [batch, seq, num_experts] (full gating weights)
        # - [batch, seq, top_k] (top-k weights from forward pass)
        # We need to handle both cases
        if gating_weights.shape[-1] == self.num_experts:
            # Full gating weights
            flat_weights = gating_weights.reshape(-1, self.num_experts)
        else:
            # Top-k weights - we can't use these for the loss, return a warning
            # In production, you should pass the full gating weights from metadata
            logger.warning("Received top-k weights instead of full gating weights. "
                          "Loss calculation may be inaccurate. Pass metadata['expert_logits'] instead.")
            # Pad to num_experts with zeros
            batch_size, seq_len, top_k = gating_weights.shape
            full_weights = torch.zeros(batch_size, seq_len, self.num_experts, device=gating_weights.device)
            # This is a hack - in production, use the full gating weights from forward pass
            flat_weights = full_weights.reshape(-1, self.num_experts)
        
        # 1. Sample-wise entropy (Minimize to increase confidence)
        sample_entropy = self.calculate_routing_entropy(flat_weights).mean()
        
        # 2. Batch-wise entropy (Maximize to ensure expert diversity)
        batch_mean_weights = flat_weights.mean(dim=0)
        batch_entropy = self.calculate_routing_entropy(batch_mean_weights.unsqueeze(0))
        
        # 3. v2.0: Topological Routing Loss
        topological_loss = torch.tensor(0.0, device=gating_weights.device)
        if graph_metrics is not None:
            # graph_metrics: [batch, seq, num_metrics] (e.g., centrality, degree, clustering)
            # We want to align expert usage with graph topology
            # Calculate per-expert average of graph metrics
            batch_size, seq_len, num_metrics = graph_metrics.shape
            graph_metrics_flat = graph_metrics.view(-1, num_metrics)  # [batch*seq, num_metrics]
            
            # Ensure flat_weights matches the batch*seq dimension
            # flat_weights: [batch*seq, num_experts]
            assert flat_weights.shape[0] == graph_metrics_flat.shape[0], \
                f"Dimension mismatch: {flat_weights.shape[0]} != {graph_metrics_flat.shape[0]}"
            
            # Weight each metric by expert assignment
            # expert_metric_dist: [num_experts, num_metrics]
            expert_metric_dist = torch.matmul(flat_weights.t(), graph_metrics_flat) / (flat_weights.sum(dim=0, keepdim=True).t() + 1e-8)
            
            # Normalize to get probability distribution over experts for each metric
            expert_metric_dist = F.softmax(expert_metric_dist, dim=0)
            
            # Target: uniform distribution (each expert should specialize in different topologies)
            uniform_dist = torch.ones_like(expert_metric_dist) / self.num_experts
            
            # KL divergence loss to encourage specialization
            topological_loss = F.kl_div(
                expert_metric_dist.log(),
                uniform_dist,
                reduction='batchmean'
            )
        
        # Total loss = SampleEntropy - BatchEntropy + TopologicalLoss
        routing_loss = sample_entropy - batch_entropy + 0.1 * topological_loss
        return self.lambda_entropy * routing_loss

def create_production_router(config: Dict[str, Any]) -> EntropyRegularizedRouter:
    """Factory function for the production router"""
    return EntropyRegularizedRouter(
        hidden_dim=config.get("hidden_dim", 768),
        graph_context_dim=config.get("graph_context_dim", 768),
        num_experts=config.get("num_experts", 8),
        top_k=config.get("top_k", 2),
        temperature=config.get("temperature", 1.0),
        entropy_threshold_high=config.get("entropy_threshold_high", 0.8),
        entropy_threshold_low=config.get("entropy_threshold_low", 0.2),
        use_gnn=config.get("use_gnn", False)
    )

if __name__ == "__main__":
    # Simple test
    hidden_dim = 384
    num_experts = 8
    router = EntropyRegularizedRouter(hidden_dim, hidden_dim, num_experts)
    
    batch_size = 2
    seq_len = 4
    h = torch.randn(batch_size, seq_len, hidden_dim)
    g = torch.randn(batch_size, seq_len, hidden_dim)
    
    weights, indices, meta = router(h, g)
    
    print(f"Weights shape: {weights.shape}")
    print(f"Indices shape: {indices.shape}")
    print(f"Entropy mean: {meta['entropy'].mean().item():.4f}")
    print(f"Path distribution: {torch.bincount(meta['path_indices'].view(-1))}")
    
    loss = router.get_loss(weights) # This is just for demonstration, usually use all weights
    print(f"Routing loss: {loss.item():.4f}")
