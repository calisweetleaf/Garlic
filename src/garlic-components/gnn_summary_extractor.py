"""
GNN Summary Node Extraction - Theory-Aligned Implementation
Based on HSGM_paper.pdf Section 3.3 "Summary Node Extraction"

This component implements the summary node extraction mechanism that identifies
and extracts representative nodes from local graphs for hierarchical processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SummaryExtractionConfig:
    """Configuration for Summary Node Extraction."""
    hidden_dim: int = 768
    summary_ratio: float = 0.25  # Ratio of nodes to keep as summaries
    min_summary_nodes: int = 2    # Minimum number of summary nodes
    max_summary_nodes: int = 16   # Maximum number of summary nodes
    centrality_threshold: float = 0.6  # Threshold for centrality-based selection
    diversity_weight: float = 0.3  # Weight for diversity in selection
    importance_weight: float = 0.7  # Weight for importance in selection
    num_heads: int = 8        # Number of attention heads
    dropout: float = 0.1      # Dropout rate


class CentralityCalculator(nn.Module):
    """
    Calculates node centrality measures for importance scoring.
    Paper: "Node importance computed via eigenvector centrality"
    """
    
    def __init__(self, config: SummaryExtractionConfig):
        super().__init__()
        self.config = config
        
    def forward(self, 
                node_embeddings: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Calculate eigenvector centrality for nodes.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
            
        Returns:
            centralities: [num_nodes] - Eigenvector centrality scores
        """
        num_nodes = node_embeddings.shape[0]
        device = node_embeddings.device

        # Handle dense (soft-mode) vs sparse (hard-mode) adjacency
        is_dense = edge_index.dim() == 2 and edge_index.shape[0] == edge_index.shape[1]

        if is_dense:
            # edge_weights IS the adjacency matrix already [N, N]
            adj_matrix = edge_weights
        else:
            # Build adjacency matrix from sparse COO
            adj_matrix = torch.zeros((num_nodes, num_nodes), device=device)
            src, dst = edge_index[0], edge_index[1]
            adj_matrix[src, dst] = edge_weights
        
        # Power iteration for eigenvector centrality
        centralities = torch.ones(num_nodes, device=device) / num_nodes
        
        for iteration in range(50):  # Max iterations
            # Multiply by adjacency matrix
            new_centralities = torch.matmul(adj_matrix, centralities)
            
            # Normalize
            norm = torch.norm(new_centralities)
            if norm > 0:
                new_centralities = new_centralities / norm
            
            # Check convergence
            if torch.allclose(centralities, new_centralities, rtol=1e-6):
                break
                
            centralities = new_centralities

        # Detach from computation graph: centrality is a non-differentiable
        # scoring signal and should not propagate gradients through the
        # power-iteration loop.
        return centralities.detach()


class SemanticDiversityCalculator(nn.Module):
    """
    Calculates semantic diversity scores for nodes.
    Paper: "Diversity measured via repulsion between embeddings"
    """
    
    def __init__(self, config: SummaryExtractionConfig):
        super().__init__()
        self.config = config
        
    def forward(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate diversity scores based on embedding distances.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            
        Returns:
            diversity_scores: [num_nodes] - Diversity scores (higher = more diverse)
        """
        num_nodes = node_embeddings.shape[0]
        
        # Normalize embeddings
        normalized_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
        
        # Compute pairwise similarities
        similarities = torch.matmul(normalized_embeddings, normalized_embeddings.t())
        
        # Diversity = negative average similarity to other nodes
        # (closer to -1 means more diverse)
        diversity_scores = -torch.sum(similarities, dim=1) / (num_nodes - 1)
        
        # Normalize to [0, 1]
        diversity_scores = (diversity_scores + 1) / 2
        
        return diversity_scores


class SummaryNodeSelector(nn.Module):
    """
    Selects summary nodes based on importance and diversity.
    Paper: "Algorithm 2: Summary Node Selection"
    """

    def __init__(self, config: SummaryExtractionConfig, soft_mode: bool = False, gumbel_tau: float = 1.0):
        super().__init__()
        self.config = config
        self.soft_mode = soft_mode
        self.gumbel_tau = gumbel_tau

    def forward(self,
                node_embeddings: torch.Tensor,
                centralities: torch.Tensor,
                diversity_scores: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Select summary nodes using greedy selection (hard) or
        Gumbel-softmax weighted sum (soft).

        Args:
            node_embeddings: [num_nodes, hidden_dim]
            centralities: [num_nodes] - Centrality scores
            diversity_scores: [num_nodes] - Diversity scores

        Returns:
            summary_embeddings: [num_summary, hidden_dim]  (hard) or [num_summary, hidden_dim] (soft)
            summary_indices: List of selected node indices (hard) or empty list (soft)
        """
        num_nodes = node_embeddings.shape[0]

        # Calculate number of summary nodes
        num_summary = max(
            self.config.min_summary_nodes,
            min(
                int(num_nodes * self.config.summary_ratio),
                self.config.max_summary_nodes
            )
        )

        # Combined scoring function
        importance_scores = centralities
        combined_scores = (
            self.config.importance_weight * importance_scores +
            self.config.diversity_weight * diversity_scores
        )

        if self.soft_mode:
            # Differentiable Gumbel-softmax weighted selection.
            # Produce num_summary soft selections via repeated Gumbel-softmax
            # draws, each yielding a weighted sum over node embeddings.
            logits = combined_scores.unsqueeze(0).expand(num_summary, -1)  # [num_summary, num_nodes]
            soft_weights = F.gumbel_softmax(logits, tau=self.gumbel_tau, hard=False, dim=-1)  # [num_summary, num_nodes]
            summary_embeddings = torch.matmul(soft_weights, node_embeddings)  # [num_summary, hidden_dim]
            # No discrete indices in soft mode
            return summary_embeddings, []

        # Hard mode (default, for inference): greedy selection with diversity constraint
        selected_indices = []
        remaining_indices = list(range(num_nodes))

        for _ in range(num_summary):
            if not remaining_indices:
                break

            # Score remaining nodes
            if not selected_indices:
                # First selection: use combined score
                scores = combined_scores[remaining_indices]
            else:
                # Subsequent selections: balance importance and diversity
                scores = torch.zeros(len(remaining_indices), device=node_embeddings.device)

                for i, idx in enumerate(remaining_indices):
                    # Distance to already selected nodes
                    if selected_indices:
                        selected_embeddings = node_embeddings[selected_indices]
                        candidate_embedding = node_embeddings[idx].unsqueeze(0)

                        # Calculate minimum distance to selected nodes
                        distances = torch.cdist(candidate_embedding, selected_embeddings)
                        min_distance = torch.min(distances)

                        # Diversity-aware score
                        importance_term = importance_scores[idx]
                        diversity_term = min_distance / torch.norm(candidate_embedding)

                        scores[i] = (
                            self.config.importance_weight * importance_term +
                            self.config.diversity_weight * diversity_term
                        )

            # Select best node
            best_idx = torch.argmax(scores).item()
            selected_node_idx = remaining_indices[best_idx]
            selected_indices.append(selected_node_idx)
            remaining_indices.pop(best_idx)

        # Extract summary embeddings
        summary_embeddings = node_embeddings[selected_indices]

        return summary_embeddings, selected_indices


class SummaryFusionEncoder(nn.Module):
    """
    Encodes and fuses summary node representations.
    Paper: "Summary representations fused via attention pooling"
    """
    
    def __init__(self, config: SummaryExtractionConfig):
        super().__init__()
        self.config = config
        
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True
        )
        
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                summary_embeddings: torch.Tensor,
                original_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Fuse summary representations with attention.
        
        Args:
            summary_embeddings: [num_summary, hidden_dim]
            original_embeddings: [num_nodes, hidden_dim]
            
        Returns:
            fused_summary: [hidden_dim] - Single summary vector
        """
        # Add batch dimension
        summary_emb = summary_embeddings.unsqueeze(0)  # [1, num_summary, hidden_dim]
        
        # Self-attention among summaries
        attn_output, _ = self.attention(
            summary_emb, summary_emb, summary_emb
        )
        
        # Remove batch dimension and pool
        attn_output = attn_output.squeeze(0)  # [num_summary, hidden_dim]
        
        # Global average pooling
        fused_summary = torch.mean(attn_output, dim=0)  # [hidden_dim]
        
        # Apply layer norm and dropout
        fused_summary = self.layer_norm(fused_summary)
        fused_summary = self.dropout(fused_summary)
        
        return fused_summary


class HierarchicalSummaryAggregator(nn.Module):
    """
    Aggregates summaries across hierarchical levels.
    Paper: "Hierarchical aggregation via level-wise pooling"
    """
    
    def __init__(self, config: SummaryExtractionConfig):
        super().__init__()
        self.config = config
        
        self.level_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(4)  # Support up to 4 hierarchical levels
        ])
        
        self.fusion_gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
    def forward(self, 
                level_summaries: List[torch.Tensor],
                target_level: int = 0) -> torch.Tensor:
        """
        Aggregate summaries across hierarchical levels.
        
        Args:
            level_summaries: List of summary vectors for each level
            target_level: Target level for aggregation
            
        Returns:
            aggregated_summary: [hidden_dim]
        """
        if not level_summaries:
            # Infer device from module parameters; fall back to CPU
            device = next(self.parameters()).device
            return torch.zeros(self.config.hidden_dim, device=device)
        
        # Project each level summary
        projected_summaries = []
        for i, summary in enumerate(level_summaries):
            if i < len(self.level_projections):
                projected = self.level_projections[i](summary)
                projected_summaries.append(projected)
            else:
                projected_summaries.append(summary)
        
        # Weighted aggregation based on target level
        device = projected_summaries[0].device
        weights = torch.softmax(
            torch.tensor(
                [1.0 / (abs(i - target_level) + 1) for i in range(len(projected_summaries))],
                device=device,
            ),
            dim=0,
        )
        
        # Weighted sum
        aggregated = torch.zeros_like(projected_summaries[0])
        for weight, summary in zip(weights, projected_summaries):
            aggregated += weight * summary
        
        return aggregated


class GNNSummaryExtractor(nn.Module):
    """
    Complete GNN Summary Node Extraction pipeline.
    Paper: "Section 3.3 Summary Node Extraction"
    """
    
    def __init__(self, config: SummaryExtractionConfig, soft_mode: bool = False):
        super().__init__()
        self.config = config
        self.soft_mode = soft_mode

        self.centrality_calculator = CentralityCalculator(config)
        self.diversity_calculator = SemanticDiversityCalculator(config)
        self.node_selector = SummaryNodeSelector(config, soft_mode=soft_mode)
        self.fusion_encoder = SummaryFusionEncoder(config)
        self.hierarchical_aggregator = HierarchicalSummaryAggregator(config)

        logger.info(f"Initialized GNN Summary Extractor (soft_mode={soft_mode})")
        
    def forward(self,
                node_embeddings: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weights: torch.Tensor,
                level: int = 0) -> Dict[str, torch.Tensor]:
        """
        Extract summary nodes from graph.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
            level: Hierarchical level (for multi-level extraction)
            
        Returns:
            Dictionary containing:
                - summary_embeddings: [num_summary, hidden_dim]
                - summary_indices: List of selected node indices
                - fused_summary: [hidden_dim] - Global summary vector
                - centrality_scores: [num_nodes]
                - diversity_scores: [num_nodes]
        """
        # Step 1: Calculate centrality scores
        centrality_scores = self.centrality_calculator(
            node_embeddings, edge_index, edge_weights
        )
        
        # Step 2: Calculate diversity scores
        diversity_scores = self.diversity_calculator(node_embeddings)
        
        # Step 3: Select summary nodes
        summary_embeddings, summary_indices = self.node_selector(
            node_embeddings, centrality_scores, diversity_scores
        )
        
        # Step 4: Fuse summary representations
        fused_summary = self.fusion_encoder(summary_embeddings, node_embeddings)
        
        logger.debug(f"Extracted {len(summary_indices)} summary nodes from {node_embeddings.shape[0]} total nodes")
        
        return {
            'summary_embeddings': summary_embeddings,
            'summary_indices': summary_indices,
            'fused_summary': fused_summary,
            'centrality_scores': centrality_scores,
            'diversity_scores': diversity_scores
        }


def test_gnn_summary_extractor():
    """Test the GNN Summary Extractor implementation."""
    
    print("Testing GNN Summary Extractor...")
    
    # Create test configuration
    config = SummaryExtractionConfig(
        hidden_dim=128,
        summary_ratio=0.3,
        min_summary_nodes=2,
        max_summary_nodes=8,
        centrality_threshold=0.5,
        diversity_weight=0.3,
        importance_weight=0.7
    )
    
    # Initialize extractor
    extractor = GNNSummaryExtractor(config)
    
    # Create test graph
    num_nodes = 16
    hidden_dim = 128
    
    node_embeddings = torch.randn(num_nodes, hidden_dim)
    
    # Create random edges
    num_edges = 40
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_weights = torch.rand(num_edges)
    
    # Extract summaries
    with torch.no_grad():
        result = extractor(node_embeddings, edge_index, edge_weights)
    
    # Verify outputs
    assert 'summary_embeddings' in result
    assert 'summary_indices' in result
    assert 'fused_summary' in result
    assert 'centrality_scores' in result
    assert 'diversity_scores' in result
    
    summary_embeddings = result['summary_embeddings']
    summary_indices = result['summary_indices']
    fused_summary = result['fused_summary']
    
    print(f"✓ Extracted {len(summary_indices)} summary nodes")
    print(f"✓ Summary embeddings shape: {summary_embeddings.shape}")
    print(f"✓ Fused summary shape: {fused_summary.shape}")
    print(f"✓ Centrality scores shape: {result['centrality_scores'].shape}")
    print(f"✓ Diversity scores shape: {result['diversity_scores'].shape}")
    
    # Test hierarchical aggregator
    aggregator = HierarchicalSummaryAggregator(config)
    
    # Create multiple level summaries
    level_summaries = [
        torch.randn(hidden_dim),
        torch.randn(hidden_dim),
        torch.randn(hidden_dim)
    ]
    
    with torch.no_grad():
        aggregated = aggregator(level_summaries, target_level=1)
    
    print(f"✓ Hierarchical aggregated shape: {aggregated.shape}")
    
    print("✅ GNN Summary Extractor test passed!")
    
    return result


if __name__ == "__main__":
    test_result = test_gnn_summary_extractor()
    print("\nGNN Summary Extractor implementation complete!")