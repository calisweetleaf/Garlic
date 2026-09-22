"""
Hierarchical Query Processing - Theory-Aligned Implementation
Based on HSGM_paper.pdf Section 3.4 "Hierarchical Query Processing"

This component implements the hierarchical query processing mechanism that
enables multi-scale reasoning across different levels of the graph hierarchy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict, deque

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HierarchicalQueryConfig:
    """Configuration for Hierarchical Query Processing."""
    hidden_dim: int = 768
    num_levels: int = 4           # Number of hierarchical levels
    query_heads: int = 8          # Number of query attention heads
    level_dropout: float = 0.1    # Dropout between levels
    cross_level_attention: bool = True  # Enable cross-level attention
    top_k_per_level: int = 5      # Top-k nodes to attend per level
    aggregation_method: str = "weighted_sum"  # "weighted_sum", "attention", "gating"
    query_fusion_weight: float = 0.5  # Weight for query fusion
    memory_fusion_weight: float = 0.5  # Weight for memory fusion


class LevelWiseQueryEncoder(nn.Module):
    """
    Encodes queries for different hierarchical levels.
    Paper: "Query representations encoded per level with level-specific projections"
    """
    
    def __init__(self, config: HierarchicalQueryConfig):
        super().__init__()
        self.config = config
        
        # Level-specific query projections
        self.level_projections = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_levels)
        ])
        
        # Query refinement layers
        self.query_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(config.level_dropout),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            )
            for _ in range(config.num_levels)
        ])
        
    def forward(self, query_embedding: torch.Tensor) -> List[torch.Tensor]:
        """
        Encode query for each hierarchical level.
        
        Args:
            query_embedding: [hidden_dim] - Input query vector
            
        Returns:
            level_queries: List of [hidden_dim] query vectors per level
        """
        level_queries = []
        
        for i, (projection, refinement) in enumerate(
            zip(self.level_projections, self.query_refinement)
        ):
            # Project query for this level
            level_query = projection(query_embedding)
            
            # Refine with non-linear transformation
            level_query = refinement(level_query)
            
            level_queries.append(level_query)
        
        return level_queries


class CrossLevelAttention(nn.Module):
    """
    Computes attention between different hierarchical levels.
    Paper: "Cross-level attention enables information flow across hierarchy"
    """
    
    def __init__(self, config: HierarchicalQueryConfig):
        super().__init__()
        self.config = config
        
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.attention_heads = config.query_heads
        self.head_dim = config.hidden_dim // config.query_heads
        
        assert config.hidden_dim % config.query_heads == 0
        
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.level_dropout)
        
    def forward(self,
                query_level: int,
                level_representations: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute cross-level attention for a specific query level.
        
        Args:
            query_level: Target level for attention
            level_representations: List of [num_nodes, hidden_dim] per level
            
        Returns:
            cross_level_context: [hidden_dim] - Aggregated context
        """
        if not self.config.cross_level_attention:
            # Return query level representation only
            return torch.mean(level_representations[query_level], dim=0)
        
        # Get query representation
        query_repr = level_representations[query_level]  # [num_nodes_q, hidden_dim]
        
        # Collect all other levels as keys/values
        all_keys = []
        all_values = []
        level_indices = []
        
        for level_idx, level_repr in enumerate(level_representations):
            if level_idx == query_level:
                continue
            all_keys.append(level_repr)
            all_values.append(level_repr)
            level_indices.extend([level_idx] * level_repr.shape[0])
        
        if not all_keys:
            return torch.mean(query_repr, dim=0)
        
        # Concatenate all levels
        keys = torch.cat(all_keys, dim=0)      # [total_nodes_k, hidden_dim]
        values = torch.cat(all_values, dim=0)  # [total_nodes_v, hidden_dim]
        
        # Project for multi-head attention
        batch_size = query_repr.shape[0]
        
        Q = self.query_proj(query_repr).view(batch_size, self.attention_heads, self.head_dim)
        K = self.key_proj(keys).view(keys.shape[0], self.attention_heads, self.head_dim)
        V = self.value_proj(values).view(values.shape[0], self.attention_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 1)  # [heads, batch_size, head_dim]
        K = K.transpose(0, 1)  # [heads, total_nodes_k, head_dim]
        V = V.transpose(0, 1)  # [heads, total_nodes_v, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [heads, batch_size, head_dim]
        context = context.transpose(0, 1).contiguous()  # [batch_size, heads, head_dim]
        context = context.view(batch_size, -1)  # [batch_size, hidden_dim]
        
        # Output projection and pooling
        context = self.output_proj(context)
        context = torch.mean(context, dim=0)  # [hidden_dim]
        
        return context


class TopKNodeSelector(nn.Module):
    """
    Selects top-k most relevant nodes per level for query processing.
    Paper: "Top-k selection reduces computational complexity"
    """
    
    def __init__(self, config: HierarchicalQueryConfig):
        super().__init__()
        self.config = config
        
    def forward(self,
                query: torch.Tensor,
                node_embeddings: torch.Tensor,
                k: int = None) -> Tuple[torch.Tensor, List[int]]:
        """
        Select top-k nodes most relevant to query.
        
        Args:
            query: [hidden_dim] - Query vector
            node_embeddings: [num_nodes, hidden_dim] - Node representations
            k: Number of nodes to select (default: config.top_k_per_level)
            
        Returns:
            selected_embeddings: [k, hidden_dim]
            selected_indices: List of selected node indices
        """
        if k is None:
            k = self.config.top_k_per_level
        
        num_nodes = node_embeddings.shape[0]
        k = min(k, num_nodes)
        
        # Compute similarity scores
        query_normalized = F.normalize(query.unsqueeze(0), p=2, dim=-1)
        node_normalized = F.normalize(node_embeddings, p=2, dim=-1)
        
        scores = torch.matmul(query_normalized, node_normalized.t()).squeeze(0)
        
        # Select top-k
        top_scores, top_indices = torch.topk(scores, k)
        
        # Get selected embeddings
        selected_embeddings = node_embeddings[top_indices]
        
        return selected_embeddings, top_indices.tolist()


class QueryLevelAggregator(nn.Module):
    """
    Aggregates information across hierarchical levels for final answer.
    Paper: "Level aggregation combines multi-scale information"
    """
    
    def __init__(self, config: HierarchicalQueryConfig):
        super().__init__()
        self.config = config
        
        self.level_weights = nn.Parameter(torch.ones(config.num_levels))
        
        if config.aggregation_method == "attention":
            self.aggregation_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.query_heads,
                dropout=config.level_dropout,
                batch_first=True
            )
        elif config.aggregation_method == "gating":
            self.gate_network = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(config.hidden_dim // 2, 1),
                nn.Sigmoid()
            )
            
    def forward(self, 
                level_outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        Aggregate outputs from different hierarchical levels.
        
        Args:
            level_outputs: List of [hidden_dim] vectors per level
            
        Returns:
            aggregated_output: [hidden_dim] - Final aggregated vector
        """
        if not level_outputs:
            return torch.zeros(self.config.hidden_dim)
        
        if self.config.aggregation_method == "weighted_sum":
            # Weighted sum with learned weights
            weights = F.softmax(self.level_weights[:len(level_outputs)], dim=0)
            
            aggregated = torch.zeros_like(level_outputs[0])
            for weight, output in zip(weights, level_outputs):
                aggregated += weight * output
                
        elif self.config.aggregation_method == "attention":
            # Stack level outputs for attention
            level_tensor = torch.stack(level_outputs).unsqueeze(0)  # [1, num_levels, hidden_dim]
            
            # Self-attention among levels
            attn_output, _ = self.aggregation_attention(
                level_tensor, level_tensor, level_tensor
            )
            
            # Pool across levels
            aggregated = torch.mean(attn_output.squeeze(0), dim=0)
            
        elif self.config.aggregation_method == "gating":
            # Gating mechanism for selective aggregation
            stacked_outputs = torch.stack(level_outputs)  # [num_levels, hidden_dim]
            
            # Compute gates
            gates = self.gate_network(stacked_outputs)  # [num_levels, 1]
            gates = F.softmax(gates, dim=0)  # Normalize across levels
            
            # Apply gates
            aggregated = torch.sum(gates * stacked_outputs, dim=0)
            
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.aggregation_method}")
        
        return aggregated


class HierarchicalQueryProcessor(nn.Module):
    """
    Complete Hierarchical Query Processing pipeline.
    Paper: "Algorithm 3: Hierarchical Query Processing"
    """
    
    def __init__(self, config: HierarchicalQueryConfig):
        super().__init__()
        self.config = config
        
        self.query_encoder = LevelWiseQueryEncoder(config)
        self.cross_level_attention = CrossLevelAttention(config)
        self.node_selector = TopKNodeSelector(config)
        self.level_aggregator = QueryLevelAggregator(config)
        
        # Query fusion components
        self.query_fusion_gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        self.memory_fusion_gate = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        logger.info(f"Initialized Hierarchical Query Processor with {config}")
        
    def forward(self,
                query_embedding: torch.Tensor,
                hierarchical_representations: List[torch.Tensor],
                memory_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process query across hierarchical levels.
        
        Args:
            query_embedding: [hidden_dim] - Input query vector
            hierarchical_representations: List of [num_nodes, hidden_dim] per level
            memory_context: [hidden_dim] - Optional memory context
            
        Returns:
            Dictionary containing:
                - final_answer: [hidden_dim] - Final query answer
                - level_answers: List of level-specific answers
                - cross_level_context: Cross-level aggregated context
                - attention_weights: Attention weights per level
                - selected_nodes_per_level: Selected node indices per level
        """
        # Step 1: Encode query for each level
        level_queries = self.query_encoder(query_embedding)
        
        level_answers = []
        selected_nodes_per_level = []
        cross_level_contexts = []
        
        # Step 2: Process each level
        for level_idx, (level_query, level_repr) in enumerate(
            zip(level_queries, hierarchical_representations)
        ):
            # Select top-k nodes for this level
            selected_nodes, selected_indices = self.node_selector(
                level_query, level_repr
            )
            selected_nodes_per_level.append(selected_indices)
            
            # Compute cross-level context
            cross_level_context = self.cross_level_attention(
                level_idx, hierarchical_representations
            )
            cross_level_contexts.append(cross_level_context)
            
            # Fuse query with cross-level context
            fused_query = self._fuse_query_context(
                level_query, cross_level_context
            )
            
            # Compute level answer (attention over selected nodes)
            level_answer = self._compute_level_answer(
                fused_query, selected_nodes
            )
            level_answers.append(level_answer)
        
        # Step 3: Aggregate answers across levels
        final_answer = self.level_aggregator(level_answers)
        
        # Step 4: Integrate memory context if provided
        if memory_context is not None:
            final_answer = self._integrate_memory(
                final_answer, memory_context
            )
        
        # Compute attention weights for analysis
        level_weights = F.softmax(self.level_aggregator.level_weights[:len(level_answers)], dim=0)
        
        return {
            'final_answer': final_answer,
            'level_answers': level_answers,
            'cross_level_context': cross_level_contexts,
            'attention_weights': level_weights,
            'selected_nodes_per_level': selected_nodes_per_level
        }
    
    def _fuse_query_context(self, 
                           query: torch.Tensor, 
                           context: torch.Tensor) -> torch.Tensor:
        """Fuse query with cross-level context."""
        # Concatenate and gate
        concatenated = torch.cat([query, context], dim=-1)
        gate = torch.sigmoid(self.query_fusion_gate(concatenated))
        
        fused = gate * query + (1 - gate) * context
        return fused
    
    def _compute_level_answer(self,
                             query: torch.Tensor,
                             selected_nodes: torch.Tensor) -> torch.Tensor:
        """Compute answer for a specific level."""
        # Attention over selected nodes
        query_expanded = query.unsqueeze(0)  # [1, hidden_dim]
        
        # Compute attention scores
        scores = torch.matmul(query_expanded, selected_nodes.t())  # [1, num_selected]
        weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of selected nodes
        level_answer = torch.matmul(weights, selected_nodes).squeeze(0)  # [hidden_dim]
        
        return level_answer
    
    def _integrate_memory(self,
                         answer: torch.Tensor,
                         memory: torch.Tensor) -> torch.Tensor:
        """Integrate memory context into final answer."""
        concatenated = torch.cat([answer, memory], dim=-1)
        gate = torch.sigmoid(self.memory_fusion_gate(concatenated))
        
        integrated = gate * answer + (1 - gate) * memory
        return integrated


class IncrementalQueryProcessor(nn.Module):
    """
    Supports incremental query processing for streaming data.
    Paper: "Incremental processing enables real-time updates"
    """
    
    def __init__(self, config: HierarchicalQueryConfig):
        super().__init__()
        self.config = config
        self.main_processor = HierarchicalQueryProcessor(config)
        
        # State tracking for incremental processing
        self.previous_summaries = {}
        self.update_buffer = deque(maxlen=100)
        
    def process_incremental(self,
                           query_embedding: torch.Tensor,
                           new_representations: Dict[int, torch.Tensor],
                           memory_context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process query incrementally with new representations.
        
        Args:
            query_embedding: [hidden_dim] - Query vector
            new_representations: Dict[level -> [num_new_nodes, hidden_dim]]
            memory_context: [hidden_dim] - Memory context
            
        Returns:
            Same as HierarchicalQueryProcessor.forward()
        """
        # Update cached representations
        for level, new_repr in new_representations.items():
            if level in self.previous_summaries:
                # Concatenate with existing representations
                existing = self.previous_summaries[level]
                updated = torch.cat([existing, new_repr], dim=0)
                self.previous_summaries[level] = updated
            else:
                self.previous_summaries[level] = new_repr
        
        # Convert to list format
        hierarchical_reprs = []
        for level in range(self.config.num_levels):
            if level in self.previous_summaries:
                hierarchical_reprs.append(self.previous_summaries[level])
            else:
                # Create empty representation
                hierarchical_reprs.append(
                    torch.zeros(1, self.config.hidden_dim)
                )
        
        # Process with main processor
        return self.main_processor(
            query_embedding, hierarchical_reprs, memory_context
        )
    
    def reset_state(self):
        """Reset incremental processing state."""
        self.previous_summaries.clear()
        self.update_buffer.clear()


def test_hierarchical_query_processor():
    """Test the Hierarchical Query Processor implementation."""
    
    print("Testing Hierarchical Query Processor...")
    
    # Create test configuration
    config = HierarchicalQueryConfig(
        hidden_dim=128,
        num_levels=3,
        query_heads=4,
        top_k_per_level=3,
        aggregation_method="attention"
    )
    
    # Initialize processor
    processor = HierarchicalQueryProcessor(config)
    
    # Create test data
    hidden_dim = 128
    num_levels = 3
    
    # Query embedding
    query_embedding = torch.randn(hidden_dim)
    
    # Hierarchical representations (different number of nodes per level)
    hierarchical_reprs = [
        torch.randn(8, hidden_dim),   # Level 0: 8 nodes
        torch.randn(4, hidden_dim),   # Level 1: 4 nodes  
        torch.randn(2, hidden_dim)    # Level 2: 2 nodes
    ]
    
    # Memory context
    memory_context = torch.randn(hidden_dim)
    
    # Process query
    with torch.no_grad():
        result = processor(query_embedding, hierarchical_reprs, memory_context)
    
    # Verify outputs
    assert 'final_answer' in result
    assert 'level_answers' in result
    assert 'cross_level_context' in result
    assert 'attention_weights' in result
    assert 'selected_nodes_per_level' in result
    
    final_answer = result['final_answer']
    level_answers = result['level_answers']
    attention_weights = result['attention_weights']
    
    print(f"✓ Final answer shape: {final_answer.shape}")
    print(f"✓ Number of level answers: {len(level_answers)}")
    print(f"✓ Level answer shapes: {[la.shape for la in level_answers]}")
    print(f"✓ Attention weights shape: {attention_weights.shape}")
    print(f"✓ Selected nodes per level: {result['selected_nodes_per_level']}")
    
    # Test incremental processor
    incremental_processor = IncrementalQueryProcessor(config)
    
    # Simulate incremental updates
    new_representations = {
        0: torch.randn(2, hidden_dim),  # Add 2 nodes to level 0
        1: torch.randn(1, hidden_dim)   # Add 1 node to level 1
    }
    
    with torch.no_grad():
        incremental_result = incremental_processor.process_incremental(
            query_embedding, new_representations, memory_context
        )
    
    print(f"✓ Incremental final answer shape: {incremental_result['final_answer'].shape}")
    
    # Test different aggregation methods
    for method in ["weighted_sum", "gating"]:
        config_agg = HierarchicalQueryConfig(
            hidden_dim=128,
            num_levels=3,
            aggregation_method=method
        )
        processor_agg = HierarchicalQueryProcessor(config_agg)
        
        with torch.no_grad():
            result_agg = processor_agg(query_embedding, hierarchical_reprs)
        
        print(f"✓ Aggregation method '{method}' works")
    
    print("✅ Hierarchical Query Processor test passed!")
    
    return result


if __name__ == "__main__":
    test_result = test_hierarchical_query_processor()
    print("\nHierarchical Query Processor implementation complete!")