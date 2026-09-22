"""
HSGM Local Graph Builder - Theory-Aligned Implementation
Based on HSGM_paper.pdf Section 3.2 "Local Graph Construction"

This component implements the local graph construction pipeline that builds
segment-level graphs from input sequences for hierarchical processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class LocalGraphConfig:
    """Configuration for Local Graph Builder following paper specifications."""
    segment_size: int = 32  # Size of each segment for local processing
    overlap_size: int = 8   # Overlap between segments
    similarity_threshold: float = 0.7  # Threshold for edge creation
    max_neighbors: int = 8  # Maximum neighbors per node
    hidden_dim: int = 768   # Dimension of hidden representations
    num_heads: int = 8      # Number of attention heads for relation encoding
    dropout: float = 0.1    # Dropout rate


class SegmentExtractor(nn.Module):
    """
    Extracts overlapping segments from input sequences.
    Paper: "Segments are extracted with stride S and overlap O"
    """
    
    def __init__(self, config: LocalGraphConfig):
        super().__init__()
        self.config = config
        self.segment_size = config.segment_size
        self.overlap_size = config.overlap_size
        self.stride = config.segment_size - config.overlap_size
        
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Extract segments from hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            
        Returns:
            segments: [num_segments, segment_size, hidden_dim]
            segment_info: List of segment metadata
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        segments = []
        segment_info = []
        
        for batch_idx in range(batch_size):
            batch_segments = []
            start_pos = 0
            segment_idx = 0
            
            while start_pos < seq_len:
                end_pos = min(start_pos + self.segment_size, seq_len)
                
                # Extract segment
                segment = hidden_states[batch_idx, start_pos:end_pos, :]
                
                # Pad if necessary
                if segment.shape[0] < self.segment_size:
                    padding = torch.zeros(
                        self.segment_size - segment.shape[0], 
                        hidden_dim, 
                        device=segment.device
                    )
                    segment = torch.cat([segment, padding], dim=0)
                
                batch_segments.append(segment)
                
                # Record segment info
                segment_info.append({
                    'batch_idx': batch_idx,
                    'segment_idx': segment_idx,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'actual_length': end_pos - start_pos
                })
                
                start_pos += self.stride
                segment_idx += 1
            
            segments.extend(batch_segments)
        
        return torch.stack(segments, dim=0), segment_info


class SimilarityCalculator(nn.Module):
    """
    Computes pairwise similarities between segments.
    Paper: "Edge weights computed via cosine similarity"
    """
    
    def __init__(self, config: LocalGraphConfig):
        super().__init__()
        self.config = config
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, segments: torch.Tensor) -> torch.Tensor:
        """
        Calculate pairwise similarities between segments.
        
        Args:
            segments: [num_segments, segment_size, hidden_dim]
            
        Returns:
            similarities: [num_segments, num_segments]
        """
        # Pool segments to get segment-level representations
        # Using mean pooling as specified in paper
        segment_reprs = torch.mean(segments, dim=1)  # [num_segments, hidden_dim]
        
        # Normalize for cosine similarity
        segment_reprs = F.normalize(segment_reprs, p=2, dim=-1)
        
        # Compute cosine similarities
        similarities = torch.matmul(segment_reprs, segment_reprs.t())
        
        # Apply temperature scaling
        similarities = similarities / self.temperature
        
        return similarities


class GraphConstructor(nn.Module):
    """
    Constructs graph from similarity matrix using thresholding.
    Paper: "Graph G = (V, E) where E = {(i,j) | sim(i,j) > τ}"
    """
    
    def __init__(self, config: LocalGraphConfig):
        super().__init__()
        self.config = config
        self.similarity_threshold = config.similarity_threshold
        self.max_neighbors = config.max_neighbors
        
    def forward(self, similarities: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct sparse graph from similarity matrix.
        
        Args:
            similarities: [num_segments, num_segments]
            
        Returns:
            edge_index: [2, num_edges] - COO format edges
            edge_weights: [num_edges] - Edge weights
        """
        num_segments = similarities.shape[0]
        device = similarities.device
        
        # Apply threshold
        mask = similarities > self.similarity_threshold
        
        # Remove self-loops
        mask.fill_diagonal_(False)
        
        # Limit number of neighbors per node
        if self.max_neighbors > 0:
            # For each node, keep only top-k connections
            for i in range(num_segments):
                row_mask = mask[i]
                if row_mask.sum() > self.max_neighbors:
                    # Get top-k similarities
                    row_sim = similarities[i].clone()
                    row_sim[~row_mask] = -float('inf')
                    _, top_indices = torch.topk(row_sim, self.max_neighbors)
                    
                    # Update mask
                    mask[i] = False
                    mask[i, top_indices] = True
        
        # Convert to COO format
        edge_index = mask.nonzero().t()  # [2, num_edges]
        
        # Get edge weights
        edge_weights = similarities[mask]
        
        return edge_index, edge_weights


class RelationEncoder(nn.Module):
    """
    Encodes edge relations using multi-head attention.
    Paper: "Edge features encoded via relation-aware attention"
    """
    
    def __init__(self, config: LocalGraphConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_dim // self.num_heads
        
        assert self.hidden_dim % self.num_heads == 0
        
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                segments: torch.Tensor, 
                edge_index: torch.Tensor,
                edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Encode edge relations and update node representations.
        
        Args:
            segments: [num_segments, segment_size, hidden_dim]
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
            
        Returns:
            updated_segments: [num_segments, segment_size, hidden_dim]
        """
        num_segments, segment_size, hidden_dim = segments.shape
        
        # Pool segments for node-level representations
        node_reprs = torch.mean(segments, dim=1)  # [num_segments, hidden_dim]
        
        # Multi-head attention
        Q = self.q_proj(node_reprs).view(num_segments, self.num_heads, self.head_dim)
        K = self.k_proj(node_reprs).view(num_segments, self.num_heads, self.head_dim)
        V = self.v_proj(node_reprs).view(num_segments, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 1)  # [num_heads, num_segments, head_dim]
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        # Apply edge weights as bias
        # Create full attention matrix from sparse edges
        full_scores = torch.full(
            (self.num_heads, num_segments, num_segments),
            -1e9,
            device=segments.device
        )
        
        # Fill in scores for existing edges
        src, dst = edge_index[0], edge_index[1]
        edge_weights_expanded = edge_weights.unsqueeze(0).expand(self.num_heads, -1)
        
        for head in range(self.num_heads):
            full_scores[head, src, dst] = scores[head, src, dst] + edge_weights_expanded[head]
        
        # Apply softmax
        attn_weights = F.softmax(full_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # [num_heads, num_segments, head_dim]
        out = out.transpose(0, 1).contiguous()  # [num_segments, num_heads, head_dim]
        out = out.view(num_segments, hidden_dim)
        
        # Output projection
        out = self.out_proj(out)
        
        # Broadcast back to segment size
        updated_segments = segments + out.unsqueeze(1)
        
        return updated_segments


class HSGMLocalGraphBuilder(nn.Module):
    """
    Complete HSGM Local Graph Builder pipeline.
    Paper: "Algorithm 1: Local Graph Construction"
    """
    
    def __init__(self, config: LocalGraphConfig):
        super().__init__()
        self.config = config
        
        self.segment_extractor = SegmentExtractor(config)
        self.similarity_calculator = SimilarityCalculator(config)
        self.graph_constructor = GraphConstructor(config)
        self.relation_encoder = RelationEncoder(config)
        
        logger.info(f"Initialized HSGM Local Graph Builder with {config}")
        
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Build local graph from hidden states.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            
        Returns:
            Dictionary containing:
                - segments: [num_segments, segment_size, hidden_dim]
                - segment_info: List of segment metadata
                - edge_index: [2, num_edges] - Graph edges
                - edge_weights: [num_edges] - Edge weights
                - node_embeddings: [num_segments, hidden_dim] - Node representations
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Step 1: Extract segments
        segments, segment_info = self.segment_extractor(hidden_states)
        
        # Step 2: Calculate similarities
        similarities = self.similarity_calculator(segments)
        
        # Step 3: Construct graph
        edge_index, edge_weights = self.graph_constructor(similarities)
        
        # Step 4: Encode relations
        updated_segments = self.relation_encoder(segments, edge_index, edge_weights)
        
        # Extract node embeddings (pooled segments)
        node_embeddings = torch.mean(updated_segments, dim=1)
        
        logger.info(f"Built local graph with {len(segment_info)} nodes and {edge_index.shape[1]} edges")
        
        return {
            'segments': updated_segments,
            'segment_info': segment_info,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'node_embeddings': node_embeddings,
            'similarities': similarities
        }


class GNNNodeAggregator(nn.Module):
    """
    Aggregates node features using GNN propagation.
    Paper: "Node features aggregated via graph convolution"
    """
    
    def __init__(self, config: LocalGraphConfig, num_layers: int = 2):
        super().__init__()
        self.config = config
        self.num_layers = num_layers
        
        self.gnn_layers = nn.ModuleList([
            nn.ModuleDict({
                'linear': nn.Linear(config.hidden_dim, config.hidden_dim),
                'activation': nn.ReLU(),
                'dropout': nn.Dropout(config.dropout)
            })
            for _ in range(num_layers)
        ])
        
    def forward(self, 
                node_embeddings: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weights: torch.Tensor) -> torch.Tensor:
        """
        Aggregate node features through GNN layers.
        
        Args:
            node_embeddings: [num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_weights: [num_edges]
            
        Returns:
            aggregated_embeddings: [num_nodes, hidden_dim]
        """
        x = node_embeddings
        
        for layer in self.gnn_layers:
            # Message passing
            src, dst = edge_index[0], edge_index[1]
            
            # Aggregate messages
            messages = x[src] * edge_weights.unsqueeze(1)
            aggregated = torch.zeros_like(x)
            
            # Scatter add messages to destination nodes
            for i, d in enumerate(dst):
                aggregated[d] += messages[i]
            
            # Normalize by degree
            degree = torch.zeros(x.shape[0], device=x.device)
            for d in dst:
                degree[d] += 1
            degree = torch.clamp(degree, min=1)
            
            aggregated = aggregated / degree.unsqueeze(1)
            
            # Apply linear transformation
            x = layer['linear'](x + aggregated)
            x = layer['activation'](x)
            x = layer['dropout'](x)
            
        return x


def test_hsgm_local_graph_builder():
    """Test the HSGM Local Graph Builder implementation."""
    
    print("Testing HSGM Local Graph Builder...")
    
    # Create test configuration
    config = LocalGraphConfig(
        segment_size=16,
        overlap_size=4,
        similarity_threshold=0.5,
        max_neighbors=4,
        hidden_dim=128,
        num_heads=4
    )
    
    # Initialize builder
    builder = HSGMLocalGraphBuilder(config)
    
    # Create test input
    batch_size = 2
    seq_len = 64
    hidden_dim = 128
    
    hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Build graph
    with torch.no_grad():
        result = builder(hidden_states)
    
    # Verify outputs
    assert 'segments' in result
    assert 'segment_info' in result
    assert 'edge_index' in result
    assert 'edge_weights' in result
    assert 'node_embeddings' in result
    
    segments = result['segments']
    segment_info = result['segment_info']
    edge_index = result['edge_index']
    node_embeddings = result['node_embeddings']
    
    print(f"✓ Extracted {len(segment_info)} segments")
    print(f"✓ Segment shape: {segments.shape}")
    print(f"✓ Edge index shape: {edge_index.shape}")
    print(f"✓ Node embeddings shape: {node_embeddings.shape}")
    
    # Test node aggregator
    aggregator = GNNNodeAggregator(config, num_layers=2)
    
    with torch.no_grad():
        aggregated = aggregator(
            node_embeddings,
            edge_index,
            result['edge_weights']
        )
    
    print(f"✓ Aggregated embeddings shape: {aggregated.shape}")
    
    print("✅ HSGM Local Graph Builder test passed!")
    
    return result


if __name__ == "__main__":
    test_result = test_hsgm_local_graph_builder()
    print("\nHSGM Local Graph Builder implementation complete!")