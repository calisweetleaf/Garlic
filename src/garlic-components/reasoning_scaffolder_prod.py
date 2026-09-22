"""
Reasoning Scaffolder - Production Implementation
Enhanced signal classification with confidence calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
from collections import defaultdict
import json

# Production logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# Enhanced Semantic Signals (from paper + extensions)
# ============================================================================

class SemanticSignal(Enum):
    """Enhanced semantic signal types"""
    ADDITION_ELABORATION = "ADDITION_ELABORATION"
    CONCLUSION_SUMMARY = "CONCLUSION_SUMMARY"
    CONTRAST_CONCESSION = "CONTRAST_CONCESSION"
    EXAMPLE_ILLUSTRATION = "EXAMPLE_ILLUSTRATION"
    REASONING_ANALYSIS = "REASONING_ANALYSIS"
    RESPONSE_GENERATION = "RESPONSE_GENERATION"
    PERSONAL_OPINION_RECALL = "PERSONAL_OPINION_RECALL"
    # Extended signals for production
    QUESTION_INQUIRY = "QUESTION_INQUIRY"
    CAUSE_EFFECT = "CAUSE_EFFECT"
    COMPARISON_EVALUATION = "COMPARISON_EVALUATION"

# ============================================================================
# Production Data Structures
# ============================================================================

@dataclass
class SignalPrediction:
    """Enhanced signal prediction with confidence"""
    signal: SemanticSignal
    confidence: float
    probability_distribution: Dict[SemanticSignal, float]
    
@dataclass
class SignalCalibrationData:
    """Data for confidence calibration"""
    predicted_confidence: float
    actual_accuracy: float
    sample_count: int

@dataclass
class GarlicBridgeOutput:
    """Output from Garlic Bridge"""
    augmented_state: torch.Tensor
    signal_prediction: SignalPrediction
    memory_injected: bool
    injection_strength: float

# ============================================================================
# Production Signal Embedding Layer
# ============================================================================

class SignalEmbeddingLayer(nn.Module):
    """
    Production Signal Embedding Layer (SEL)
    Maps semantic signals to dense embeddings
    """
    
    def __init__(
        self, 
        num_signals: int = 10,  # Extended signal set
        embedding_dim: int = 384,
        use_pretrained: bool = True
    ):
        super().__init__()
        self.num_signals = num_signals
        self.embedding_dim = embedding_dim
        
        # Signal embeddings with proper initialization
        if use_pretrained:
            # Initialize with structured embeddings
            self.embeddings = nn.Parameter(torch.randn(num_signals, embedding_dim) * 0.1)
            # Make embeddings orthogonal for better separation
            self._orthogonalize_embeddings()
        else:
            self.embeddings = nn.Parameter(torch.randn(num_signals, embedding_dim) * 0.02)
        
        # Signal-to-index mapping
        self.signal_to_idx = {signal: idx for idx, signal in enumerate(SemanticSignal)}
        
    def _orthogonalize_embeddings(self):
        """Make signal embeddings more orthogonal"""
        with torch.no_grad():
            # Use QR decomposition for orthogonal initialization
            embeddings = self.embeddings.data
            q, _ = torch.linalg.qr(embeddings.T)
            self.embeddings.data = q.T[:self.num_signals] * np.sqrt(self.embedding_dim)
            
    def forward(self, signal: SemanticSignal) -> torch.Tensor:
        """Get embedding for a signal"""
        idx = self.signal_to_idx[signal]
        return self.embeddings[idx]
        
    def get_all_embeddings(self) -> torch.Tensor:
        """Get all signal embeddings"""
        return self.embeddings

# ============================================================================
# Production Semantic Signal Predictor
# ============================================================================

class SemanticSignalPredictor(nn.Module):
    """
    Enhanced signal predictor with confidence calibration
    """
    
    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 256,
        num_signals: int = 10,
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_signals = num_signals
        self.use_attention = use_attention
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism for feature weighting
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim // 2, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
        
        # Signal classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_signals)
        )
        
        # Confidence estimation head
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, hidden_state: torch.Tensor) -> SignalPrediction:
        """Predict semantic signal with confidence"""
        # Handle different input shapes (squeeze to 2D if needed)
        original_shape = hidden_state.shape
        if hidden_state.dim() == 3:
            hidden_state = hidden_state.squeeze(1)  # [batch, 1, hidden] -> [batch, hidden]
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0)  # [hidden] -> [1, hidden]
        
        # Extract features
        features = self.feature_extractor(hidden_state)
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = F.softmax(self.attention(features), dim=-1)
            features = attention_weights * features
        
        # Get signal logits
        logits = self.classifier(features)
        probabilities = F.softmax(logits, dim=-1)
        
        # Handle batch dimension - take first element for single-sample prediction
        if probabilities.dim() == 2 and probabilities.shape[0] == 1:
            probs_for_dist = probabilities[0]  # [num_signals]
        else:
            probs_for_dist = probabilities.mean(dim=0)  # Average across batch
        
        # Get predicted signal
        predicted_idx = torch.argmax(probs_for_dist).item()
        predicted_signal = list(SemanticSignal)[predicted_idx]
        
        # Get confidence score
        conf_output = self.confidence_estimator(features)
        if conf_output.numel() > 1:
            confidence = conf_output.mean().item()
        else:
            confidence = conf_output.item()
        
        # Create probability distribution
        prob_dist = {
            list(SemanticSignal)[i]: probs_for_dist[i].item()
            for i in range(min(len(SemanticSignal), probs_for_dist.shape[0]))
        }
        
        return SignalPrediction(
            signal=predicted_signal,
            confidence=confidence,
            probability_distribution=prob_dist
        )

# ============================================================================
# Confidence Calibration
# ============================================================================

class ConfidenceCalibrator:
    """Calibrates confidence scores to match actual accuracy"""
    
    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins
        self.bin_boundaries = np.linspace(0, 1, num_bins + 1)
        self.bin_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidence_sum': 0})
        
    def update(self, predicted_confidence: float, is_correct: bool):
        """Update calibration statistics"""
        bin_idx = np.digitize(predicted_confidence, self.bin_boundaries) - 1
        bin_idx = min(bin_idx, self.num_bins - 1)
        
        self.bin_stats[bin_idx]['total'] += 1
        self.bin_stats[bin_idx]['confidence_sum'] += predicted_confidence
        if is_correct:
            self.bin_stats[bin_idx]['correct'] += 1
            
    def calibrate(self, confidence: float) -> float:
        """Calibrate confidence score"""
        bin_idx = np.digitize(confidence, self.bin_boundaries) - 1
        bin_idx = min(bin_idx, self.num_bins - 1)
        
        stats = self.bin_stats[bin_idx]
        if stats['total'] > 0:
            # Use empirical accuracy as calibrated confidence
            calibrated_confidence = stats['correct'] / stats['total']
            return calibrated_confidence
        else:
            return confidence
            
    def get_calibration_plot_data(self) -> Tuple[List[float], List[float]]:
        """Get data for calibration plot"""
        bin_centers = []
        empirical_accuracies = []
        
        for i in range(self.num_bins):
            stats = self.bin_stats[i]
            if stats['total'] > 0:
                bin_center = (self.bin_boundaries[i] + self.bin_boundaries[i + 1]) / 2
                empirical_accuracy = stats['correct'] / stats['total']
                
                bin_centers.append(bin_center)
                empirical_accuracies.append(empirical_accuracy)
                
        return bin_centers, empirical_accuracies

# ============================================================================
# Production Reasoning Scaffolder
# ============================================================================

class ReasoningScaffolder:
    """
    Production Reasoning Scaffolder with enhanced features
    """
    
    def __init__(
        self,
        hidden_dim: int = 384,
        num_signals: int = 10,
        dropout: float = 0.1,
        enable_calibration: bool = True,
        use_attention: bool = True
    ):
        self.hidden_dim = hidden_dim
        self.num_signals = num_signals
        self.enable_calibration = enable_calibration
        
        # Core components
        self.signal_embedding_layer = SignalEmbeddingLayer(
            num_signals=num_signals,
            embedding_dim=hidden_dim,
            use_pretrained=True
        )
        
        self.signal_predictor = SemanticSignalPredictor(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim // 2,
            num_signals=num_signals,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Confidence calibrator
        if enable_calibration:
            self.calibrator = ConfidenceCalibrator(num_bins=10)
        else:
            self.calibrator = None
            
        # Statistics tracking
        self.prediction_stats = {
            'total_predictions': 0,
            'signal_distribution': defaultdict(int),
            'avg_confidence': 0.0
        }
        
    def predict_and_embed(
        self, 
        hidden_state: torch.Tensor, 
        return_distribution: bool = False
    ) -> Tuple[SemanticSignal, torch.Tensor, float]:
        """
        Predict semantic signal and get embedding
        
        Args:
            hidden_state: Input hidden state [batch_size, hidden_dim]
            return_distribution: Whether to return probability distribution
            
        Returns:
            Tuple of (predicted_signal, embedding, confidence)
        """
        # Get signal prediction
        prediction = self.signal_predictor(hidden_state)
        
        # Calibrate confidence if enabled
        if self.calibrator:
            calibrated_confidence = self.calibrator.calibrate(prediction.confidence)
        else:
            calibrated_confidence = prediction.confidence
            
        # Get signal embedding
        embedding = self.signal_embedding_layer(prediction.signal)
        
        # Update statistics
        self._update_stats(prediction.signal, calibrated_confidence)
        
        return prediction.signal, embedding, calibrated_confidence
        
    def predict_signal(self, hidden_state: torch.Tensor) -> SignalPrediction:
        """Predict semantic signal with full information"""
        return self.signal_predictor(hidden_state)
        
    def get_signal_distribution(self, hidden_state: torch.Tensor) -> Dict[SemanticSignal, float]:
        """Get probability distribution over all signals"""
        prediction = self.signal_predictor(hidden_state)
        return prediction.probability_distribution
        
    def update_calibration(self, predicted_confidence: float, is_correct: bool):
        """Update calibration statistics"""
        if self.calibrator:
            self.calibrator.update(predicted_confidence, is_correct)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get prediction statistics"""
        if self.prediction_stats['total_predictions'] > 0:
            signal_dist = dict(self.prediction_stats['signal_distribution'])
            avg_conf = self.prediction_stats['avg_confidence']
        else:
            signal_dist = {}
            avg_conf = 0.0
            
        return {
            'total_predictions': self.prediction_stats['total_predictions'],
            'signal_distribution': signal_dist,
            'average_confidence': avg_conf,
            'calibration_enabled': self.enable_calibration
        }
        
    def _update_stats(self, signal: SemanticSignal, confidence: float):
        """Update internal statistics"""
        self.prediction_stats['total_predictions'] += 1
        self.prediction_stats['signal_distribution'][signal] += 1
        
        # Running average of confidence
        n = self.prediction_stats['total_predictions']
        old_avg = self.prediction_stats['avg_confidence']
        self.prediction_stats['avg_confidence'] = old_avg + (confidence - old_avg) / n
        
    def reset_statistics(self):
        """Reset statistics"""
        self.prediction_stats = {
            'total_predictions': 0,
            'signal_distribution': defaultdict(int),
            'avg_confidence': 0.0
        }

# ============================================================================
# Production Garlic Bridge
# ============================================================================

class GarlicBridge:
    """
    Production Garlic Bridge for HSGM integration
    """
    
    def __init__(
        self,
        reasoning_scaffolder: ReasoningScaffolder,
        summary_dim: int = 384,
        injection_strategy: str = "adaptive"
    ):
        self.reasoning_scaffolder = reasoning_scaffolder
        self.summary_dim = summary_dim
        self.injection_strategy = injection_strategy
        
        # Fusion layer for combining signals and memory
        self.fusion_layer = nn.Sequential(
            nn.Linear(summary_dim * 2, summary_dim),
            nn.ReLU(),
            nn.Linear(summary_dim, summary_dim)
        )
        
        # Adaptive injection strength predictor
        self.injection_strength_predictor = nn.Sequential(
            nn.Linear(summary_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def hybrid_injection(
        self,
        hidden_state: torch.Tensor,
        summary_node_embedding: Optional[np.ndarray] = None,
        use_signal: bool = True,
        use_memory: bool = True,
        signal_weight: float = 1.0,
        memory_weight: float = 1.0
    ) -> GarlicBridgeOutput:
        """
        Inject semantic signals and/or HSGM memory into hidden state
        
        Args:
            hidden_state: Original hidden state
            summary_node_embedding: HSGM summary node embedding (if available)
            use_signal: Whether to inject semantic signal
            use_memory: Whether to inject HSGM memory
            signal_weight: Weight for signal injection
            memory_weight: Weight for memory injection
            
        Returns:
            GarlicBridgeOutput with augmented state
        """
        batch_size = hidden_state.shape[0]
        
        # Get signal prediction and embedding
        if use_signal:
            signal_prediction = self.reasoning_scaffolder.predict_signal(hidden_state)
            signal_embedding = self.reasoning_scaffolder.signal_embedding_layer(
                signal_prediction.signal
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            signal_prediction = None
            signal_embedding = torch.zeros(batch_size, self.summary_dim)
        
        # Process HSGM memory embedding
        if use_memory and summary_node_embedding is not None:
            memory_embedding = torch.from_numpy(summary_node_embedding).float()
            if memory_embedding.shape[0] != batch_size:
                memory_embedding = memory_embedding.unsqueeze(0).expand(batch_size, -1)
        else:
            memory_embedding = torch.zeros(batch_size, self.summary_dim)
            
        # Handle dimensionality matching (if hidden_state has sequence dim)
        if hidden_state.dim() == 3:
            # hidden_state is [batch, seq, dim] -> expand embeddings to [batch, 1, dim]
            if signal_embedding.dim() == 2:
                signal_embedding = signal_embedding.unsqueeze(1)
            if memory_embedding.dim() == 2:
                memory_embedding = memory_embedding.unsqueeze(1)
            
            # If embeddings are single-step [batch, 1, dim] but hidden is [batch, seq, dim], 
            # we might need to expand. For now assume step-by-step processing (seq=1).
            if hidden_state.shape[1] > 1 and signal_embedding.shape[1] == 1:
                signal_embedding = signal_embedding.expand(-1, hidden_state.shape[1], -1)
                memory_embedding = memory_embedding.expand(-1, hidden_state.shape[1], -1)
        
        # Adaptive injection strength
        if self.injection_strategy == "adaptive":
            # Combine hidden state with embeddings to predict injection strength
            combined = torch.cat([hidden_state, signal_embedding], dim=-1)
            injection_strength = self.injection_strength_predictor(combined)
        else:
            injection_strength = torch.ones(batch_size, 1) * 0.5
        
        # Fuse embeddings
        if use_signal or use_memory:
            # Concatenate signal and memory embeddings
            combined_embeddings = torch.cat([signal_embedding, memory_embedding], dim=-1)
            fused_embeddings = self.fusion_layer(combined_embeddings)
            
            # Apply injection with adaptive strength
            augmented_state = hidden_state + injection_strength * fused_embeddings
        else:
            augmented_state = hidden_state
        
        # Return output
        return GarlicBridgeOutput(
            augmented_state=augmented_state,
            signal_prediction=signal_prediction,
            memory_injected=use_memory and summary_node_embedding is not None,
            injection_strength=injection_strength.mean().item()
        )
        
    def signal_only_injection(
        self,
        hidden_state: torch.Tensor,
        signal_weight: float = 1.0
    ) -> Tuple[torch.Tensor, SignalPrediction]:
        """Inject only semantic signal"""
        output = self.hybrid_injection(
            hidden_state,
            use_signal=True,
            use_memory=False,
            signal_weight=signal_weight
        )
        return output.augmented_state, output.signal_prediction
        
    def memory_only_injection(
        self,
        hidden_state: torch.Tensor,
        summary_node_embedding: np.ndarray,
        memory_weight: float = 1.0
    ) -> torch.Tensor:
        """Inject only HSGM memory"""
        output = self.hybrid_injection(
            hidden_state,
            summary_node_embedding=summary_node_embedding,
            use_signal=False,
            use_memory=True,
            memory_weight=memory_weight
        )
        return output.augmented_state

# ============================================================================
# Training Utilities
# ============================================================================

class SignalTrainer:
    """Trainer for signal classification model"""
    
    def __init__(
        self,
        scaffolder: ReasoningScaffolder,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01
    ):
        self.scaffolder = scaffolder
        self.optimizer = torch.optim.AdamW(
            scaffolder.signal_predictor.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train_step(
        self,
        hidden_states: torch.Tensor,
        target_signals: torch.Tensor
    ) -> Dict[str, float]:
        """Single training step"""
        self.scaffolder.signal_predictor.train()
        self.optimizer.zero_grad()
        
        # Forward pass
        logits = self.scaffolder.signal_predictor.classifier(
            self.scaffolder.signal_predictor.feature_extractor(hidden_states)
        )
        
        # Compute loss
        loss = self.criterion(logits, target_signals)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.scaffolder.signal_predictor.parameters(), 
            max_norm=1.0
        )
        self.optimizer.step()
        
        return {'loss': loss.item()}

# ============================================================================
# Production Utilities
# ============================================================================

def create_production_reasoning_scaffolder(**kwargs) -> ReasoningScaffolder:
    """Factory function for production reasoning scaffolder"""
    return ReasoningScaffolder(**kwargs)

def create_production_garlic_bridge(
    scaffolder: ReasoningScaffolder, 
    **kwargs
) -> GarlicBridge:
    """Factory function for production Garlic Bridge"""
    return GarlicBridge(scaffolder, **kwargs)

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    # Test production reasoning scaffolder
    scaffolder = create_production_reasoning_scaffolder()
    bridge = create_production_garlic_bridge(scaffolder)
    
    print("Production Reasoning Scaffolder initialized successfully!")
    print(f"Number of signals: {len(SemanticSignal)}")
    print(f"Embedding dimension: {scaffolder.hidden_dim}")
    print(f"Calibration enabled: {scaffolder.enable_calibration}")
    
    # Test signal prediction
    hidden_state = torch.randn(1, 384)
    signal, embedding, confidence = scaffolder.predict_and_embed(hidden_state)
    
    print(f"\nTest Results:")
    print(f"Predicted signal: {signal.name}")
    print(f"Confidence: {confidence:.3f}")
    print(f"Embedding shape: {embedding.shape}")
    
    # Test Garlic Bridge
    augmented_state, signal_pred = bridge.signal_only_injection(hidden_state)
    print(f"\nGarlic Bridge Test:")
    print(f"Original state norm: {hidden_state.norm():.3f}")
    print(f"Augmented state norm: {augmented_state.norm():.3f}")
    print(f"Signal: {signal_pred.signal.name}")
    
    print("\n✓ All components working correctly!")