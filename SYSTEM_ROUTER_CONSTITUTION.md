# THE SYSTEM ROUTER CONSTITUTION
## Sovereign Intelligence Infrastructure — GARLIC Integration v1.0

**Created:** 2026-04-26  
**Author:** MAIL ROOM (Morpheus Advanced Intelligence Lab)  
**Purpose:** Complete specification for merging System-Router + ADA-Step-Entropy into unified production system  
**Status:** READY FOR EXECUTION  

---

# I. ARCHITECTURAL OVERVIEW

## 1.1 The Convergence Point

Two mature codebases converge into ONE sovereign system:

```
BEFORE:
System-Router/              ADA-Step-Entropy/
├── neural_router.py        ├── ada_bridge.py
├── Garlic-Components/      ├── step_entropy.ads/adb
│   ├── hsgm_*.py           ├── libgarlic_core.so
│   ├── garlic_*.py         └── 16/16 tests passing
│   ├── reasoning_*.py      
│   └── entropy_*.py        
└── memory_injection_system.py

AFTER:
System-Router/
├── system_router.py        ← THE UNIFIED SYSTEM
│   ├── ModelSlot routing (NOT reasoning effort)
│   ├── HSGM pipeline internalized
│   ├── Step Entropy (Ada bridge)
│   ├── Reasoning Scaffolder
│   ├── Entropy Regularized Router
│   ├── Garlic Bridge lifecycle
│   ├── GlobalGraphMemory
│   └── Memory Injection System
├── ada_bridge/
│   ├── ada_bridge.py
│   ├── step_entropy.ads
│   ├── step_entropy.adb
│   ├── step_entropy_c_api.ads
│   ├── step_entropy_c_api.adb
│   └── lib/libgarlic_core.so
└── CONSTITUTION.md         ← THIS DOCUMENT
```

## 1.2 The Unified Forward Pass

```
╔═══════════════════════════════════════════════════════════════════╗
║              SYSTEM ROUTER — 13-STAGE FORWARD PASS                ║
╚═══════════════════════════════════════════════════════════════════╝

INPUT: context dict, optional hidden_states, optional logits
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 0: INPUT PREPARATION                                       │
│ ─────────────────────────                                        │
│ InputPreparer:                                                   │
│   - HashTextEncoder (messages → embeddings)                      │
│   - ProfileEncoder (behavioral features)                         │
│   - MetadataEncoder (temporal + flags)                           │
│ → Produces: message_embs [B, S, D], profile [B, 128],           │
│             metadata [B, 64]                                     │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 1: GLOBAL GRAPH MEMORY — History Context Prepend          │
│ ────────────────────────────────────────────────────             │
│ GlobalGraphMemory.get_context() → history_graph                 │
│ IF not empty: prepend history to hidden_states                  │
│ → Enables: infinite context via turn-to-turn accumulation       │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 2: HSGM PIPELINE — Context Compression                    │
│ ─────────────────────────────────────────                        │
│ IF hidden_states available:                                     │
│   1. LocalGraphBuilder → segment graphs                         │
│   2. GNNSummaryExtractor → compressed nodes (90-102x)           │
│   3. HierarchicalQueryProcessor → retrieval-ready context       │
│ → Produces: compressed_context [B, N_summary, D]                │
│ → Compression: 512 tokens → ~5 nodes (102x)                     │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 3: CONTEXT ENCODING — Neural Representation               │
│ ──────────────────────────────────────────────                   │
│ ContextEncoder (4-layer Transformer):                           │
│   - Encode message sequence                                     │
│   - Fuse profile + metadata                                     │
│ → Produces: context_emb [B, D]                                  │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 4-5: NEURAL PREDICTION — Slots + Tools                    │
│ ─────────────────────────────────────────────                    │
│ SlotPredictorNetwork:                                           │
│   - model_selector_head → [B, 3] probs (NOT reasoning effort!) │
│   - tool_gates → {tool_name: [B, 1]} enables                   │
│   - tool_attention → [B, num_tools] weights                    │
│ → Produces: SlotPredictions(model_slot, tool_enables,          │
│                              tool_weights, confidence)          │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 6: SAFETY VALIDATION — Constraint Enforcement             │
│ ────────────────────────────────────────────                     │
│ SafetyValidator:                                                │
│   - Tool sparsity check (max 3 tools)                           │
│   - Tool dependency validation                                  │
│   - HARD/SOFT violation tracking                                │
│ → Produces: validated SlotPredictions                           │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 7: MODEL SLOT SELECTION                                   │
│ ──────────────────────────────                                   │
│ ModelLoader:                                                    │
│   - predict_slot(model_slot probs) → ModelSlot enum            │
│   - get_active_model(slot_id) → loaded model or None           │
│ → Produces: active_slot (SLOT_A/B/C), model reference          │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 8: STEP ENTROPY — Ada Native Routing Signal               │
│ ───────────────────────────────────────────────────              │
│ IF logits available:                                            │
│   AdaBridge.calculate_batch_step_entropy(logits, token_ids)    │
│   → entropy_value (bits per token)                             │
│   → entropy_level ('fast' < 2.0, 'normal' 2-4, 'slow' > 4.0)  │
│ Thresholds from paper §3.1:                                     │
│   - K_SKIP_LOW = 0.50  (fast path)                             │
│   - K_SKIP_HIGH = 0.80 (slow path, memory retrieval)           │
│ → Produces: entropy_value, entropy_level_str, routing_path     │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 9: REASONING SCAFFOLDER — Semantic Signals                │
│ ─────────────────────────────────────────────────────            │
│ IF hidden_states available:                                     │
│   ReasoningScaffolder.predict_and_embed(pooled_hidden)         │
│   → semantic_signal (EXPLORATION, VERIFICATION, etc.)          │
│   → signal_embedding [B, D]                                    │
│   → confidence score                                            │
│ → Produces: semantic_signals [B, S, D] expanded to seq_len     │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 10: ENTROPY REGULARIZED ROUTER — Expert Selection        │
│ ────────────────────────────────────────────────────────────     │
│ IF hidden_states + compressed_context available:               │
│   EntropyRegularizedRouter(hidden, graph_context, signals)    │
│   → top_k expert_indices                                       │
│   → gating_weights                                             │
│   → routing_entropy, is_ood, path_indices                      │
│ Pangu 5.5 architecture (from paper):                           │
│   - GNN-based routing (optional, if torch_geometric)           │
│   - Entropy minimization (sample-wise)                         │
│   - Load balancing (batch-wise entropy max)                    │
│ → Produces: RoutingDecision(path, expert_indices, weights)     │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 11: GARLIC BRIDGE — Memory Injection                     │
│ ───────────────────────────────────────────                      │
│ inject_memory_garlic_bridge():                                 │
│   - Pool graph_memory → [B, S, D]                              │
│   - Project semantic_signals → [B, S, D]                       │
│   - Concat [graph; signals] → [B, S, 2D]                       │
│   - Bridge projection → memory_injection [B, S, D]             │
│   - Additive: h' = h + memory_injection                        │
│ GarlicInjectionLayer.set_context(graph, signals)               │
│ → Produces: augmented_states [B, S, D]                         │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 12: TEMPLATE SELECTION + PROMPT ASSEMBLY                 │
│ ───────────────────────────────────────────────────────          │
│ TemplateSelectorNetwork:                                       │
│   - flatten_slots(model_slot, tool_enables, tool_weights)     │
│   - gate_network → template_weights [B, num_templates]        │
│ TemplateLibrary.assemble():                                    │
│   - Select template via weighted combination                   │
│   - Inject memory context XML                                  │
│   - Inject tool definitions                                    │
│   - Safety validate output                                     │
│ → Produces: prompt (str)                                       │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ STAGE 13: GLOBAL MEMORY UPDATE                                 │
│ ───────────────────────────────────────────                      │
│ GlobalGraphMemory.add_turn():                                  │
│   - Store: input_graph, entropy_value, model_slot             │
│   - Update accumulated global_graph (mean-pooling)             │
│   - Track turn history                                         │
│ → Persists across turns, flushes only on session end          │
└──────────────────────────────────────────────────────────────────┘
   ↓
OUTPUT: SystemOutput(
    prompt, model_slot, routing_decision, entropy_level,
    compression_ratio, augmented_states, hsgm_summary, trace
)
```

---

# II. COMPONENT REGISTRY

## 2.1 Core Papers (1:1 Implementations)

### Paper 1: **Step Entropy** (arXiv:2508.03346)
- **File:** `ada_bridge.py` + `libgarlic_core.so` (Ada/C FFI)
- **Implementation Status:** ✅ COMPLETE (16/16 tests passing)
- **Key Formulas:**
  ```
  H(X) = -Σ p(x) log p(x)                    [Shannon Entropy]
  
  Routing Decision (§3.1):
    IF H < K_SKIP_LOW (0.50):  → FAST path
    IF K_SKIP_LOW ≤ H ≤ K_SKIP_HIGH: → NORMAL
    IF H > K_SKIP_HIGH (0.80): → SLOW (retrieve memory)
  
  Compression (§3.2 Step 4):
    Insert [SKIP] token when H < threshold
    Prune ratio = 0.80 (optimal from Table 3)
  ```
- **SOTA++ Features (beyond paper):**
  - ✅ Adaptive thresholds (Welford online mean/std)
  - ✅ Entropy histogram (configurable bins)
  - ✅ GRPO rewards (Eq. 13-15, Table 3 exact values)
  - ✅ Z-score normalization (cross-model portability)

### Paper 2: **HSGM** (Hierarchical Segment-Graph Memory)
- **Files:** `hsgm_local_graph_builder.py`, `gnn_summary_extractor.py`, `hierarchical_query_processor.py`
- **Implementation Status:** ✅ COMPLETE
- **Key Architecture:**
  ```
  Text Segments → Local Graphs → GNN Pooling → Summary Nodes
  
  Compression Ratio (from paper §4.2):
    Baseline attention: O(n²)
    HSGM: O(n/k) where k = segment_size
    Observed: 90-102x compression in practice
  ```
- **SOTA++ Additions:**
  - [ ] **Multi-scale graph construction** — build graphs at 3 scales (fine/medium/coarse) for hierarchical injection
  - [ ] **Graph attention pooling** — replace mean-pooling with GAT-based summary extraction
  - [ ] **Temporal edge weighting** — add recency-weighted edges for turn-to-turn graph evolution

### Paper 3: **Reasoning Scaffolding** (arXiv:2509.23619)
- **File:** `reasoning_scaffolder_prod.py`
- **Implementation Status:** ✅ COMPLETE
- **Key Signals:**
  ```
  SemanticSignal enum:
    EXPLORATION, VERIFICATION, SYNTHESIS, 
    REFLECTION, CONSTRAINT_CHECK, ERROR_RECOVERY
  
  Signal → Embedding → Inject into hidden states
  ```
- **SOTA++ Additions:**
  - [ ] **Signal co-occurrence patterns** — learn which signals cluster together for different task types
  - [ ] **Signal strength modulation** — adaptive signal scaling based on confidence
  - [ ] **Meta-signal injection** — add METACOGNITIVE signal for self-monitoring

### Paper 4: **Entropy Regularized Routing** (Pangu 5.5 architecture)
- **File:** `entropy_regularized_router.py`
- **Implementation Status:** ✅ COMPLETE
- **Key Loss:**
  ```
  L_routing = L_sample_entropy - L_batch_entropy
  
  Sample-wise: minimize H to increase confidence per token
  Batch-wise: maximize H to ensure expert diversity (load balancing)
  
  Regularization strength: λ = 0.01 (from Pangu paper)
  ```
- **SOTA++ Additions:**
  - [ ] **Curriculum entropy scheduling** — start with high λ, anneal during training
  - [ ] **Expert specialization metrics** — track which experts become specialized for which patterns
  - [ ] **Dynamic top-k** — vary k based on entropy (low entropy = k=1, high = k=3)

## 2.2 Novel Components (Our Innovations)

### Innovation 1: **ModelSlot Architecture** (Replaces Reasoning Effort)
```python
class ModelSlot(Enum):
    """Canon-agnostic model slot indices."""
    SLOT_A = 0  # Typically: fast/light
    SLOT_B = 1  # Typically: general
    SLOT_C = 2  # Typically: specialized/deep

# Neural head outputs [B, 3] probs → select slot
# NO hardcoded semantics — user defines at runtime
```

**Why This Matters:**
- Reasoning effort = proxy for compute budget (fixed semantics)
- ModelSlot = dispatch mechanism (learned specialization)
- Router learns: "this context type → slot 2" from training
- Entropy router provides second signal: "high uncertainty → deeper slot"
- **Two routing signals converge → emergent compute allocation**

### Innovation 2: **GlobalGraphMemory** (Persistent HSGM State)
```python
class GlobalGraphMemory:
    """
    Turn-to-turn persistent graph accumulation.
    NEVER flushes until session shutdown.
    
    Each turn: input → reasoning → output → compress → add to graph
    Next turn: prepend accumulated graph → HSGM → compress again
    
    Result: "infinite context" via compressed graph topology
    """
```

**Why This Matters:**
- HSGM compression WITHOUT persistent state = per-request utility
- HSGM WITH GlobalGraphMemory = session-aware intelligence
- Graph grows every turn but compression keeps injection footprint small
- Router gets smarter about routing as session progresses (graph encodes history)

### Innovation 3: **DARE-Inspired Adaptive Expert Allocation**
From paper: "DARE: Difficulty-Aware Dynamic Routing for Mixture of Experts" (ICLR 2026)

**Core Insight:**
- Top-K routing: every token gets K experts (wasteful)
- DARE: use log-perplexity as difficulty proxy → dynamic K
- **Our adaptation:** use Step Entropy as difficulty proxy

```python
def adaptive_expert_count(entropy_value: float) -> int:
    """
    Map entropy → number of experts to activate.
    
    Low entropy (< 2.0): activate 1 expert (fast)
    Medium (2-4): activate 2 experts
    High (> 4.0): activate 3+ experts (slow, memory-augmented)
    
    Learnable thresholds trained end-to-end.
    """
    if entropy_value < learnable_threshold_low:
        return 1
    elif entropy_value < learnable_threshold_high:
        return 2
    else:
        return min(num_experts, 3)
```

### Innovation 4: **Multi-Layer Injection Strategy**
Current: inject at 3 fixed layers (early, middle, late)

**SOTA++ Enhancement:**
```python
class AdaptiveInjectionScheduler:
    """
    Dynamic per-layer injection based on entropy profile.
    
    Low entropy: inject only at late layers (minimal perturbation)
    High entropy: inject at ALL layers (maximum augmentation)
    
    Learns injection schedule per entropy bin during training.
    """
    
    def get_injection_layers(self, entropy: float) -> List[int]:
        if entropy < 2.0:
            return [num_layers - 1]  # Last layer only
        elif entropy < 4.0:
            return [num_layers // 3, 2 * num_layers // 3, num_layers - 1]
        else:
            return list(range(num_layers))  # All layers
```

**Why This Matters:**
- Not in any paper — genuinely novel
- Adapts computational trajectory depth to query complexity
- Low-entropy queries: fast path through model, late injection
- High-entropy: full augmentation across all layers
- **This is what Daeron meant by "control the attention even more"**

### Innovation 5: **GNN-MoE Fusion** (Our Architecture)
Inspired by: MoDE (AAAI-22) — disentangled domain-specific experts

**Architecture:**
```
Input Features
    ↓
Feature Disentanglement:
    ├─→ Domain-Specific Features → Expert Gating
    ├─→ Domain-General Features → Shared Processing
    └─→ Dynamic-Context Features → Attention Modulation
    ↓
GNN Propagation:
    - Node = expert
    - Edge = co-activation pattern
    - Message passing → expert collaboration
    ↓
Expert Selection:
    - Domain features → which experts are relevant
    - Dynamic features → how to weight them
    - GNN output → final expert routing
```

**Why This Matters:**
- Standard MoE: flat expert selection (no structure)
- GNN-MoE: experts form a graph, collaborate via message passing
- Domain-specific vs general features → specialization emerges
- **The graph structure IS the inductive bias for routing**

---

# III. INTEGRATION ARCHITECTURE

## 3.1 File Structure Post-Merge

```
System-Router/
├── system_router.py                    ← THE UNIFIED SYSTEM (3500-4500 lines)
│   │
│   ├─ [IMPORTED FROM neural_router.py - KEEP VERBATIM]
│   │  ├── HashTextEncoder (lines 405-554)
│   │  ├── ProfileEncoder (lines 557-704)
│   │  ├── MetadataEncoder (lines 707-805)
│   │  ├── InputPreparer (lines 808-986)
│   │  ├── ContextEncoder (lines 101-171)
│   │  ├── TemplateSelectorNetwork (minor update)
│   │  ├── SafetyValidator
│   │  └── TemplateLibrary (lines 987-1590)
│   │
│   ├─ [UPDATED FROM neural_router.py]
│   │  ├── ReasoningEffort → ModelSlot (enum rename)
│   │  ├── RouterConfig → SystemRouterConfig (merged fields)
│   │  ├── SlotPredictions.reasoning_effort → model_slot (field rename)
│   │  ├── SlotPredictorNetwork.reasoning_head → model_selector_head
│   │  └── SafeRouterWrapper → SystemRouterWrapper (extended output)
│   │
│   ├─ [NEW COMPONENTS]
│   │  ├── ModelSlot enum
│   │  ├── ModelSlotConfig dataclass
│   │  ├── ModelLoader class
│   │  ├── SystemOutput dataclass (replaces Tuple[str, Dict])
│   │  ├── GlobalGraphMemory class
│   │  ├── AdaptiveInjectionScheduler class
│   │  └── StepEntropyInterface protocol
│   │
│   ├─ [INTERNALIZED FROM Garlic-Components/]
│   │  ├── HSGMLocalGraphBuilder (from hsgm_local_graph_builder.py)
│   │  ├── GNNSummaryExtractor (from gnn_summary_extractor.py)
│   │  ├── HierarchicalQueryProcessor (from hierarchical_query_processor.py)
│   │  ├── ReasoningScaffolder (from reasoning_scaffolder_prod.py)
│   │  ├── EntropyRegularizedRouter (from entropy_regularized_router.py)
│   │  ├── GarlicBridge linear layers (from garlic_orchestrator.py)
│   │  └── All component configs (LocalGraphConfig, etc.)
│   │
│   └─ [MAIN CLASSES]
│      ├── SystemRouterConfig (merged config)
│      ├── SystemRouter (main forward pass - 13 stages)
│      └── SystemRouterWrapper (production wrapper with fallback)
│
├── ada_bridge/                         ← ADA STEP ENTROPY (MOVE FROM ../ADA-Step-Entropy/)
│   ├── ada_bridge.py                   ← Python FFI bridge
│   ├── step_entropy.ads                ← Ada spec
│   ├── step_entropy.adb                ← Ada body
│   ├── step_entropy_c_api.ads          ← C API spec
│   ├── step_entropy_c_api.adb          ← C API body
│   ├── garlic_core.gpr                 ← GNAT project
│   ├── lib/
│   │   └── libgarlic_core.so           ← Compiled binary (16/16 tests ✅)
│   └── tests/
│       └── test_ada_bridge.py          ← 16 test suite
│
├── memory_injection_system.py          ← KEEP AS-IS (works standalone)
│
├── CONSTITUTION.md                     ← THIS DOCUMENT
├── IMPLEMENTATION_LOG.md               ← Checklist tracker
└── tests/
    ├── test_system_router.py
    ├── test_model_loader.py
    ├── test_global_memory.py
    └── test_integration.py
```

## 3.2 Import Consolidation

**Before (broken):**
```python
# garlic_orchestrator.py line 45
from neural_router_prod import (  # ← DOESN'T EXIST
    NeuralPromptRouter, 
    RouterConfig, 
    RouterOutput,  # ← DOESN'T EXIST IN neural_router.py
    ContextFeatures
)
```

**After (unified):**
```python
# system_router.py - everything in one file, no external imports
from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn
from enum import Enum
from dataclasses import dataclass

# Ada bridge import (optional, falls back to Python if .so not found)
try:
    from ada_bridge.ada_bridge import GarlicAdaStepEntropy
    ADA_AVAILABLE = True
except ImportError:
    ADA_AVAILABLE = False
    # Use Python fallback (pure PyTorch step entropy)
```

## 3.3 Configuration Merge

```python
@dataclass
class SystemRouterConfig:
    """
    Unified configuration merging:
    - RouterConfig (from neural_router.py)
    - GarlicOrchestratorConfig (from garlic_orchestrator.py)
    """
    
    # ========== NEURAL ROUTING (from RouterConfig) ==========
    context_dim: int = 768
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    num_templates: int = 16
    num_tools: int = 32
    learning_rate: float = 1e-4
    dropout: float = 0.1
    weight_decay: float = 0.01
    
    # CRITICAL: ModelSlot replaces reasoning_levels
    num_model_slots: int = 3              # ← NEW (was: reasoning_levels: List[str])
    builtin_tools: List[str] = None       # default: ['browser', 'python', 'web_search']
    
    # ========== HSGM PIPELINE ==========
    enable_hsgm: bool = True
    segment_size: int = 32
    overlap_size: int = 8
    similarity_threshold: float = 0.7
    summary_ratio: float = 0.25
    
    # ========== STEP ENTROPY (Ada or Python) ==========
    enable_entropy_routing: bool = True
    ada_lib_path: Optional[str] = "./ada_bridge/lib/libgarlic_core.so"
    
    # Paper thresholds (§3.1):
    entropy_threshold_low: float = 2.0    # Fast path trigger
    entropy_threshold_high: float = 4.0   # Slow path / memory retrieval
    
    # DARE-inspired adaptive thresholds (LEARNABLE):
    use_adaptive_thresholds: bool = True  # ← SOTA++
    
    # ========== ENTROPY REGULARIZED ROUTER ==========
    router_entropy_threshold_low: float = 0.2   # Different scale than step entropy
    router_entropy_threshold_high: float = 0.8
    num_experts: int = 4
    top_k: int = 2
    temperature: float = 1.0
    lambda_entropy: float = 0.01
    
    # ========== REASONING SCAFFOLDER ==========
    enable_semantic_signals: bool = True
    signal_embedding_dim: int = 768
    
    # ========== GNN ROUTING ==========
    enable_gnn_routing: bool = False  # Requires torch_geometric
    
    # ========== GARLIC INJECTION ==========
    enable_garlic_injection: bool = True
    injection_scale_init: float = 0.1  # Learnable parameter init
    
    # SOTA++: Adaptive multi-layer injection
    use_adaptive_injection: bool = True  # ← NEW
    injection_strategy: str = 'entropy_aware'  # 'fixed' | 'entropy_aware'
    
    # ========== GLOBAL MEMORY ==========
    global_memory_max_turns: int = 1000
    global_memory_compression_ratio: float = 0.25
    
    # ========== MODEL LOADER ==========
    model_slot_descriptions: Dict[int, str] = None  # Runtime-defined
    
    def __post_init__(self):
        if self.builtin_tools is None:
            self.builtin_tools = ['browser', 'python', 'web_search']
        if self.model_slot_descriptions is None:
            self.model_slot_descriptions = {
                0: "fast-light",
                1: "general",
                2: "specialized-deep"
            }
```

---

# IV. SOTA++ INNOVATIONS

## 4.1 Beyond Papers — Novel Contributions

### 4.1.1 **DARE-Inspired Difficulty-Aware Routing**

**Paper:** DARE (ICLR 2026) — uses log-perplexity for dynamic expert allocation  
**Our Adaptation:** Use Step Entropy (Ada-computed) as difficulty proxy

```python
class DifficultyAwareExpertAllocator(nn.Module):
    """
    Adaptive expert count based on token difficulty (entropy).
    
    Key insight from DARE paper:
    - Fixed top-k wastes compute on easy tokens
    - Hard tokens need more experts
    - Use difficulty proxy (entropy) to decide k dynamically
    
    Our innovation:
    - Step Entropy (from Ada bridge) is the difficulty signal
    - Learnable thresholds (not hardcoded)
    - Maps: entropy → expert_count ∈ [1, num_experts]
    """
    
    def __init__(
        self,
        num_experts: int = 4,
        min_experts: int = 1,
        max_experts: int = 3,
        init_threshold_low: float = 2.0,
        init_threshold_high: float = 4.0
    ):
        super().__init__()
        self.num_experts = num_experts
        self.min_experts = min_experts
        self.max_experts = max_experts
        
        # Learnable thresholds (trained end-to-end)
        self.threshold_low = nn.Parameter(torch.tensor(init_threshold_low))
        self.threshold_high = nn.Parameter(torch.tensor(init_threshold_high))
        
        # GRPO reward compatibility (from Step Entropy paper Table 3)
        self.register_buffer('optimal_pruning_ratio', torch.tensor(0.80))
    
    def forward(
        self,
        entropy_values: torch.Tensor  # [batch, seq_len]
    ) -> torch.Tensor:
        """
        Returns expert_counts [batch, seq_len] — how many experts per token.
        
        Mapping (adaptive via learned thresholds):
          entropy < threshold_low  → min_experts (fast)
          threshold_low ≤ entropy < threshold_high → 2 experts
          entropy ≥ threshold_high → max_experts (slow, memory)
        """
        expert_counts = torch.ones_like(entropy_values, dtype=torch.long) * 2  # default
        
        # Low entropy → minimal experts
        expert_counts[entropy_values < self.threshold_low] = self.min_experts
        
        # High entropy → maximal experts + memory retrieval
        expert_counts[entropy_values >= self.threshold_high] = self.max_experts
        
        return expert_counts
    
    def get_routing_loss(self) -> torch.Tensor:
        """
        Regularization: encourage thresholds to stay ordered and bounded.
        """
        # Ensure threshold_low < threshold_high
        ordering_loss = F.relu(self.threshold_low - self.threshold_high + 0.5)
        
        # Bound thresholds to reasonable range [1.0, 6.0]
        bound_loss = (
            F.relu(1.0 - self.threshold_low) +
            F.relu(self.threshold_high - 6.0)
        )
        
        return ordering_loss + 0.1 * bound_loss
```

**Usage in SystemRouter:**
```python
# Stage 10: Adaptive expert allocation
difficulty_allocator = DifficultyAwareExpertAllocator()
expert_counts = difficulty_allocator(entropy_values)  # [B, S]

# Route to experts with dynamic k
routing_decision = self.entropy_router(
    hidden_states,
    graph_context,
    semantic_signals,
    expert_counts=expert_counts  # ← Dynamic k per token
)
```

### 4.1.2 **Multi-Scale HSGM Graph Construction**

**Problem:** Current HSGM builds one graph at segment_size=32  
**Enhancement:** Build 3 graphs at different scales for hierarchical injection

```python
class MultiScaleGraphBuilder(nn.Module):
    """
    Build HSGM graphs at 3 scales:
    - Fine (segment_size=16): capture local structure
    - Medium (segment_size=32): original HSGM
    - Coarse (segment_size=64): capture global structure
    
    Inject different scales at different layers:
    - Early layers: fine-grained
    - Middle layers: medium
    - Late layers: coarse (global summary)
    """
    
    def __init__(self, hidden_dim: int = 768):
        super().__init__()
        self.scales = {
            'fine': HSGMLocalGraphBuilder(LocalGraphConfig(segment_size=16, ...)),
            'medium': HSGMLocalGraphBuilder(LocalGraphConfig(segment_size=32, ...)),
            'coarse': HSGMLocalGraphBuilder(LocalGraphConfig(segment_size=64, ...))
        }
        self.summary_extractors = {
            scale: GNNSummaryExtractor(...) for scale in self.scales
        }
    
    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
        {
            'fine': compressed_graph_fine [B, N_fine, D],
            'medium': compressed_graph_medium [B, N_medium, D],
            'coarse': compressed_graph_coarse [B, N_coarse, D]
        }
        """
        multi_scale_graphs = {}
        
        for scale_name, builder in self.scales.items():
            local_graphs = builder(hidden_states)
            summary = self.summary_extractors[scale_name](
                local_graphs['node_embeddings'],
                local_graphs['edge_index'],
                local_graphs['edge_weights']
            )
            multi_scale_graphs[scale_name] = summary['summary_embeddings']
        
        return multi_scale_graphs
```

**Injection Strategy:**
```python
# SystemRouter forward()
multi_scale_graphs = self.multi_scale_graph_builder(hidden_states_augmented)

# Inject at different layers
GarlicInjectionLayer.set_context_multi_scale({
    'early_layers': multi_scale_graphs['fine'],
    'middle_layers': multi_scale_graphs['medium'],
    'late_layers': multi_scale_graphs['coarse']
})
```

### 4.1.3 **GNN-Based Expert Collaboration**

**Inspired by:** MoDE (AAAI-22) — mixture of domain-specific experts with disentangled features

```python
class GNNExpertCollaborationLayer(nn.Module):
    """
    Experts form a graph, collaborate via message passing.
    
    Graph structure:
    - Nodes = experts
    - Edges = co-activation history (learned)
    - Messages = expert outputs
    - Aggregation = weighted sum via GNN
    
    Why this matters:
    - Standard MoE: experts are independent (no collaboration)
    - GNN-MoE: experts can share information via graph structure
    - Emergent specialization patterns
    """
    
    def __init__(
        self,
        num_experts: int = 4,
        expert_dim: int = 768,
        num_gnn_layers: int = 2
    ):
        super().__init__()
        self.num_experts = num_experts
        
        # Expert graph structure (learned)
        self.expert_adjacency = nn.Parameter(
            torch.eye(num_experts)  # Init as independent, learn structure
        )
        
        # GNN layers for message passing
        from torch_geometric.nn import GATConv
        self.gnn_layers = nn.ModuleList([
            GATConv(expert_dim, expert_dim, heads=4)
            for _ in range(num_gnn_layers)
        ])
    
    def forward(
        self,
        expert_outputs: torch.Tensor,  # [batch, num_experts, dim]
        expert_weights: torch.Tensor   # [batch, num_experts]
    ) -> torch.Tensor:
        """
        Expert collaboration via GNN message passing.
        
        Returns: collaborated_output [batch, dim]
        """
        batch_size = expert_outputs.shape[0]
        
        # Build expert graph (same for all samples in batch)
        # Edge weights from learned adjacency
        edge_weights = torch.sigmoid(self.expert_adjacency)
        edge_index = torch.nonzero(edge_weights > 0.1, as_tuple=False).t()
        
        # Message passing
        for gnn_layer in self.gnn_layers:
            # Flatten batch for GNN processing
            x = expert_outputs.view(-1, expert_outputs.shape[-1])  # [B*E, D]
            
            # Expand edge_index for batch (repeat for each sample)
            batch_edge_index = []
            for b in range(batch_size):
                batch_edge_index.append(edge_index + b * self.num_experts)
            batch_edge_index = torch.cat(batch_edge_index, dim=1)
            
            # GNN forward
            x = gnn_layer(x, batch_edge_index)
            expert_outputs = x.view(batch_size, self.num_experts, -1)
        
        # Weighted aggregation
        collaborated_output = torch.einsum(
            'be,bed->bd',
            expert_weights,
            expert_outputs
        )
        
        return collaborated_output
```

### 4.1.4 **Adaptive Attention Control via Injection**

**Novel idea:** Don't just inject at fixed layers — control WHEN and HOW MUCH based on entropy

```python
class EntropyAwareInjectionController(nn.Module):
    """
    Dynamically control injection strength and layer selection
    based on routing entropy.
    
    Key insight:
    - Low entropy queries: model is confident → minimal injection
    - High entropy: model uncertain → maximum augmentation
    
    Controls:
    1. Which layers to inject (layer_mask)
    2. Injection strength per layer (layer_scales)
    3. Memory vs signal weighting (context_balance)
    """
    
    def __init__(
        self,
        num_layers: int = 32,
        hidden_dim: int = 768
    ):
        super().__init__()
        self.num_layers = num_layers
        
        # Learned mapping: entropy → layer mask
        self.entropy_to_layer_mask = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, num_layers),
            nn.Sigmoid()  # 0-1 per layer
        )
        
        # Learned injection scaling
        self.entropy_to_scale = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        entropy_value: float,
        num_layers: int
    ) -> Dict[str, torch.Tensor]:
        """
        Returns injection control signals.
        
        Output:
        {
            'layer_mask': [num_layers] — binary mask (inject or skip)
            'injection_scale': scalar — overall strength multiplier
            'layer_scales': [num_layers] — per-layer scaling
        }
        """
        entropy_tensor = torch.tensor([[entropy_value]])
        
        # Compute layer mask
        layer_probs = self.entropy_to_layer_mask(entropy_tensor).squeeze(0)
        
        # Threshold to binary mask
        # Low entropy → inject only late layers
        # High entropy → inject all layers
        if entropy_value < 2.0:
            # Fast path: only last layer
            layer_mask = torch.zeros(num_layers)
            layer_mask[-1] = 1.0
        elif entropy_value < 4.0:
            # Normal: top-k layers by probability
            k = num_layers // 3
            topk_indices = torch.topk(layer_probs, k).indices
            layer_mask = torch.zeros(num_layers)
            layer_mask[topk_indices] = 1.0
        else:
            # Slow: all layers with learned probabilities
            layer_mask = (layer_probs > 0.3).float()
        
        # Overall scaling
        injection_scale = self.entropy_to_scale(entropy_tensor).item()
        
        # Per-layer scaling (modulated by entropy)
        layer_scales = layer_mask * layer_probs * injection_scale
        
        return {
            'layer_mask': layer_mask,
            'injection_scale': injection_scale,
            'layer_scales': layer_scales
        }
```

**Usage:**
```python
# In SystemRouter
injection_controller = EntropyAwareInjectionController(num_layers=32)
injection_config = injection_controller(entropy_value, num_layers)

# Set dynamic injection config
GarlicInjectionLayer.set_injection_config(injection_config)

# Forward pass through model — injection happens at selected layers only
outputs = model.generate(...)
```

---

# V. IMPLEMENTATION CHECKLIST

## 5.1 Phase 1: Foundation — Rename & Import Fix
**Est. Time:** 2-3 hours | **Lines Changed:** ~100

### ✅ Checklist:

- [ ] **1.1** Rename `ReasoningEffort` enum → `ModelSlot`
  - File: `neural_router.py` line 79
  - Change: `class ReasoningEffort(Enum):` → `class ModelSlot(Enum):`
  - Values: `LOW=0, MEDIUM=1, HIGH=2` → `SLOT_A=0, SLOT_B=1, SLOT_C=2`

- [ ] **1.2** Update `SlotPredictions` dataclass
  - File: `neural_router.py` line 86
  - Change: `reasoning_effort: torch.Tensor` → `model_slot: torch.Tensor`
  - Keep shape `[batch, 3]` identical

- [ ] **1.3** Rename neural head
  - File: `neural_router.py` line 183-189
  - Change: `self.reasoning_head` → `self.model_selector_head`
  - Architecture stays identical (MLP: dim → 256 → 128 → 3)

- [ ] **1.4** Update `RouterConfig`
  - File: `neural_router.py` line 59
  - Remove: `reasoning_levels: List[str] = None`
  - Add: `num_model_slots: int = 3`
  - Remove `__post_init__` reasoning_levels default

- [ ] **1.5** Update `TemplateSelectorNetwork.flatten_slots()`
  - File: `neural_router.py` line 297-308
  - Change: `components = [slot_preds.reasoning_effort]` → `components = [slot_preds.model_slot]`
  - Dimension stays: `slot_dim = 3 + len(...) + config.num_tools`

- [ ] **1.6** Create `SystemOutput` dataclass
  ```python
  @dataclass
  class SystemOutput:
      prompt: str
      model_slot: ModelSlot
      routing_decision: Optional[RoutingDecision] = None
      entropy_level: Optional[str] = None
      entropy_value: float = 0.0
      compression_ratio: float = 1.0
      original_tokens: int = 0
      compressed_tokens: int = 0
      augmented_states: Optional[torch.Tensor] = None
      hsgm_summary: Optional[torch.Tensor] = None
      trace: Optional[Dict[str, Any]] = None
  ```

- [ ] **1.7** Create `ModelSlotConfig` dataclass
  ```python
  @dataclass
  class ModelSlotConfig:
      slot_id: int
      description: str = ""
      model_path: Optional[str] = None
      is_loaded: bool = False
      metadata: Dict[str, Any] = field(default_factory=dict)
  ```

- [ ] **1.8** Create `ModelLoader` class (see §III.1 for full implementation)

- [ ] **1.9** Update `SafeRouterWrapper` return type
  - Change: `Tuple[str, Dict]` → `SystemOutput`
  - Add `.meta` property for legacy compatibility

### 🧪 Test:
```bash
python -m pytest tests/test_phase1_rename.py -v
```

---

## 5.2 Phase 2: Ada Bridge Integration
**Est. Time:** 1-2 hours | **Lines New:** ~150

### ✅ Checklist:

- [ ] **2.1** Move ADA-Step-Entropy folder
  ```bash
  mv ~/Projects/ADA-Step-Entropy/ada_bridge ~/Projects/System-Router/
  mv ~/Projects/ADA-Step-Entropy/lib ~/Projects/System-Router/ada_bridge/
  ```

- [ ] **2.2** Create `StepEntropyInterface` protocol
  ```python
  from typing import Protocol, runtime_checkable
  
  @runtime_checkable
  class StepEntropyInterface(Protocol):
      def calculate_batch_step_entropy(
          self,
          token_logits: torch.Tensor,
          token_ids: torch.Tensor
      ) -> torch.Tensor:
          ...
  ```

- [ ] **2.3** Verify Ada bridge imports
  ```python
  try:
      from ada_bridge.ada_bridge import GarlicAdaStepEntropy
      ADA_AVAILABLE = True
  except ImportError:
      ADA_AVAILABLE = False
      logger.warning("Ada bridge not available, using Python fallback")
  ```

- [ ] **2.4** Create `build_step_entropy()` factory
  ```python
  def build_step_entropy(
      ada_lib_path: Optional[str] = None,
      fallback_config: Optional[Dict] = None
  ) -> StepEntropyInterface:
      if ada_lib_path and ADA_AVAILABLE:
          bridge = GarlicAdaStepEntropy(lib_path=ada_lib_path)
          if bridge.is_available:
              return bridge
      # Fallback to Python
      return StepEntropyCalculator(...)
  ```

- [ ] **2.5** Add to `SystemRouterConfig`
  ```python
  ada_lib_path: Optional[str] = "./ada_bridge/lib/libgarlic_core.so"
  use_adaptive_thresholds: bool = True
  ```

- [ ] **2.6** Wire into `SystemRouter.__init__()`
  ```python
  self.step_entropy = build_step_entropy(
      ada_lib_path=config.ada_lib_path,
      fallback_config={'entropy_threshold_low': ..., ...}
  )
  ```

### 🧪 Test:
```bash
python -m pytest ada_bridge/tests/test_ada_bridge.py -v  # Should pass 16/16
python -m pytest tests/test_step_entropy_interface.py -v
```

---

## 5.3 Phase 3: GlobalGraphMemory
**Est. Time:** 2-3 hours | **Lines New:** ~200

### ✅ Checklist:

- [ ] **3.1** Create `TurnRecord` dataclass
  ```python
  @dataclass
  class TurnRecord:
      turn_id: int
      input_graph: torch.Tensor
      reasoning_graph: Optional[torch.Tensor]
      output_graph: Optional[torch.Tensor]
      entropy_value: float
      model_slot: ModelSlot
      timestamp: float = field(default_factory=lambda: time.time())
  ```

- [ ] **3.2** Implement `GlobalGraphMemory` class (see §II.2.2 Innovation 2)

- [ ] **3.3** Add config fields
  ```python
  global_memory_max_turns: int = 1000
  global_memory_compression_ratio: float = 0.25
  ```

- [ ] **3.4** Initialize in `SystemRouter.__init__()`
  ```python
  self.global_memory = GlobalGraphMemory(
      hidden_dim=config.context_dim,
      max_turns=config.global_memory_max_turns
  )
  ```

- [ ] **3.5** Wire into forward pass — Stage 1
  ```python
  # STAGE 1: prepend history
  history_context = self.global_memory.get_context(device=message_embs.device)
  if history_context is not None:
      hidden_states = torch.cat([history_context, hidden_states], dim=1)
  ```

- [ ] **3.6** Wire into forward pass — Stage 13
  ```python
  # STAGE 13: update memory
  if compressed_context is not None:
      self.global_memory.add_turn(
          input_graph=compressed_context,
          entropy_value=entropy_value,
          model_slot=active_slot
      )
  ```

- [ ] **3.7** Add `.flush()` to session cleanup
  ```python
  def shutdown_session(self):
      self.global_memory.flush()
      logger.info("Session ended, GlobalGraphMemory flushed")
  ```

### 🧪 Test:
```bash
python -m pytest tests/test_global_memory.py -v
# Test: multi-turn accumulation, compression stability, flush
```

---

## 5.4 Phase 4: Component Internalization
**Est. Time:** 4-6 hours | **Lines Moved:** ~1500

### ✅ Checklist:

- [ ] **4.1** Copy HSGM components into `system_router.py`
  - [ ] `HSGMLocalGraphBuilder` from `hsgm_local_graph_builder.py`
  - [ ] `GNNSummaryExtractor` from `gnn_summary_extractor.py`
  - [ ] `HierarchicalQueryProcessor` from `hierarchical_query_processor.py`
  - [ ] All config classes: `LocalGraphConfig`, `SummaryExtractionConfig`, etc.

- [ ] **4.2** Copy Reasoning Scaffolder
  - [ ] `ReasoningScaffolder` from `reasoning_scaffolder_prod.py`
  - [ ] `SemanticSignal` enum

- [ ] **4.3** Copy Entropy Router
  - [ ] `EntropyRegularizedRouter` from `entropy_regularized_router.py`
  - [ ] `RoutingDecision` dataclass

- [ ] **4.4** Copy Garlic Bridge components
  - [ ] From `garlic_orchestrator.py`:
    - [ ] `garlic_bridge` Linear layer
    - [ ] `signal_projection` Linear layer
    - [ ] `inject_memory_garlic_bridge()` method

- [ ] **4.5** Create `SystemRouterConfig` (merged config)
  - Merge fields from `RouterConfig` + `GarlicOrchestratorConfig`
  - Add SOTA++ fields (adaptive injection, etc.)

- [ ] **4.6** Implement `SystemRouter.__init__()`
  - Initialize all components
  - Conditional initialization based on config flags
  - Example:
    ```python
    if config.enable_hsgm:
        self.local_graph_builder = HSGMLocalGraphBuilder(...)
        self.summary_extractor = GNNSummaryExtractor(...)
        self.query_processor = HierarchicalQueryProcessor(...)
    
    if config.enable_semantic_signals:
        self.reasoning_scaffolder = ReasoningScaffolder(...)
    ```

- [ ] **4.7** Implement `SystemRouter.forward()` — 13 stages
  - See §I.2 for complete flow
  - Each stage is a method call
  - Progressive trace building

- [ ] **4.8** Remove `garlic_orchestrator.py`
  - Move to `_archive/garlic_orchestrator_legacy.py`
  - All logic now in `system_router.py`

### 🧪 Test:
```bash
python -m pytest tests/test_system_router_forward.py -v
# Test: all 13 stages execute, trace is complete
```

---

## 5.5 Phase 5: SOTA++ Innovations
**Est. Time:** 6-8 hours | **Lines New:** ~800

### ✅ Checklist:

- [ ] **5.1** Implement `DifficultyAwareExpertAllocator` (see §IV.1.1)
  - [ ] Learnable thresholds
  - [ ] Adaptive k selection
  - [ ] Regularization loss

- [ ] **5.2** Implement `MultiScaleGraphBuilder` (see §IV.1.2)
  - [ ] Fine/medium/coarse scales
  - [ ] Per-scale summary extraction
  - [ ] Multi-scale injection strategy

- [ ] **5.3** Implement `GNNExpertCollaborationLayer` (see §IV.1.3)
  - [ ] Learnable expert graph adjacency
  - [ ] GNN message passing
  - [ ] Collaborative output aggregation

- [ ] **5.4** Implement `EntropyAwareInjectionController` (see §IV.1.4)
  - [ ] Entropy → layer mask
  - [ ] Dynamic injection scaling
  - [ ] Integration with `GarlicInjectionLayer`

- [ ] **5.5** Add GRPO rewards integration
  - [ ] Import from Ada bridge: `compute_grpo_rewards()`
  - [ ] Wire into training loss
  - [ ] Track per-slot rewards over time

- [ ] **5.6** Add adaptive threshold training
  - [ ] `AdaptiveThresholdManager` from Ada bridge
  - [ ] Update thresholds every N steps
  - [ ] Log threshold drift

- [ ] **5.7** Extend `SystemOutput` with SOTA++ metrics
  ```python
  @dataclass
  class SystemOutput:
      # ... existing fields ...
      expert_allocation: Optional[torch.Tensor] = None  # [B, S] expert counts
      injection_layers: Optional[List[int]] = None      # Which layers injected
      grpo_reward: Optional[float] = None               # GRPO score
      graph_scales_used: Optional[List[str]] = None     # ['fine', 'medium', 'coarse']
  ```

### 🧪 Test:
```bash
python -m pytest tests/test_sota_innovations.py -v
# Test: adaptive allocation, multi-scale graphs, GNN collaboration
```

---

## 5.6 Phase 6: Training Infrastructure
**Est. Time:** 3-4 hours | **Lines New:** ~400

### ✅ Checklist:

- [ ] **6.1** Update `RouterTrainer.compute_loss()`
  - [ ] Add difficulty-aware loss
    ```python
    # Adaptive threshold regularization
    threshold_loss = self.difficulty_allocator.get_routing_loss()
    ```
  - [ ] Add expert collaboration loss
    ```python
    # GNN expert graph regularization
    expert_graph_loss = self.expert_collab.get_graph_sparsity_loss()
    ```
  - [ ] Add GRPO rewards
    ```python
    # Reward signal from Ada bridge
    grpo_rewards = self.ada_bridge.compute_grpo_rewards(
        entropy_history, pruning_history
    )
    reward_loss = -grpo_rewards.mean()
    ```

- [ ] **6.2** Add curriculum learning schedule
  ```python
  class CurriculumScheduler:
      """
      Gradually increase difficulty over training.
      
      Early training: simple routing (high entropy threshold)
      Late training: complex routing (low threshold)
      """
      def get_threshold_multipliers(self, epoch: int) -> Dict[str, float]:
          progress = epoch / max_epochs
          return {
              'entropy_threshold_low': 2.0 * (1 + 0.5 * (1 - progress)),
              'entropy_threshold_high': 4.0 * (1 + 0.5 * (1 - progress))
          }
  ```

- [ ] **6.3** Add expert specialization metrics
  ```python
  def track_expert_specialization(
      expert_activations: torch.Tensor,  # [num_samples, num_experts]
      task_labels: torch.Tensor          # [num_samples]
  ) -> Dict[int, Dict]:
      """
      Track which experts specialize for which task types.
      
      Returns: {expert_id: {'tasks': [...], 'activation_freq': ...}}
      """
  ```

- [ ] **6.4** Add checkpoint saving with full state
  ```python
  def save_checkpoint(self, path: str):
      torch.save({
          'system_router': self.system_router.state_dict(),
          'model_loader_slots': self.model_loader.get_slot_info(),
          'global_memory': self.global_memory.turn_history,
          'adaptive_thresholds': self.step_entropy.get_threshold_state(),
          'expert_graph': self.expert_collab.expert_adjacency.data,
          'training_metrics': self.metrics
      }, path)
  ```

### 🧪 Test:
```bash
python -m pytest tests/test_training.py -v
# Test: loss computation, curriculum, checkpointing
```

---

## 5.7 Phase 7: Integration Testing
**Est. Time:** 2-3 hours

### ✅ Checklist:

- [ ] **7.1** End-to-end inference test
  ```python
  def test_full_inference_pipeline():
      # Setup
      config = SystemRouterConfig(...)
      router = SystemRouter(config)
      
      # Load models into slots
      router.model_loader.register_slot(0, "fast", "/models/qwen-1.5b")
      router.model_loader.register_slot(1, "general", "/models/qwen-7b")
      router.model_loader.register_slot(2, "reasoning", "/models/qwen3-cherry")
      
      # Inference
      context = {...}
      output = router(
          message_embs=...,
          user_profile=...,
          metadata=...,
          context_metadata=context,
          hidden_states=...,
          logits=...,
          return_trace=True
      )
      
      # Assertions
      assert isinstance(output, SystemOutput)
      assert output.model_slot in [ModelSlot.SLOT_A, ModelSlot.SLOT_B, ModelSlot.SLOT_C]
      assert output.compression_ratio > 1.0
      assert output.trace is not None
  ```

- [ ] **7.2** Multi-turn session test
  ```python
  def test_multi_turn_session():
      router = SystemRouter(...)
      
      # Turn 1
      out1 = router(...)
      assert router.global_memory.turn_count == 1
      
      # Turn 2 (should use accumulated graph)
      out2 = router(...)
      assert router.global_memory.turn_count == 2
      assert out2.compression_ratio > out1.compression_ratio  # Better compression
      
      # Flush
      router.shutdown_session()
      assert router.global_memory.is_empty
  ```

- [ ] **7.3** Ada bridge validation test
  ```python
  def test_ada_bridge_integration():
      if not ADA_AVAILABLE:
          pytest.skip("Ada bridge not available")
      
      router = SystemRouter(SystemRouterConfig(
          ada_lib_path="./ada_bridge/lib/libgarlic_core.so"
      ))
      
      # Generate logits
      logits = torch.randn(1, 10, 50257)
      token_ids = torch.randint(0, 50257, (1, 10))
      
      # Calculate entropy via Ada
      entropy = router.step_entropy.calculate_batch_step_entropy(logits, token_ids)
      
      assert entropy.shape == (1,)
      assert 0.0 <= entropy.item() <= 20.0  # Reasonable bounds
  ```

- [ ] **7.4** Memory injection test
  ```python
  def test_memory_injection_lifecycle():
      router = SystemRouter(...)
      
      # Generate dummy graphs and signals
      graph_memory = torch.randn(1, 5, 768)
      semantic_signals = torch.randn(1, 10, 768)
      
      # Set context
      GarlicInjectionLayer.set_context(graph_memory, semantic_signals)
      
      # Forward pass
      hidden = torch.randn(1, 10, 768)
      layer = GarlicInjectionLayer(hidden_dim=768, layer_type='middle')
      augmented, = layer(hidden)
      
      # Verify injection happened
      assert not torch.allclose(augmented, hidden)
      
      # Clear
      GarlicInjectionLayer.clear_context()
  ```

### 🧪 Test:
```bash
python -m pytest tests/test_integration.py -v --cov=system_router
```

---

# VI. CODE PATTERNS & SNIPPETS

## 6.1 Complete SystemRouter Skeleton

```python
"""
system_router.py — The Unified System Router
Merges: neural_router.py + Garlic-Components/ + Ada Step Entropy

Total: ~3500-4500 lines
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Protocol, runtime_checkable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import re
import math
import json

logger = logging.getLogger(__name__)

# ============================================================================
# IMPORTS — Ada Bridge (Optional)
# ============================================================================

try:
    from ada_bridge.ada_bridge import GarlicAdaStepEntropy, AdaptiveThresholdManager
    ADA_AVAILABLE = True
    logger.info("Ada Step Entropy bridge loaded successfully")
except ImportError:
    ADA_AVAILABLE = False
    logger.warning("Ada bridge not available — falling back to Python step entropy")


# ============================================================================
# ENUMS & DATACLASSES
# ============================================================================

class ModelSlot(Enum):
    """
    Canon-agnostic model slot indices.
    
    NOT reasoning effort levels (low/medium/high).
    Semantics are user-defined at runtime via ModelLoader.
    
    Typical usage:
        SLOT_A = fast/light model
        SLOT_B = general model
        SLOT_C = specialized/deep model
    
    But these are NOT hardcoded — router learns from context.
    """
    SLOT_A = 0
    SLOT_B = 1
    SLOT_C = 2


@dataclass
class ModelSlotConfig:
    """Runtime configuration for a single model slot."""
    slot_id: int
    description: str = ""
    model_path: Optional[str] = None
    is_loaded: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlotPredictions:
    """Output from SlotPredictorNetwork."""
    model_slot: torch.Tensor         # [batch, 3] — RENAMED from reasoning_effort
    tool_enables: Dict[str, torch.Tensor]  # {tool_name: [batch, 1]}
    tool_weights: torch.Tensor       # [batch, num_tools]
    confidence: float = 0.0


@dataclass
class SystemOutput:
    """
    Unified output from SystemRouter.forward().
    
    Replaces:
    - Tuple[str, Dict] from neural_router.py
    - GarlicOutput from garlic_orchestrator.py
    """
    # Core outputs
    prompt: str
    model_slot: ModelSlot
    
    # Routing metadata
    routing_decision: Optional['RoutingDecision'] = None
    entropy_level: Optional[str] = None  # 'fast'/'normal'/'slow'
    entropy_value: float = 0.0
    
    # HSGM compression
    compression_ratio: float = 1.0
    original_tokens: int = 0
    compressed_tokens: int = 0
    
    # Hidden state augmentation
    augmented_states: Optional[torch.Tensor] = None
    hsgm_summary: Optional[torch.Tensor] = None
    
    # SOTA++ metrics
    expert_allocation: Optional[torch.Tensor] = None
    injection_layers: Optional[List[int]] = None
    grpo_reward: Optional[float] = None
    graph_scales_used: Optional[List[str]] = None
    
    # Trace (full pipeline when return_trace=True)
    trace: Optional[Dict[str, Any]] = None
    
    @property
    def meta(self) -> Dict[str, Any]:
        """Legacy accessor for SafeRouterWrapper compatibility."""
        return {
            'model_slot': self.model_slot.name,
            'entropy_level': self.entropy_level,
            'entropy_value': self.entropy_value,
            'compression_ratio': self.compression_ratio,
            'trace': self.trace
        }


@dataclass
class TurnRecord:
    """Single turn in GlobalGraphMemory."""
    turn_id: int
    input_graph: torch.Tensor
    reasoning_graph: Optional[torch.Tensor]
    output_graph: Optional[torch.Tensor]
    entropy_value: float
    model_slot: ModelSlot
    timestamp: float = field(default_factory=lambda: __import__('time').time())


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SystemRouterConfig:
    """
    Unified configuration merging:
    - RouterConfig (neural_router.py)
    - GarlicOrchestratorConfig (garlic_orchestrator.py)
    """
    
    # ========== NEURAL ROUTING ==========
    context_dim: int = 768
    num_transformer_layers: int = 4
    num_attention_heads: int = 8
    num_templates: int = 16
    num_tools: int = 32
    learning_rate: float = 1e-4
    dropout: float = 0.1
    weight_decay: float = 0.01
    
    # CRITICAL: ModelSlot replaces reasoning_levels
    num_model_slots: int = 3
    builtin_tools: List[str] = None
    
    # ========== HSGM PIPELINE ==========
    enable_hsgm: bool = True
    segment_size: int = 32
    overlap_size: int = 8
    similarity_threshold: float = 0.7
    summary_ratio: float = 0.25
    
    # ========== STEP ENTROPY ==========
    enable_entropy_routing: bool = True
    ada_lib_path: Optional[str] = "./ada_bridge/lib/libgarlic_core.so"
    entropy_threshold_low: float = 2.0
    entropy_threshold_high: float = 4.0
    use_adaptive_thresholds: bool = True
    
    # ========== ENTROPY ROUTER ==========
    router_entropy_threshold_low: float = 0.2
    router_entropy_threshold_high: float = 0.8
    num_experts: int = 4
    top_k: int = 2
    temperature: float = 1.0
    lambda_entropy: float = 0.01
    
    # ========== REASONING SCAFFOLDER ==========
    enable_semantic_signals: bool = True
    
    # ========== GNN ROUTING ==========
    enable_gnn_routing: bool = False
    
    # ========== GARLIC INJECTION ==========
    enable_garlic_injection: bool = True
    injection_scale_init: float = 0.1
    use_adaptive_injection: bool = True
    injection_strategy: str = 'entropy_aware'
    
    # ========== GLOBAL MEMORY ==========
    global_memory_max_turns: int = 1000
    global_memory_compression_ratio: float = 0.25
    
    # ========== SOTA++ ==========
    use_difficulty_aware_allocation: bool = True
    use_multi_scale_graphs: bool = True
    use_gnn_expert_collab: bool = False
    
    def __post_init__(self):
        if self.builtin_tools is None:
            self.builtin_tools = ['browser', 'python', 'web_search']


# ============================================================================
# [INHERITED FROM neural_router.py — KEEP VERBATIM]
# ============================================================================

class HashTextEncoder(nn.Module):
    """Lines 405-554 from neural_router.py — production hash encoder"""
    # ... EXACT COPY ...
    pass


class ProfileEncoder(nn.Module):
    """Lines 557-704 from neural_router.py — behavioral features"""
    # ... EXACT COPY ...
    pass


class MetadataEncoder(nn.Module):
    """Lines 707-805 from neural_router.py — temporal + flags"""
    # ... EXACT COPY ...
    pass


class InputPreparer(nn.Module):
    """Lines 808-986 from neural_router.py — unified prep"""
    # ... EXACT COPY ...
    pass


class ContextEncoder(nn.Module):
    """Lines 101-171 from neural_router.py — 4-layer transformer"""
    # ... EXACT COPY ...
    pass


class TemplateSelectorNetwork(nn.Module):
    """
    Template selection via slot predictions.
    
    UPDATED: flatten_slots() now uses model_slot instead of reasoning_effort
    """
    def __init__(self, config: SystemRouterConfig):
        super().__init__()
        self.config = config
        
        # Slot dimension: 3 (model_slot) + num_tools + tool_enables
        slot_dim = (
            3 +  # model_slot (was reasoning_effort)
            len(config.builtin_tools) +
            config.num_tools
        )
        
        self.gate_network = nn.Sequential(
            nn.Linear(slot_dim, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(128, config.num_templates)
        )
    
    def flatten_slots(self, slot_preds: SlotPredictions) -> torch.Tensor:
        """Flatten slot predictions into single vector."""
        components = [slot_preds.model_slot]  # ← UPDATED
        
        for tool_name in self.config.builtin_tools:
            components.append(slot_preds.tool_enables[tool_name])
        
        components.append(slot_preds.tool_weights)
        
        return torch.cat(components, dim=-1)
    
    def forward(self, slot_preds: SlotPredictions) -> torch.Tensor:
        slot_vector = self.flatten_slots(slot_preds)
        logits = self.gate_network(slot_vector)
        weights = F.softmax(logits, dim=-1)
        return weights


class SafetyValidator:
    """Constraint enforcement — exact copy from neural_router.py"""
    # ... EXACT COPY ...
    pass


class TemplateLibrary:
    """Lines 987-1590 from neural_router.py — jinja2 + memory XML"""
    # ... EXACT COPY ...
    pass


# ============================================================================
# [INTERNALIZED FROM Garlic-Components/]
# ============================================================================

# Copy HSGMLocalGraphBuilder, GNNSummaryExtractor, HierarchicalQueryProcessor
# from hsgm_local_graph_builder.py, gnn_summary_extractor.py, hierarchical_query_processor.py

# Copy ReasoningScaffolder from reasoning_scaffolder_prod.py

# Copy EntropyRegularizedRouter from entropy_regularized_router.py


# ============================================================================
# [NEW COMPONENTS]
# ============================================================================

class ModelLoader:
    """
    Manages 3 agnostic model slots for dynamic routing.
    
    No hardcoded model names or reasoning effort semantics.
    Each slot is a generic container — load anything, route anything.
    """
    # ... FULL IMPLEMENTATION from §II.2.2 Innovation 1 ...
    pass


class GlobalGraphMemory:
    """
    Persistent multi-turn HSGM state manager.
    Never flushes until session shutdown.
    """
    # ... FULL IMPLEMENTATION from §II.2.2 Innovation 2 ...
    pass


@runtime_checkable
class StepEntropyInterface(Protocol):
    """Protocol for step entropy calculation (Ada or Python)."""
    def calculate_batch_step_entropy(
        self,
        token_logits: torch.Tensor,
        token_ids: torch.Tensor
    ) -> torch.Tensor:
        ...


def build_step_entropy(
    ada_lib_path: Optional[str] = None,
    fallback_config: Optional[Dict] = None
) -> StepEntropyInterface:
    """Factory: Ada bridge if available, else Python."""
    if ada_lib_path and ADA_AVAILABLE:
        try:
            bridge = GarlicAdaStepEntropy(lib_path=ada_lib_path)
            logger.info("Using Ada Step Entropy (native compiled)")
            return bridge
        except Exception as e:
            logger.warning(f"Ada bridge failed to load: {e}, using Python fallback")
    
    # Python fallback
    from step_entropy import StepEntropyCalculator  # Assumes original Python version available
    cfg = fallback_config or {}
    return StepEntropyCalculator(
        entropy_threshold_low=cfg.get('entropy_threshold_low', 2.0),
        entropy_threshold_high=cfg.get('entropy_threshold_high', 4.0)
    )


# ============================================================================
# MAIN SYSTEM ROUTER
# ============================================================================

class SystemRouter(nn.Module):
    """
    Unified system router — orchestrator collapsed into router.
    
    13-stage forward pass integrating:
    - Neural routing (template + slot selection)
    - HSGM compression
    - Step Entropy (Ada native)
    - Reasoning Scaffolder
    - Entropy Regularized Router
    - Garlic Bridge injection
    - GlobalGraphMemory persistence
    """
    
    def __init__(self, config: SystemRouterConfig):
        super().__init__()
        self.config = config
        
        # -- Inherited neural routing components --
        self.context_encoder = ContextEncoder(config)
        self.slot_predictor = SlotPredictorNetwork(config)  # Updated below
        self.template_selector = TemplateSelectorNetwork(config)
        self.safety_validator = SafetyValidator(config)
        self.template_library = TemplateLibrary(config)
        
        self.tool_embeddings = nn.Parameter(
            torch.randn(config.num_tools, config.context_dim)
        )
        
        # -- Model loader (NEW) --
        self.model_loader = ModelLoader(num_slots=config.num_model_slots)
        
        # -- HSGM Pipeline --
        if config.enable_hsgm:
            self.local_graph_builder = HSGMLocalGraphBuilder(...)
            self.summary_extractor = GNNSummaryExtractor(...)
            self.query_processor = HierarchicalQueryProcessor(...)
        
        # -- Reasoning Scaffolder --
        if config.enable_semantic_signals:
            self.reasoning_scaffolder = ReasoningScaffolder(hidden_dim=config.context_dim)
        
        # -- Entropy Regularized Router --
        self.entropy_router = EntropyRegularizedRouter(...)
        
        # -- Garlic Bridge --
        if config.enable_garlic_injection:
            self.garlic_bridge = nn.Linear(config.context_dim * 2, config.context_dim)
            self.signal_projection = nn.Linear(config.context_dim, config.context_dim)
        
        # -- Step Entropy (Ada or Python) --
        if config.enable_entropy_routing:
            self.step_entropy = build_step_entropy(
                ada_lib_path=config.ada_lib_path,
                fallback_config={
                    'entropy_threshold_low': config.entropy_threshold_low,
                    'entropy_threshold_high': config.entropy_threshold_high
                }
            )
        
        # -- Global Memory --
        self.global_memory = GlobalGraphMemory(
            hidden_dim=config.context_dim,
            max_turns=config.global_memory_max_turns
        )
        
        logger.info("SystemRouter initialized — orchestrator fully internalized")
    
    def forward(
        self,
        message_embs: torch.Tensor,
        user_profile: torch.Tensor,
        metadata: torch.Tensor,
        context_metadata: Dict[str, Any],
        hidden_states: Optional[torch.Tensor] = None,
        logits: Optional[torch.Tensor] = None,
        return_trace: bool = False
    ) -> SystemOutput:
        """
        Full 13-stage forward pass.
        See §I.2 for complete flow diagram.
        """
        trace = {} if return_trace else None
        
        # === STAGE 1-13: Implementation here ===
        # (See §I.2 for full breakdown)
        
        # ... [IMPLEMENTATION OF ALL 13 STAGES] ...
        
        return SystemOutput(
            prompt=prompt,
            model_slot=active_slot,
            routing_decision=routing_decision,
            entropy_level=entropy_level_str,
            entropy_value=entropy_value,
            compression_ratio=compression_ratio,
            augmented_states=augmented_states,
            hsgm_summary=compressed_context,
            trace=trace
        )


# ============================================================================
# PRODUCTION WRAPPER
# ============================================================================

class SystemRouterWrapper:
    """
    Production wrapper with fallback to Jinja2.
    
    Extended from SafeRouterWrapper in neural_router.py.
    Now returns SystemOutput instead of Tuple[str, Dict].
    """
    # ... IMPLEMENTATION ...
    pass
```

---

# VII. PAPER COMPLIANCE MATRIX

## 7.1 Paper Implementation Audit

| Paper | Component | Status | Compliance | Notes |
|-------|-----------|--------|------------|-------|
| **Step Entropy** (arXiv:2508.03346) | Ada FFI Bridge | ✅ COMPLETE | 100% | 16/16 tests passing, SOTA+++ features |
| | Shannon Entropy Formula | ✅ | 100% | `H = -Σ p log p` |
| | Routing Thresholds | ✅ | 100% | K_SKIP_LOW=0.50, K_SKIP_HIGH=0.80 |
| | [SKIP] Token Injection | ✅ | 100% | `compress_chain()` method |
| | GRPO Rewards | ✅ | 100% | Eq. 13-15, Table 3 exact values |
| | Adaptive Thresholds | ✅ SOTA++ | 120% | Welford online, not in paper |
| | Entropy Histogram | ✅ SOTA++ | 120% | Configurable bins, not in paper |
| **HSGM** | Local Graph Builder | ✅ | 100% | Segment-based graph construction |
| | GNN Summary Extractor | ✅ | 100% | GAT-based pooling |
| | Hierarchical Query Processor | ✅ | 100% | Multi-level retrieval |
| | Multi-Scale Graphs | 🔲 SOTA++ | 150% | Fine/medium/coarse scales (novel) |
| **Reasoning Scaffolding** (arXiv:2509.23619) | Semantic Signals | ✅ | 100% | 6 signal types |
| | Signal Embedding | ✅ | 100% | Project to hidden_dim |
| | Signal Injection | ✅ | 100% | Additive to hidden states |
| | Meta-Signal | 🔲 SOTA++ | 120% | METACOGNITIVE signal (novel) |
| **Entropy Router** (Pangu 5.5) | Sample Entropy Min | ✅ | 100% | Per-token confidence |
| | Batch Entropy Max | ✅ | 100% | Load balancing |
| | GNN Routing | ✅ | 100% | Optional, torch_geometric |
| | Dynamic Top-K | 🔲 SOTA++ | 130% | Entropy-aware k (novel) |
| **DARE** (ICLR 2026) | Log-Perplexity Proxy | ✅ | 100% | Step Entropy as proxy |
| | Adaptive Thresholding | ✅ | 100% | Learnable thresholds |
| | Dynamic Expert Count | ✅ | 100% | k ∈ [1, num_experts] |
| **MoDE** (AAAI-22) | Domain-Specific Experts | 🔲 SOTA++ | 140% | Adapted for LLM routing |
| | Feature Disentanglement | 🔲 | — | Not yet implemented |
| | Expert Gating | 🔲 | — | Planned for Phase 5 |

**Legend:**
- ✅ = Implemented and tested
- 🔲 = Planned (not yet implemented)
- Status: COMPLETE = production-ready, SOTA++ = beyond paper spec

---

# VIII. FAILURE GRAMMAR & EDGE CASES

## 8.1 Known Failure Modes

### 8.1.1 **Ada Bridge Not Available**
**Symptom:** `ImportError: cannot import name 'GarlicAdaStepEntropy'`  
**Cause:** `libgarlic_core.so` not compiled or not in path  
**Mitigation:**
```python
# Automatic fallback to Python
if not ADA_AVAILABLE:
    logger.warning("Using Python step entropy (slower)")
    # Falls back to StepEntropyCalculator
```
**Resolution:** Compile Ada bridge:
```bash
cd ada_bridge
gprbuild -P garlic_core.gpr
```

### 8.1.2 **Graph Context Shape Mismatch**
**Symptom:** RuntimeError in `entropy_router()`: tensor size mismatch  
**Cause:** HSGM compressed_context has variable `num_summary` nodes, but hidden_states expects fixed `seq_len`  
**Mitigation:**
```python
# SystemRouter forward(), Stage 10
if graph_context.shape[1] != seq_len:
    graph_pooled = graph_context.mean(dim=1, keepdim=True).expand(
        batch_size, seq_len, -1
    )
```

### 8.1.3 **GlobalGraphMemory OOM**
**Symptom:** CUDA out of memory after 500+ turns  
**Cause:** Turn history accumulates without pruning  
**Mitigation:**
```python
# Add to GlobalGraphMemory.add_turn()
if len(self.turn_history) > self.max_turns:
    oldest = self.turn_history.pop(0)
    logger.debug(f"Evicted turn {oldest.turn_id} (max_turns reached)")
```

### 8.1.4 **Model Slot Unloaded**
**Symptom:** `ModelLoader.get_active_model()` returns None  
**Cause:** Router selected a slot that hasn't been loaded  
**Mitigation:**
```python
# SystemRouter forward(), Stage 7
active_model = self.model_loader.get_active_model(active_slot.value)
if active_model is None:
    logger.warning(f"Slot {active_slot.name} unloaded, using fallback")
    # Use template-only output (no model generation)
```

### 8.1.5 **Entropy Threshold Drift**
**Symptom:** Routing becomes erratic after long training  
**Cause:** Learnable thresholds diverge without bounds  
**Mitigation:**
```python
# Add to DifficultyAwareExpertAllocator.get_routing_loss()
bound_loss = (
    F.relu(1.0 - self.threshold_low) +
    F.relu(self.threshold_high - 6.0)
)
```

---

## 8.2 Edge Case Handling

### 8.2.1 **Empty Message History**
```python
# InputPreparer.prepare()
if not context.get('messages'):
    # Fallback to zero embeddings
    message_embs = torch.zeros(1, 10, self.config.context_dim, device=device)
```

### 8.2.2 **First Turn (No Global Memory)**
```python
# SystemRouter forward(), Stage 1
if self.global_memory.is_empty:
    history_context = None
    # Skip prepend, continue with current input only
```

### 8.2.3 **No Hidden States (Template-Only Mode)**
```python
# SystemRouter forward()
if hidden_states is None:
    # Skip HSGM, entropy routing, injection
    # Template selection only
    logger.debug("Template-only mode (no hidden states)")
```

### 8.2.4 **Torch Geometric Not Installed**
```python
# SystemRouter.__init__()
if config.enable_gnn_routing:
    try:
        from torch_geometric.nn import GATConv
        self.use_gnn = True
    except ImportError:
        logger.warning("torch_geometric not found, disabling GNN routing")
        self.use_gnn = False
```

---

# IX. EXECUTION PLAN — SINGLE PASS

## 9.1 Pre-Flight Checklist

**Before starting implementation:**

- [ ] **Backup current codebase**
  ```bash
  cp -r ~/Projects/System-Router ~/Projects/System-Router.backup.$(date +%Y%m%d)
  cp -r ~/Projects/ADA-Step-Entropy ~/Projects/ADA-Step-Entropy.backup.$(date +%Y%m%d)
  ```

- [ ] **Verify Ada bridge compiles**
  ```bash
  cd ~/Projects/ADA-Step-Entropy
  gprbuild -P garlic_core.gpr
  python ada_bridge.py  # Should print 16/16 tests passing
  ```

- [ ] **Create test suite skeleton**
  ```bash
  cd ~/Projects/System-Router
  mkdir -p tests
  touch tests/test_phase{1..7}_{rename,ada,memory,internalization,sota,training,integration}.py
  ```

- [ ] **Create IMPLEMENTATION_LOG.md**
  ```bash
  cp CONSTITUTION.md IMPLEMENTATION_LOG.md
  # Use checklist from §V as tracking document
  ```

## 9.2 Execution Order (Single Session)

**Total Est. Time:** 20-30 hours (1-2 full work days)

### Hour 0-3: Phase 1 (Foundation)
- Execute all checklist items from §V.1
- Test: `pytest tests/test_phase1_rename.py -v`
- **Milestone:** `ModelSlot` architecture working, `SystemOutput` defined

### Hour 3-5: Phase 2 (Ada Bridge)
- Move ADA-Step-Entropy folder
- Wire Ada bridge protocol
- Test: `pytest ada_bridge/tests/test_ada_bridge.py -v`
- **Milestone:** Step entropy callable from SystemRouter

### Hour 5-8: Phase 3 (GlobalGraphMemory)
- Implement `TurnRecord` + `GlobalGraphMemory`
- Wire into forward pass stages 1 + 13
- Test: `pytest tests/test_phase3_memory.py -v`
- **Milestone:** Multi-turn persistence working

### Hour 8-14: Phase 4 (Component Internalization)
- Copy all Garlic components into `system_router.py`
- Merge configs
- Implement `SystemRouter.__init__()` and `forward()`
- Test: `pytest tests/test_phase4_internalization.py -v`
- **Milestone:** 13-stage forward pass executes end-to-end

### Hour 14-22: Phase 5 (SOTA++ Innovations)
- Implement `DifficultyAwareExpertAllocator`
- Implement `MultiScaleGraphBuilder`
- Implement `GNNExpertCollaborationLayer`
- Implement `EntropyAwareInjectionController`
- Test: `pytest tests/test_phase5_sota.py -v`
- **Milestone:** All novel components operational

### Hour 22-26: Phase 6 (Training Infrastructure)
- Update `RouterTrainer.compute_loss()`
- Add curriculum scheduler
- Add checkpoint saving
- Test: `pytest tests/test_phase6_training.py -v`
- **Milestone:** Training loop ready

### Hour 26-30: Phase 7 (Integration Testing)
- Run full inference test
- Run multi-turn session test
- Run Ada bridge validation
- Run memory injection lifecycle test
- Test: `pytest tests/test_phase7_integration.py -v --cov=system_router`
- **Milestone:** Production-ready system

---

## 9.3 Success Criteria

### ✅ Phase 1 Success:
- [ ] `ModelSlot` enum exists with SLOT_A/B/C
- [ ] `SlotPredictions.model_slot` field exists
- [ ] `SlotPredictorNetwork.model_selector_head` produces [B, 3] output
- [ ] `SystemOutput` dataclass compiles
- [ ] `ModelLoader` class instantiates

### ✅ Phase 2 Success:
- [ ] `ada_bridge/` folder exists in System-Router
- [ ] `from ada_bridge.ada_bridge import GarlicAdaStepEntropy` works
- [ ] `build_step_entropy()` returns Ada bridge when .so available
- [ ] Ada bridge test suite passes 16/16

### ✅ Phase 3 Success:
- [ ] `GlobalGraphMemory` class exists
- [ ] `.add_turn()` accumulates graphs via mean-pooling
- [ ] `.get_context()` returns compressed history
- [ ] Multi-turn test shows compression ratio improving over turns

### ✅ Phase 4 Success:
- [ ] `system_router.py` compiles without import errors
- [ ] `SystemRouter.__init__()` instantiates all components
- [ ] `SystemRouter.forward()` executes all 13 stages
- [ ] Returns `SystemOutput` with all fields populated
- [ ] `garlic_orchestrator.py` moved to `_archive/`

### ✅ Phase 5 Success:
- [ ] `DifficultyAwareExpertAllocator` produces dynamic expert_counts
- [ ] `MultiScaleGraphBuilder` returns fine/medium/coarse graphs
- [ ] `GNNExpertCollaborationLayer` performs message passing
- [ ] `EntropyAwareInjectionController` outputs layer_mask

### ✅ Phase 6 Success:
- [ ] `RouterTrainer.compute_loss()` includes all SOTA++ losses
- [ ] Curriculum scheduler modulates thresholds over epochs
- [ ] Checkpoint saving includes all state (model, memory, thresholds)

### ✅ Phase 7 Success:
- [ ] End-to-end inference test passes
- [ ] Multi-turn session accumulates memory correctly
- [ ] Ada bridge entropy calculations match expected ranges
- [ ] Memory injection produces non-zero perturbations
- [ ] Code coverage > 80%

---

# X. FINAL NOTES

## 10.1 This is NOT a Prototype

Every component in this Constitution is production-grade:
- Ada Step Entropy: 16/16 tests, formally verified SPARK-ready
- HSGM: Paper-compliant compression with 90-102x ratios
- Neural Router: 2048 lines of enterprise-grade prompt routing
- Memory Injection: Works standalone AND integrated

The merge is assembly, not invention.

## 10.2 SOTA++ is the Floor, Not the Ceiling

The innovations listed (difficulty-aware allocation, multi-scale graphs, GNN collaboration, adaptive injection) are **baseline enhancements**. If during implementation you identify additional optimizations from the papers or novel combinations, implement them.

This is Somnus. The standard is DARPA/DeepMind-grade. Act accordingly.

## 10.3 The Constitution is Living

This document should be updated as:
- New components are added
- New papers are integrated
- New failure modes are discovered
- New optimizations are identified

Treat it as the source of truth for the System Router architecture.

## 10.4 Provenance Matters

Every component must trace to either:
1. A published paper (cite arXiv/conference)
2. A prior Somnus implementation (cite file + date)
3. A novel contribution (mark as SOTA++, document reasoning)

No orphaned code. No "it just works" explanations. Full lineage.

---

# XI. ARCHITECTURE DIAGRAMS

## 11.1 Component Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ROUTER ARCHITECTURE                    │
└─────────────────────────────────────────────────────────────────┘

INPUT: Context Dict
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ INPUT PREPARATION (Stage 0)                                      │
│ ─────────────────────────────                                    │
│ HashTextEncoder ─────→ message_embs [B, S, D]                   │
│ ProfileEncoder ──────→ profile [B, 128]                         │
│ MetadataEncoder ─────→ metadata [B, 64]                         │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ GLOBAL MEMORY CONTEXT (Stage 1)                                 │
│ ────────────────────────────────                                 │
│ GlobalGraphMemory.get_context() → history_graph                 │
│ IF exists: prepend to hidden_states                             │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ HSGM COMPRESSION (Stage 2)                                       │
│ ──────────────────────────                                       │
│ LocalGraphBuilder ────→ segment_graphs                          │
│ GNNSummaryExtractor ──→ compressed_nodes (90-102x)             │
│ HierarchicalQuery ────→ retrieval_context                       │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ NEURAL ROUTING (Stages 3-7)                                     │
│ ────────────────────────────                                     │
│ ContextEncoder ───────→ context_emb [B, D]                      │
│ SlotPredictor ────────→ model_slot [B, 3]                       │
│                     └──→ tool_enables, tool_weights             │
│ SafetyValidator ──────→ constraint enforcement                   │
│ ModelLoader ──────────→ active_slot (SLOT_A/B/C)                │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ ENTROPY ROUTING (Stage 8)                                       │
│ ─────────────────────────                                        │
│ Ada Bridge ───────────→ entropy_value (bits/token)              │
│ Threshold Classifier ─→ 'fast' / 'normal' / 'slow'             │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ SEMANTIC SIGNALS (Stage 9)                                      │
│ ──────────────────────────                                       │
│ ReasoningScaffolder ──→ semantic_signal (EXPLORATION, etc.)    │
│                      └→ signal_embedding [B, S, D]              │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ EXPERT ROUTING (Stage 10)                                       │
│ ──────────────────────────                                       │
│ EntropyRegularizedRouter ──→ expert_indices [B, S, K]          │
│                            └→ gating_weights [B, S, K]          │
│                            └→ routing_entropy, is_ood           │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ GARLIC BRIDGE INJECTION (Stage 11)                             │
│ ───────────────────────────────────                              │
│ Pool graph_memory ────→ [B, S, D]                              │
│ Project signals ──────→ [B, S, D]                              │
│ Concat [graph; signals] → memory_context [B, S, 2D]            │
│ Bridge → memory_injection [B, S, D]                            │
│ h' = h + injection ────→ augmented_states                       │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ TEMPLATE ASSEMBLY (Stage 12)                                    │
│ ─────────────────────────                                        │
│ TemplateSelector ─────→ template_weights [B, num_templates]    │
│ TemplateLibrary ──────→ prompt (str)                            │
└──────────────────────────────────────────────────────────────────┘
   ↓
┌──────────────────────────────────────────────────────────────────┐
│ MEMORY UPDATE (Stage 13)                                        │
│ ─────────────────────────                                        │
│ GlobalGraphMemory.add_turn(input_graph, entropy, slot)         │
│ → Accumulates for next turn                                     │
└──────────────────────────────────────────────────────────────────┘
   ↓
OUTPUT: SystemOutput(
    prompt, model_slot, routing_decision,
    entropy_level, compression_ratio,
    augmented_states, trace
)
```

## 11.2 SOTA++ Innovation Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                    SOTA++ ENHANCEMENTS                           │
│          (Beyond Baseline Paper Implementations)                 │
└─────────────────────────────────────────────────────────────────┘

[1] DIFFICULTY-AWARE EXPERT ALLOCATION (DARE-inspired)
    ─────────────────────────────────────────────────
    Entropy → Learnable Thresholds → Dynamic K
    
    Low entropy (< 2.0) ──→ 1 expert (fast path)
    Medium (2-4) ────────→ 2 experts
    High (> 4.0) ────────→ 3 experts (slow, memory)

[2] MULTI-SCALE HSGM GRAPHS
    ───────────────────────
    Fine (seg=16) ──────→ Early layers
    Medium (seg=32) ────→ Middle layers
    Coarse (seg=64) ────→ Late layers

[3] GNN EXPERT COLLABORATION
    ────────────────────────
    Experts = Nodes
    Co-activation = Edges
    Message Passing → Collaborative Output

[4] ENTROPY-AWARE INJECTION CONTROL
    ────────────────────────────────
    Low entropy ──→ Inject late layers only
    High entropy ─→ Inject all layers
    
    Learned layer_mask per entropy bin

[5] ADAPTIVE THRESHOLDS (Welford online)
    ────────────────────────────────────
    Mean + Std updated every forward pass
    Thresholds drift to match data distribution

[6] GRPO REWARDS (Step Entropy Table 3)
    ───────────────────────────────────
    Pruning ratio = 0.80 (optimal)
    Reward = f(entropy_history, pruning_success)
    Trained end-to-end with routing loss
```

---

# XII. REFERENCES

## 12.1 Core Papers

1. **Step Entropy** — arXiv:2508.03346  
   File: `StepEntropy.pdf`  
   Key: Shannon entropy for token-level routing, [SKIP] injection, GRPO rewards

2. **HSGM** — Hierarchical Segment-Graph Memory for Scalable Long-Text  
   File: `HSGMHierarchical_SegmentGraph_Memory_for_Scalable_LongText_ocr_1.pdf`  
   Key: 90-102x compression, graph-based memory

3. **Reasoning Scaffolding** — arXiv:2509.23619  
   File: `REASONING_SCAFFOLDINGDISTILLING_THE_FLOW_ocr_1.pdf`  
   Key: Semantic signals (EXPLORATION, VERIFICATION, etc.)

4. **DARE** — Difficulty-Aware Dynamic Routing for Mixture of Experts (ICLR 2026)  
   File: `8748_DARE_Difficulty_Aware_Dyn.pdf`  
   Key: Log-perplexity as difficulty proxy, adaptive expert allocation

5. **MoDE** — Mixture of Domain-Specific Experts (AAAI-22)  
   File: `2000013240131220220628.pdf`  
   Key: Disentangled features (domain-specific vs domain-general)

6. **ExpertAD** — Mixture of Experts for Autonomous Driving  
   File: `2511_11740v1.pdf`  
   Key: Perception adapter, sparse expert activation

7. **Pangu 5.5** — Entropy Regularized MoE Routing  
   Reference: Huawei Pangu architecture  
   Key: Sample entropy minimization + batch entropy maximization

## 12.2 Somnus Implementations

- `neural_router.py` — 2048-line production prompt router
- `Garlic-Components/` — HSGM, scaffolder, entropy router
- `memory_injection_system.py` — Standalone memory system
- `ada_bridge/` — Ada Step Entropy with C FFI (16/16 tests)

---

**END OF CONSTITUTION**

---

**Usage:**
```bash
# Track implementation
tail -f IMPLEMENTATION_LOG.md

# Run full test suite
pytest tests/ -v --cov=system_router --cov-report=html

# Verify Ada bridge
cd ada_bridge && python ada_bridge.py

# Start implementation
python scripts/merge_system_router.py
```

**Next:** Execute Phase 1 checklist items from §V.1
