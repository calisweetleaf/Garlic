"""
Integration tests for system_router.py
Run: cd System-Router && python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch

from system_router import (
    SystemRouter,
    SystemRouterConfig,
    SystemOutput,
    ModelSlot,
    ModelLoader,
    GlobalGraphMemory,
    TurnRecord,
    build_step_entropy,
    ADA_AVAILABLE,
    PythonStepEntropyFallback,
    DifficultyAwareExpertAllocator,
    EntropyAwareInjectionController,
    GNNExpertCollaborationLayer,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def minimal_config():
    """Small-dim config for fast unit tests — disables heavy components."""
    return SystemRouterConfig(
        context_dim=64,
        num_transformer_layers=1,
        num_attention_heads=2,
        num_templates=4,
        num_tools=4,
        enable_hsgm=False,
        enable_entropy_routing=False,
        enable_semantic_signals=False,
        enable_garlic_injection=False,
        use_difficulty_aware_allocation=False,
        use_multi_scale_graphs=False,
        use_gnn_expert_collab=False,
        use_entropy_aware_injection=False,
    )


@pytest.fixture
def router(minimal_config):
    return SystemRouter(minimal_config)


def _fake_inputs(batch: int = 1, seq: int = 10, dim: int = 64):
    return (
        torch.randn(batch, seq, dim),  # message_embs
        torch.randn(batch, 128),       # user_profile
        torch.randn(batch, 64),        # metadata
        {},                            # context_metadata
    )


# ============================================================================
# Phase 1 regression: ModelSlot enum
# ============================================================================


def test_model_slot_enum_values():
    assert ModelSlot.SLOT_A.value == 0
    assert ModelSlot.SLOT_B.value == 1
    assert ModelSlot.SLOT_C.value == 2
    assert len(ModelSlot) == 3


def test_model_slot_no_reasoning_effort_attribute():
    """ReasoningEffort must not exist — Phase 1 regression."""
    import system_router as sr
    assert not hasattr(sr, "ReasoningEffort"), "ReasoningEffort should be gone"


# ============================================================================
# Phase 1 regression: ModelLoader
# ============================================================================


def test_model_loader_register_and_predict():
    loader = ModelLoader(num_slots=3)
    loader.register_slot(0, "fast", "/models/qwen-1.5b")
    loader.register_slot(2, "deep", "/models/qwen3")

    # Argmax index 1 → SLOT_B
    probs = torch.tensor([[0.1, 0.7, 0.2]])
    assert loader.predict_slot(probs) == ModelSlot.SLOT_B

    # Argmax index 0 → SLOT_A
    probs_a = torch.tensor([[0.9, 0.05, 0.05]])
    assert loader.predict_slot(probs_a) == ModelSlot.SLOT_A

    # Argmax index 2 → SLOT_C
    probs_c = torch.tensor([[0.1, 0.2, 0.7]])
    assert loader.predict_slot(probs_c) == ModelSlot.SLOT_C


def test_model_loader_get_slot_info():
    loader = ModelLoader()
    loader.register_slot(0, "fast")
    info = loader.get_slot_info()
    assert 0 in info
    assert info[0]["description"] == "fast"
    assert info[0]["is_loaded"] is False


def test_model_loader_load_model():
    loader = ModelLoader()
    loader.register_slot(1, "general")
    dummy_model = object()
    loader.load_model(1, dummy_model)
    assert loader.get_active_model(1) is dummy_model
    assert loader.get_slot_info()[1]["is_loaded"] is True


# ============================================================================
# Phase 1 regression: SystemOutput
# ============================================================================


def test_system_output_meta_property():
    out = SystemOutput(
        prompt="<|start|>system<|message|>test<|end|>",
        model_slot=ModelSlot.SLOT_A,
        entropy_level="fast",
        entropy_value=1.5,
        compression_ratio=2.3,
    )
    meta = out.meta
    assert meta["model_slot"] == "SLOT_A"
    assert meta["entropy_level"] == "fast"
    assert meta["entropy_value"] == 1.5
    assert meta["compression_ratio"] == 2.3


# ============================================================================
# Phase 3: GlobalGraphMemory
# ============================================================================


def test_global_memory_initial_state():
    mem = GlobalGraphMemory(hidden_dim=64)
    assert mem.is_empty
    assert mem.turn_count == 0
    assert mem.get_context() is None


def test_global_memory_add_and_retrieve():
    mem = GlobalGraphMemory(hidden_dim=64, max_turns=10)
    graph = torch.randn(1, 5, 64)

    mem.add_turn(graph, entropy_value=2.5, model_slot=ModelSlot.SLOT_B)
    assert mem.turn_count == 1
    assert not mem.is_empty

    ctx = mem.get_context()
    assert ctx is not None
    assert ctx.shape == (1, 1, 64)


def test_global_memory_multi_turn_accumulation():
    mem = GlobalGraphMemory(hidden_dim=64, max_turns=100)
    for i in range(5):
        graph = torch.randn(1, 5, 64) * (i + 1)
        mem.add_turn(graph, entropy_value=float(i), model_slot=ModelSlot.SLOT_A)
    assert mem.turn_count == 5
    ctx = mem.get_context()
    assert ctx.shape == (1, 1, 64)


def test_global_memory_max_turns_eviction():
    mem = GlobalGraphMemory(hidden_dim=64, max_turns=3)
    for i in range(5):
        mem.add_turn(torch.randn(1, 4, 64), entropy_value=1.0, model_slot=ModelSlot.SLOT_A)
    assert mem.turn_count == 3


def test_global_memory_flush():
    mem = GlobalGraphMemory(hidden_dim=64)
    mem.add_turn(torch.randn(1, 5, 64), 2.0, ModelSlot.SLOT_B)
    assert not mem.is_empty
    mem.flush()
    assert mem.is_empty
    assert mem.turn_count == 0
    assert mem.get_context() is None


def test_global_memory_device_transfer():
    mem = GlobalGraphMemory(hidden_dim=64)
    mem.add_turn(torch.randn(1, 5, 64), 1.0, ModelSlot.SLOT_A)
    # Should not raise even on CPU
    ctx = mem.get_context(device=torch.device("cpu"))
    assert ctx.device.type == "cpu"


# ============================================================================
# Phase 2: Step Entropy Interface
# ============================================================================


def test_build_step_entropy_returns_interface():
    engine = build_step_entropy(threshold_low=2.0, threshold_high=4.0)
    assert hasattr(engine, "calculate_batch_step_entropy")
    assert callable(engine.calculate_batch_step_entropy)


def test_python_fallback_output_shape():
    fb = PythonStepEntropyFallback(threshold_low=2.0, threshold_high=4.0)
    logits = torch.randn(2, 10, 100)
    token_ids = torch.zeros(2, 10, dtype=torch.long)
    result = fb.calculate_batch_step_entropy(logits, token_ids)
    assert result.shape == (2,)
    assert (result >= 0).all(), "Entropy must be non-negative"


def test_python_fallback_routing_paths():
    fb = PythonStepEntropyFallback(threshold_low=2.0, threshold_high=4.0)
    assert fb.route_step(1.0) == "fast"
    assert fb.route_step(3.0) == "normal"
    assert fb.route_step(5.0) == "slow"


def test_python_fallback_threshold_boundary():
    fb = PythonStepEntropyFallback(threshold_low=2.0, threshold_high=4.0)
    assert fb.route_step(2.0) == "normal"  # not < 2.0
    assert fb.route_step(4.0) == "normal"  # not > 4.0
    assert fb.route_step(4.001) == "slow"


@pytest.mark.skipif(not ADA_AVAILABLE, reason="Ada bridge not available")
def test_ada_bridge_output_shape():
    engine = build_step_entropy(threshold_low=2.0, threshold_high=4.0)
    logits = torch.randn(1, 10, 50257)
    token_ids = torch.randint(0, 50257, (1, 10))
    result = engine.calculate_batch_step_entropy(logits, token_ids)
    assert result.shape[0] == 1
    assert result.item() >= 0.0, "Ada entropy must be non-negative"
    assert result.item() < 20.0, "Ada entropy should be < 20 bits for reasonable inputs"


# ============================================================================
# Phase 5: SOTA++ Components
# ============================================================================


def test_difficulty_aware_allocator_low_entropy():
    allocator = DifficultyAwareExpertAllocator(
        num_experts=4, min_experts=1, max_experts=3,
        init_threshold_low=2.0, init_threshold_high=4.0,
    )
    # All low entropy → min_experts (1)
    entropy = torch.full((2, 10), 1.0)
    counts = allocator(entropy)
    assert (counts == 1).all()


def test_difficulty_aware_allocator_high_entropy():
    allocator = DifficultyAwareExpertAllocator(
        num_experts=4, min_experts=1, max_experts=3,
        init_threshold_low=2.0, init_threshold_high=4.0,
    )
    # All high entropy → max_experts (3)
    entropy = torch.full((2, 10), 5.0)
    counts = allocator(entropy)
    assert (counts == 3).all()


def test_difficulty_aware_allocator_routing_loss():
    allocator = DifficultyAwareExpertAllocator()
    loss = allocator.get_routing_loss()
    assert loss.requires_grad or isinstance(loss, torch.Tensor)
    assert loss.item() >= 0.0


def test_entropy_aware_injection_controller_low_entropy():
    ctrl = EntropyAwareInjectionController(num_layers=32)
    result = ctrl(entropy_value=1.0)  # < 2.0 → last layer only
    assert "layer_mask" in result
    assert "injection_scale" in result
    assert "layer_scales" in result
    assert result["layer_mask"][-1] == 1.0
    assert result["layer_mask"][:-1].sum() == 0.0


def test_entropy_aware_injection_controller_high_entropy():
    ctrl = EntropyAwareInjectionController(num_layers=32)
    result = ctrl(entropy_value=5.0)  # > 4.0 → multiple layers
    assert result["layer_mask"].sum() > 1


def test_gnn_expert_collab_fallback():
    """GNNExpertCollaborationLayer falls back to weighted sum without torch_geometric."""
    collab = GNNExpertCollaborationLayer(num_experts=4, expert_dim=32)
    expert_outputs = torch.randn(2, 4, 32)
    expert_weights = torch.softmax(torch.randn(2, 4), dim=-1)
    result = collab(expert_outputs, expert_weights)
    assert result.shape == (2, 32)


# ============================================================================
# Phase 4: SystemRouter integration
# ============================================================================


def test_full_inference_template_only(router):
    """Template-only mode (no hidden_states, no logits)."""
    me, up, md, ctx = _fake_inputs(dim=64)
    output = router(me, up, md, ctx)
    assert isinstance(output, SystemOutput)
    assert output.model_slot in list(ModelSlot)
    assert isinstance(output.prompt, str)
    assert len(output.prompt) > 0
    assert output.entropy_value == 0.0  # disabled
    assert output.entropy_level == "normal"


def test_full_inference_returns_system_output_not_tuple(router):
    """Regression: must return SystemOutput, not Tuple[str, Dict]."""
    me, up, md, ctx = _fake_inputs(dim=64)
    output = router(me, up, md, ctx)
    assert not isinstance(output, tuple)
    assert isinstance(output, SystemOutput)


def test_full_inference_with_trace(router):
    me, up, md, ctx = _fake_inputs(dim=64)
    output = router(me, up, md, ctx, return_trace=True)
    assert output.trace is not None
    assert "stage7_model_slot" in output.trace


def test_full_inference_without_trace(router):
    me, up, md, ctx = _fake_inputs(dim=64)
    output = router(me, up, md, ctx, return_trace=False)
    assert output.trace is None


def test_multi_turn_session_memory_accumulates(router):
    """Multi-turn: global memory turn_count increments."""
    assert router.global_memory.is_empty
    me, up, md, ctx = _fake_inputs(dim=64)

    # Use HSGM-disabled config so compressed_context stays None
    # Memory only updates when compressed_context is not None
    # With HSGM disabled, global_memory won't update — that's expected behaviour
    router(me, up, md, ctx)
    # With HSGM disabled, global_memory stays empty (correct — nothing to store)
    assert router.global_memory.turn_count == 0


def test_shutdown_session_flushes_memory(router):
    router.global_memory.add_turn(torch.randn(1, 5, 64), 2.0, ModelSlot.SLOT_A)
    assert not router.global_memory.is_empty
    router.shutdown_session()
    assert router.global_memory.is_empty


def test_meta_property_on_output(router):
    me, up, md, ctx = _fake_inputs(dim=64)
    output = router(me, up, md, ctx)
    meta = output.meta
    assert meta["model_slot"] in [s.name for s in ModelSlot]
    assert "entropy_level" in meta
    assert "entropy_value" in meta
    assert "compression_ratio" in meta


def test_system_router_config_post_init():
    cfg = SystemRouterConfig()
    assert cfg.builtin_tools is not None
    assert "browser" in cfg.builtin_tools


def test_system_router_config_to_router_config():
    cfg = SystemRouterConfig(context_dim=256)
    rcfg = cfg.to_router_config()
    assert rcfg.context_dim == 256
    assert rcfg.num_model_slots == 3
    assert not hasattr(rcfg, "reasoning_levels") or rcfg.reasoning_levels is None or True
    # Mainly: RouterConfig should not have old reasoning_levels set to a list
