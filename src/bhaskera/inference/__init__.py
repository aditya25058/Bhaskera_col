"""
bhaskera.inference
==================
Complete inference capability for the Bhaskera LLM framework.

Modules
-------
engine        InferenceEngine — main generation entry point
kv_cache      StaticKVCache, TurboQuantKVCache — KV cache strategies
lloyd_max     LloydMaxCodebook — optimal scalar quantizer for TurboQuant
speculative   SpeculativeDecoder — lossless 2–3× decode speedup
sampling      top_k_filter, top_p_filter, sample_from_logits — sampling utils
colossus      ZSSRPredictor, ColumnDirectory — COLOSSUS MoE offload (opt-in)

Quick start
-----------
    from bhaskera.config import load_config
    from bhaskera.inference import InferenceEngine

    cfg    = load_config("config.yaml")
    engine = InferenceEngine(cfg)
    texts  = engine.generate(["Tell me about quantum computing"])
    print(texts[0])

Config YAML keys (under `inference:`)
--------------------------------------
    max_new_tokens: 512
    temperature:    1.0
    top_p:          0.9
    top_k:          50
    do_sample:      true
    kv_cache:       turboquant   # static | turboquant | none
    device:         auto         # cuda | cpu | mps | auto
    torch_compile:  false

    turboquant:
      key_bits:          4
      value_bits:        2
      residual_window:   128
      protected_layers:  2

    speculative:
      enabled:          false
      draft_model_name: ""
      num_draft_tokens: 5
"""
from .engine     import InferenceEngine
from .kv_cache   import (
    BaseKVCache,
    StaticKVCache,
    TurboQuantKVCache,
    build_kv_cache,
)
from .lloyd_max  import LloydMaxCodebook, solve_lloyd_max
from .sampling   import (
    greedy_sample,
    sample_from_logits,
    temperature_scale,
    top_k_filter,
    top_p_filter,
)
from .speculative import SpeculativeDecoder, build_speculative_decoder
from .colossus import (
    ColumnDirectory,
    ZSSRPredictor,
    columns_batched,
    columns_loop,
    plan_fixed_packets,
    quantize_int8,
    stack_expert_weights,
)

__all__ = [
    # Engine
    "InferenceEngine",
    # KV caches
    "BaseKVCache",
    "StaticKVCache",
    "TurboQuantKVCache",
    "build_kv_cache",
    # Quantizer
    "LloydMaxCodebook",
    "solve_lloyd_max",
    # Sampling
    "greedy_sample",
    "sample_from_logits",
    "temperature_scale",
    "top_k_filter",
    "top_p_filter",
    # Speculative
    "SpeculativeDecoder",
    "build_speculative_decoder",
    # COLOSSUS (opt-in, default-off)
    "ZSSRPredictor",
    "quantize_int8",
    "columns_batched",
    "columns_loop",
    "stack_expert_weights",
    "ColumnDirectory",
    "plan_fixed_packets",
]
