"""
bhaskera-infer — command-line inference entry point (v2).

Changes vs v1:
  - Tokens/second reported after every generation
  - Token count measured from actual output ids (not char count)
  - Thinking models: <think> block stripped from terminal output by default
    (--show-thinking to display it; raw text is always saved if --output-file)
  - TurboQuant stats line always shown when cache is active
  - Cleaner separator / stats block

Examples
--------
    # Standard generation
    bhaskera-infer --config configs/inference_turboquant.yaml \\
                   --prompt "Explain attention mechanisms."

    # Param2 Thinking model (strips <think> by default)
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --prompt "What is 17 × 23?"

    # Show the chain-of-thought
    bhaskera-infer --config configs/inference_param2.yaml \\
                   --prompt "What is 17 × 23?" --show-thinking

    # Benchmark throughput
    bhaskera-infer --config configs/inference_turboquant.yaml \\
                   --prompt-file prompts.txt --max-new-tokens 256
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("bhaskera.infer")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bhaskera-infer",
        description="Bhaskera LLM inference engine CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config / model
    p.add_argument("--config",  default=None,   help="Path to Bhaskera YAML config")
    p.add_argument("--model",   default=None,   help="HuggingFace model id (overrides config)")
    p.add_argument("--device",  default="auto", help="Device: auto | cuda | cpu | mps")

    # Input (required unless --plan, which serves nothing)
    inp = p.add_mutually_exclusive_group(required=False)
    inp.add_argument("--prompt",      default=None, help="Single prompt string")
    inp.add_argument("--prompt-file", default=None, metavar="FILE",
                     help="File with one prompt per line")

    # Generation
    p.add_argument("--max-new-tokens", type=int,   default=None)
    p.add_argument("--temperature",    type=float, default=None)
    p.add_argument("--top-p",          type=float, default=None)
    p.add_argument("--top-k",          type=int,   default=None)
    p.add_argument("--no-sample",      action="store_true",
                   help="Greedy decoding (overrides do_sample=true in config)")

    # KV cache
    p.add_argument("--kv-cache", default=None, choices=["static", "turboquant", "none"])
    p.add_argument("--key-bits",          type=int, default=None)
    p.add_argument("--value-bits",        type=int, default=None)
    p.add_argument("--residual-window",   type=int, default=None)

    # Speculative decoding
    p.add_argument("--speculative",    action="store_true")
    p.add_argument("--draft-model",    default=None)
    p.add_argument("--num-draft-tokens", type=int, default=None)

    # Thinking model
    p.add_argument("--show-thinking",  action="store_true",
                   help="Print <think> reasoning block (Param2 / thinking models)")
    p.add_argument("--system-prompt",  default="You are a helpful assistant.",
                   help="System prompt injected for chat-template models")

    # Output
    p.add_argument("--output-file", default=None, metavar="FILE",
                   help="Write full raw output to file (one response per line)")
    p.add_argument("--return-full",    action="store_true",
                   help="Include the prompt in the output")
    p.add_argument("--torch-compile",  action="store_true")
    p.add_argument("--verbose", "-v",  action="store_true")
    # COLOSSUS SA-FFN
    p.add_argument("--missing-col-ratio", type=float, default=None,
                   help="Missing column fraction for SA-FFN (e.g. 0.50, 0.25, 0.10)")

    # COLOSSUS huge-model tiers (models larger than HBM; default off).
    # Flag defaults are None so YAML config values apply when flags are absent.
    p.add_argument("--offload-tier", default=None, choices=["off", "slots", "cpu"],
                   help="Tiered MoE execution: slots (bitwise, GPU) | cpu (ulp1, oneDNN)")
    p.add_argument("--capacity", type=int, default=None,
                   help="Dynamic expert slots per MoE layer (tiered paths)")
    p.add_argument("--prefill-chunk", type=int, default=None,
                   help="Chunked prefill size (0=single shot; bounds activation memory)")
    p.add_argument("--exactness-mode", default=None, choices=["bitwise", "ulp1"],
                   help="bitwise: GPU-only exact; ulp1: allow CPU placement")
    # Phase B remote weights (research branch only; PRs frozen/unaffected).
    p.add_argument("--remote-repo", default=None, metavar="REPO_ID",
                   help="Serve without full download: fetch on demand from Hub "
                        "(e.g. Qwen/Qwen3-30B-A3B); needs --remote-cache")
    p.add_argument("--remote-cache", default=None, metavar="DIR",
                   help="Persistent local tensor cache for --remote-repo")
    p.add_argument("--remote-cap-gb", type=float, default=200.0,
                   help="Remote cache byte cap (LRU, GB)")
    p.add_argument("--local-mirror", default=None, metavar="DIR",
                   help="Partial local checkout: serve local-first, remote fills "
                        "the rest (TieredHandles; no index.json needed)")
    p.add_argument("--log-routing", default=None, metavar="PATH",
                   help="Dump per-layer per-step routing unions to JSON")
    p.add_argument("--teacher-tokens", default=None, metavar="PATH",
                   help="JSON int list; teacher-forced flip protocol (B=1 only)")
    p.add_argument("--audit-logits", default=None, metavar="PATH",
                   help="Dump per-position {own, ref, margin} flip audit")
    p.add_argument("--dump-ids", default=None, metavar="PATH",
                   help="Dump B=1 continuation token IDs (teacher artifacts)")
    p.add_argument("--prefault", action="store_true",
                   help="Page in weight shards up front (cold-start tax)")
    p.add_argument("--hot-col-frac", type=float, default=None, metavar="F",
                   help="Column-granular slots: hot column fraction per expert "
                        "(omit = whole-expert slots)")
    p.add_argument("--hot-tier-file", default=None, metavar="PATH",
                   help="Adaptive-f tier JSON from colprofile "
                        "(overrides --hot-col-frac)")
    p.add_argument("--cold-cache", type=int, default=0, metavar="N",
                   help="Cold-column second-tier LRU entries per MoE layer "
                        "(column slots only; N=8/layer ~= 11 GB at f=0.5)")
    p.add_argument("--grouped-gemm", action="store_true",
                   help="Batched dispatch: one bmm trio per tier per layer "
                        "(fewer launches; audit-gated, ulp risk)")
    p.add_argument("--prefill-seed", default="off", choices=["off", "freq"],
                   help="Seed decode slots from prefill self-profile "
                        "(data movement only; exactness unaffected)")
    p.add_argument("--matrix-tiers", action="store_true",
                   help="FIRM-like control: per-matrix LRU pools instead of "
                        "whole-expert or column slots")
    # Planner (inspect + probe + feasibility; serves nothing)
    p.add_argument("--plan", action="store_true",
                   help="Print ranked serving plans instead of serving")
    p.add_argument("--plan-batch", type=int, default=1)
    p.add_argument("--plan-gen", type=int, default=16)
    p.add_argument("--plan-diverse", action="store_true",
                   help="Diverse (non-shared) batch workload assumption")
    p.add_argument("--plan-fidelity", default="bitwise", choices=["bitwise", "ulp1"])
    p.add_argument("--plan-apply", action="store_true",
                   help="With --plan: serve immediately with the recommended "
                        "flags (needs --prompt/--prompt-file)")
    p.add_argument("--remote-token", default=None, metavar="TOKEN",
                   help="Hub token for gated repos (or HF_TOKEN env)")

    return p


# ---------------------------------------------------------------------------
# Config assembly
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace):
    from bhaskera.config import (
        Config, InferenceConfig, TurboQuantConfig, SpeculativeConfig,
    )
    if args.config:
        from bhaskera.config import load_config
        cfg = load_config(args.config)
    else:
        cfg = Config()

    if args.model:
        cfg.model.name = args.model
    if args.device:
        cfg.inference.device = args.device

    infer = cfg.inference
    if args.max_new_tokens is not None: infer.max_new_tokens = args.max_new_tokens
    if args.temperature   is not None: infer.temperature    = args.temperature
    if args.top_p         is not None: infer.top_p          = args.top_p
    if args.top_k         is not None: infer.top_k          = args.top_k
    if args.no_sample:                 infer.do_sample       = False
    if args.torch_compile:             infer.torch_compile   = True
    if args.kv_cache:                  infer.kv_cache        = args.kv_cache

    if infer.kv_cache == "turboquant":
        if args.key_bits        is not None: infer.turboquant.key_bits        = args.key_bits
        if args.value_bits      is not None: infer.turboquant.value_bits      = args.value_bits
        if args.residual_window is not None: infer.turboquant.residual_window = args.residual_window
        infer.turboquant.enabled = True

    if args.speculative:
        infer.speculative.enabled = True
    if args.draft_model:
        infer.speculative.draft_model_name = args.draft_model
        infer.speculative.enabled = True
    if args.num_draft_tokens is not None:
        infer.speculative.num_draft_tokens = args.num_draft_tokens

    if args.missing_col_ratio is not None:
        if hasattr(infer, "colossus"):
            infer.colossus.missing_col_ratio = args.missing_col_ratio

    return cfg


# ---------------------------------------------------------------------------
# Token counting helper
# ---------------------------------------------------------------------------

def _count_output_tokens(text: str, tokenizer=None) -> int:
    """
    Count output tokens.
    Uses the tokenizer when available (exact); falls back to word-count
    heuristic (rough but doesn't require tokenizer access).
    """
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    # Heuristic: ~0.75 tokens per word for English, ~1.3 for code/mixed
    words = len(text.split())
    return max(1, int(words * 0.9))


# ---------------------------------------------------------------------------
# Output rendering
# ---------------------------------------------------------------------------

SEP = "─" * 72

def _render_output(
    idx: int,
    total: int,
    prompt: str,
    output_text: str,
    is_thinking_model: bool,
    show_thinking: bool,
) -> str:
    """Format one output for terminal display."""
    lines = []
    if total > 1:
        lines.append(f"\n{SEP}")
        lines.append(f"[{idx + 1}/{total}] Prompt: {prompt[:80]}{'…' if len(prompt) > 80 else ''}")
        lines.append(SEP)

    if is_thinking_model and not show_thinking:
        # Strip <think>...</think> for clean terminal output
        import re
        clean = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()
        # Also strip any leftover tool_call blocks
        clean = re.sub(r"<tool_call>.*?</tool_call>", "", clean, flags=re.DOTALL).strip()
        lines.append(clean)
    else:
        lines.append(output_text)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # ── Prompts ──────────────────────────────────────────────────────
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        path = Path(args.prompt_file)
        if not path.exists():
            parser.error(f"Prompt file not found: {args.prompt_file}")
        with open(path) as f:
            prompts = [line.rstrip("\n") for line in f if line.strip()]
        if not prompts:
            parser.error(f"No prompts found in {args.prompt_file}")
        logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_file}")
    elif not args.plan:
        parser.error("one of --prompt/--prompt-file is required (unless --plan)")
    else:
        prompts = []

    # ── Config ───────────────────────────────────────────────────────
    cfg = _build_config(args)
    infer = cfg.inference

    # ── Planner ────────────────────────────────────────────────────
    # Model + hardware discovery -> feasibility -> ranked plan table.
    # No weights loaded, no serving; planner answers precede execution.
    if args.plan:
        from bhaskera.inference.colossus.feasibility import plan, render_table
        from bhaskera.inference.colossus.hwprobe import probe as hw_probe
        from bhaskera.inference.colossus.inspector import describe_model
        from bhaskera.introspect import introspect_model
        src = args.remote_repo or args.model
        if not src:
            parser.error("--plan needs --model <dir> or --remote-repo <id>")
        if args.remote_repo:
            from bhaskera.inference.colossus.remote import RemoteShardHandles
            import os as _os
            handles = RemoteShardHandles.from_hub(
                args.remote_repo,
                cache_dir=args.remote_cache or "/tmp/plan_cache",
                token=args.remote_token or _os.environ.get("HF_TOKEN"))
            from transformers import AutoConfig
            hf_cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
            from accelerate import init_empty_weights
            with init_empty_weights():
                from transformers import AutoModelForCausalLM
                _model = AutoModelForCausalLM.from_config(
                    hf_cfg, trust_remote_code=True)
            profile = introspect_model(_model)
            desc = describe_model("", profile, name=args.remote_repo, handles=handles)
        else:
            if not Path(src).is_dir():
                parser.error(f"model dir not found: {src}")
            from transformers import AutoConfig, AutoModelForCausalLM
            from accelerate import init_empty_weights
            hf_cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
            with init_empty_weights():
                _model = AutoModelForCausalLM.from_config(
                    hf_cfg, trust_remote_code=True)
            profile = introspect_model(_model)
            desc = describe_model(src, profile)
        hw = hw_probe(fast=True)
        workload = {"batch": args.plan_batch, "seq_len": 32,
                    "gen_tokens": args.plan_gen,
                    "shared": not args.plan_diverse}
        result = plan(desc, hw, workload, fidelity=args.plan_fidelity)
        print(render_table(result, desc, hw))
        if not args.plan_apply:
            return
        rec = result.get("recommended") or {}
        if not rec.get("fits"):
            parser.error("plan --apply: recommended candidate does not fit "
                         f"({(rec.get('reasons') or ['unknown'])[:1]})")
        if not prompts:
            parser.error("plan --apply needs --prompt/--prompt-file to serve")
        args.offload_tier = rec["placement"]
        args.capacity = rec["capacity"]
        if rec["placement"] == "cpu":
            args.exactness_mode = "ulp1"
        print(f"APPLYING: --offload-tier {rec['placement']} "
              f"--capacity {rec['capacity']} "
              f"(est ~{rec['steady_tps']:.1f} tok/s)")
        # fall through to the offload-tier branch below

    # ── COLOSSUS huge-model tier ─────────────────────────────────────
    # Models larger than HBM: meta-load, mmap residency, tiered execution.
    # Bypasses engine.generate (which assumes the model fits).
    # Resolution: explicit flags > YAML colossus.* > off/defaults.
    _col = cfg.inference.colossus if hasattr(cfg.inference, "colossus") else None
    _tier = args.offload_tier or (getattr(_col, "placement", "off") if _col else "off")
    _cap = args.capacity if args.capacity is not None else (
        getattr(_col, "capacity", 12) if _col else 12)
    _chunk = args.prefill_chunk if args.prefill_chunk is not None else (
        getattr(_col, "prefill_chunk", 0) if _col else 0)
    _mode = args.exactness_mode or (getattr(_col, "exactness_mode", "bitwise")
                                    if _col else "bitwise")
    if _tier != "off":
        import os
        import torch
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        if _tier == "cpu" and _mode != "ulp1":
            parser.error("--offload-tier cpu needs --exactness-mode ulp1")
        from bhaskera.introspect import introspect_model
        from bhaskera.inference.colossus.loading import ShardHandles
        from bhaskera.inference.colossus.serve import serve_huge_moe

        model_dir = args.model
        remote = args.remote_repo
        if remote:
            assert args.remote_cache, "--remote-repo needs --remote-cache"
        elif not model_dir or not Path(model_dir).is_dir():
            parser.error("--offload-tier needs --model <local sharded dir with "
                         "model.safetensors.index.json> or --remote-repo")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        src = remote or model_dir
        logger.info(f"Loading tokenizer + meta model from {src} ...")
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from accelerate import init_empty_weights
        tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
        hf_cfg = AutoConfig.from_pretrained(src, trust_remote_code=True)
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                hf_cfg, trust_remote_code=True)
        profile = introspect_model(model)
        logger.info(f"Profile: {profile.num_experts} experts/layer, "
                    f"top-{profile.experts_per_token}, "
                    f"layer={getattr(profile.decoder_layer_cls, '__name__', '?')}")
        if remote:
            from bhaskera.inference.colossus.remote import RemoteShardHandles
            import os as _os
            remote_handles = RemoteShardHandles.from_hub(
                remote, cache_dir=args.remote_cache,
                cache_cap_gb=args.remote_cap_gb,
                token=args.remote_token or _os.environ.get("HF_TOKEN"))
            if args.local_mirror:
                from bhaskera.inference.colossus.loading import ShardHandles as _SH
                from bhaskera.inference.colossus.remote import TieredHandles
                partial = _SH.open_partial(args.local_mirror)
                logger.info(f"Local mirror: {len(partial.weight_map)} tensors; "
                            f"remote fills the rest")
                handles = TieredHandles(partial, remote_handles)
            else:
                handles = remote_handles
        else:
            handles = ShardHandles.open(model_dir)
        t0 = time.perf_counter()
        _teacher = None
        if args.teacher_tokens:
            import json as _json
            with open(args.teacher_tokens) as f:
                _teacher = _json.load(f)
            if isinstance(_teacher, dict):
                _teacher = _teacher.get("continuation_ids") or \
                    _teacher.get("teacher_tokens")
            if isinstance(_teacher, list) and _teacher and \
                    isinstance(_teacher[0], list):
                _teacher = _teacher[0] if len(_teacher) == 1 else _teacher
        res = serve_huge_moe(
            model, tokenizer, profile, handles, device, prompts,
            max_new_tokens=args.max_new_tokens or infer.max_new_tokens,
            capacity=_cap, placement=_tier,
            prefill_chunk=_chunk, log_routing=args.log_routing,
            teacher_tokens=_teacher, audit_logits=args.audit_logits,
            config=hf_cfg, prefault=args.prefault,
            dump_ids=args.dump_ids,
            hot_col_frac=args.hot_col_frac,
            hot_tier_file=args.hot_tier_file,
            cold_cache_cap=args.cold_cache,
            matrix_tiers=args.matrix_tiers,
            prefill_seed=args.prefill_seed)
        elapsed = time.perf_counter() - t0
        outputs = res["texts"]
        total_output_tokens = sum(_count_output_tokens(o, tokenizer) for o in outputs)
        raw_outputs = []
        for i, (prompt, output) in enumerate(zip(prompts, outputs)):
            print(_render_output(i, len(prompts), prompt, output, False, args.show_thinking))
            raw_outputs.append(output)
        print(f"\n{SEP}")
        print(f"Generated {len(prompts)} response(s) | {total_output_tokens} tokens | "
              f"{elapsed:.2f}s | \033[1;32m{res['batch_decode_tps']:.1f} tok/s agg\033[0m")
        print(f"COLOSSUS tier={res['placement']} C={res['capacity']} | "
              f"hits={res['total_hits']} misses={res['total_misses']} | "
              f"DMA={res['total_dma_mb']:.1f} MB | prefill={res['prefill_s']:.1f}s | "
              f"Peak VRAM: {res['peak_vram_gb']:.2f} GB")
        if res.get("seed_loads"):
            print(f"SEED: prefill-seeded {res['seed_loads']} slot loads")
        if res.get("hot_dma_mb") is not None:
            print(f"COLUMN f={res['hot_frac']} | "
                  f"hot={res['hot_dma_mb']:.1f} MB "
                  f"cold={res['cold_dma_mb']:.1f} MB "
                  f"(cold_hits={res.get('cold_hits', 0)}, "
                  f"stall={res.get('stall_ms', 0.0):.0f} ms)")
        if remote:
            st = handles.stats()
            print(f"REMOTE: net={st['network_bytes'] / 1e9:.2f} GB fetched, "
                  f"disk={st['disk_bytes'] / 1e9:.2f} GB reread, "
                  f"reqs={st['requests']}, "
                  f"cache={st['cache_bytes'] / 1e9:.2f}/{st['cache_cap_bytes'] / 1e9:.0f} GB "
                  f"(hit {st['hit_rate'] * 100:.1f}%)")
        if args.output_file:
            out_path = Path(args.output_file)
            with open(out_path, "w") as f:
                for raw in raw_outputs:
                    f.write(raw.replace("\n", "\\n") + "\n")
            logger.info(f"Raw outputs written to {args.output_file}")
        return

    # ── Engine ───────────────────────────────────────────────────────
    from bhaskera.inference import InferenceEngine
    engine = InferenceEngine(cfg)
    engine.load()

    # ── Log active settings ───────────────────────────────────────────
    infer = cfg.inference
    logger.info(
        f"Settings: kv_cache={infer.kv_cache!r} "
        f"temperature={infer.temperature} top_p={infer.top_p} "
        f"max_new_tokens={infer.max_new_tokens} "
        f"speculative={infer.speculative.enabled}"
    )
    if infer.kv_cache == "turboquant":
        logger.info(
            f"TurboQuant: K{infer.turboquant.key_bits}/V{infer.turboquant.value_bits} bits, "
            f"residual_window={infer.turboquant.residual_window}, "
            f"protected_layers={infer.turboquant.protected_layers}"
        )

    is_thinking = getattr(engine._backend, "_is_thinking", False) if engine._loaded else False

    # Try to get tokenizer for accurate token counting
    _tokenizer = None
    try:
        _tokenizer = getattr(engine._backend, "_tok", None)
    except Exception:
        pass

    # ── Generate ─────────────────────────────────────────────────────
    t0 = time.perf_counter()

    outputs = engine.generate(
        prompts,
        max_new_tokens = args.max_new_tokens or infer.max_new_tokens,
        temperature    = args.temperature or infer.temperature,
        top_p          = args.top_p or infer.top_p,
        top_k          = args.top_k or infer.top_k,
        do_sample      = not args.no_sample and infer.do_sample,
        return_full_text = args.return_full,
    )

    elapsed = time.perf_counter() - t0

    # ── Count tokens ──────────────────────────────────────────────────
    total_output_tokens = sum(_count_output_tokens(o, _tokenizer) for o in outputs)
    tokens_per_second   = total_output_tokens / elapsed if elapsed > 0 else 0.0

    # ── Print results ──────────────────────────────────────────────────
    raw_outputs = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        rendered = _render_output(
            i, len(prompts), prompt, output,
            is_thinking_model=is_thinking,
            show_thinking=args.show_thinking,
        )
        print(rendered)
        raw_outputs.append(output)

    # ── Stats block ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print(
        f"Generated {len(prompts)} response(s) | "
        f"{total_output_tokens} tokens | "
        f"{elapsed:.2f}s | "
        f"\033[1;32m{tokens_per_second:.1f} tok/s\033[0m"
    )

    # KV cache stats (TurboQuant)
    stats = engine.kv_cache_stats()
    if stats and stats.get("compression_ratio", 0) > 0:
        print(
            f"TurboQuant KV cache: {stats['tq_mb']:.1f} MB "
            f"(bf16 baseline: {stats['bf16_mb']:.1f} MB, "
            f"ratio: {stats['compression_ratio']:.1f}×)"
        )
    elif infer.kv_cache == "turboquant":
        # Cache exists but was bypassed (e.g. Param2) — still note it
        print(f"TurboQuant: active (model uses internal cache)")

    # Peak VRAM
    try:
        import torch
        if torch.cuda.is_available():
            peak_vram_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"Peak VRAM: {peak_vram_gb:.2f} GB")
    except Exception:
        pass

    # COLOSSUS MoE offload / dynamic cache stats
    cstats = engine.colossus_status()
    if cstats and cstats.get("mode") != "off":
        print(
            f"COLOSSUS MoE: mode={cstats.get('mode')} | "
            f"layers={cstats.get('layers_hooked')} | "
            f"steps={cstats.get('shadow_steps')} | "
            f"hits={cstats.get('hits_count', 0)} | "
            f"misses={cstats.get('misses_count', 0)}"
        )
        dc = cstats.get("dynamic_cache")
        if dc:
            print("=" * 80)
            print(f"COLOSSUS DYNAMIC CACHE SUMMARY (Capacity C={dc['capacity']} of 64 Experts across {dc['layers_wrapped']} Layers):")
            print(f"  Hit Rate     : {dc['hit_rate_pct']:.1f}% ({dc['hits']} hits / {dc['misses']} demand misses)")
            print(f"  ZSSR Recall@6: {dc['recall_pct']:.1f}%")
            print(f"  PCIe DMA     : {dc['prefetch_mb']:.1f} MB Prefetched | {dc['demand_mb']:.1f} MB Demand Fetched")
            tot_pcie_s = dc.get("pcie_time_s", 0.0)
            dem_stall_s = dc.get("demand_stall_s", 0.0)
            pref_stall_s = dc.get("prefetch_stall_s", 0.0)
            hidden_s = max(0.0, tot_pcie_s - dem_stall_s - pref_stall_s)
            overlap_pct = (hidden_s / tot_pcie_s * 100.0) if tot_pcie_s > 0 else 0.0
            print(f"  PCIe DMA Time: {tot_pcie_s:.2f}s total | Demand Stall: {dem_stall_s:.2f}s | Prefetch Stall: {pref_stall_s:.2f}s | Hidden: {hidden_s:.2f}s ({overlap_pct:.1f}% Overlap)")
            if "missing_col_mean" in dc:
                dma_mb_tok = dc.get("dma_bytes_per_tok", 0.0) / (1024 * 1024)
                print(f"  Col Missing  : mean={dc['missing_col_mean']:.1f}% | p50={dc['missing_col_p50']:.1f}% | p90={dc['missing_col_p90']:.1f}% | p95={dc['missing_col_p95']:.1f}% | max={dc['missing_col_max']:.1f}% | DMA: {dma_mb_tok:.2f} MB/tok")
            print("-" * 80)
            print(f"{'Layer':>6} | {'C':>3} | {'Hit Rate':>9} | {'Hits':>6} | {'Misses':>6} | {'Prefetch MB':>12} | {'Demand MB':>10} | {'Recall@6':>9} | {'DemStall':>8}")
            print("-" * 80)
            for m in dc.get("layers", []):
                print(f"{m['layer_idx']:6d} | {m['capacity']:3d} | {m['hit_rate_pct']:8.1f}% | {m['hits']:6d} | {m['misses']:6d} | {m['prefetch_mb']:11.1f} | {m['demand_mb']:9.1f} | {m['recall_pct']:8.1f}% | {m.get('demand_stall_s', 0.0):7.2f}s")
            print("=" * 80)
        else:
            # Legacy offload stats
            offload = cstats.get("offload", {})
            if offload.get("mode") == "active":
                print(
                    f"COLOSSUS offload: "
                    f"{offload.get('vram_saved_gb', 0):.2f} GB freed | "
                    f"{offload.get('offload_ratio', 0)*100:.0f}% experts offloaded | "
                    f"{offload.get('experts_offloaded', 0)} experts on CPU"
                )

    # Thinking model note
    if is_thinking and not args.show_thinking:
        print("(Thinking/reasoning block hidden — use --show-thinking to display)")

    # ── Optional file output ───────────────────────────────────────────
    if args.output_file:
        out_path = Path(args.output_file)
        with open(out_path, "w") as f:
            for raw in raw_outputs:
                f.write(raw.replace("\n", "\\n") + "\n")
        logger.info(f"Raw outputs written to {out_path}")


if __name__ == "__main__":
    main()