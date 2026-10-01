# Phase B Spec: Remote-Backed Shard Handles (serve without full download)

Status: SPEC (no code). Target branch: research (`colossus-zssr`).
Explicitly out of the frozen PRs (they depend only on the
`ShardHandles.get_tensor()` seam, which this consumes unchanged).

## 1. Goal

Serve a sharded safetensors MoE model with only its working set downloaded:
router-demanded experts fetch on first miss (~17 GB of 471 GB for the
reference workload — a measured example, never a headline), then serve from
a persistent local cache. ZSSR prefetch composes as background network fetch.

Non-goals: executor changes, training, gossip integration, quantized formats,
changing any frozen PR file.

## 2. Interface (mirrors `ShardHandles`; duck-compatible)

```python
class RemoteShardHandles:
    def __init__(self, repo_id: str, revision: str, cache_dir: str,
                 cache_cap_gb: float, token: str | None = None): ...
    @classmethod
    def open(cls, repo_id: str, **kw) -> "RemoteShardHandles": ...
    @property
    def weight_map(self) -> dict[str, str]: ...   # key -> shard filename
    def get_tensor(self, key: str) -> torch.Tensor: ...
    def header(self, key: str) -> dict: ...
    def shards(self) -> list[str]: ...
    def stats(self) -> dict: ...
    # stats: network_bytes, disk_bytes, net_hits, net_misses,
    #        first_token_s (cold only), cache_hit_rate
```

Drop-in where `ShardHandles` is used today (`wrap` + executors unchanged).

## 3. Fetch protocol

Per shard, in order (each step cached in a JSON manifest, so repeats are free):

1. **Length prefix**: `GET Range: bytes=0-7` → header length `H`.
2. **Header**: `GET Range: bytes=8-(8+H)` → parse JSON → tensor table.
   `weight_map` derives from all shard headers (same shape as the local
   `model.safetensors.index.json`; optionally written to cache for reuse).
3. **Tensor bytes**: `GET Range: bytes=[data_start+b, data_start+e)` per
   `data_offsets`. Blob URLs resolved via `huggingface_hub` (CDN-backed;
   public repos need no token; private take `token`).

Only standard HTTP Range semantics required of the server.

## 4. Cache layout (`cache_dir/`)

```text
manifest.json            # {repo, revision, shards: {file: {header_len, tensors...}}}
tensors/<sha256(key)>.bin   # raw tensor bytes (exact data span, no header)
tensors/<sha256(key)>.sha   # expected sha256 (written at fetch)
lru.json                 # {key: last_access} for eviction
```

- Addressing by tensor bytes (not shard ranges): eviction is per-tensor,
  verification is per-tensor, partial shards never materialize.
- **Integrity: verify-once-cache-forever.** sha256 checked at fetch against
  the Hub's blob hash where available (else stored first-seen + warned);
  after verification, cache hits skip hashing (per-fetch hashing would
  dominate small-tensor latency — same fragmentation moral as C-1).
- **Eviction**: LRU by bytes under `cache_cap_gb` (default: min(200 GB,
  free-space − 50 GB)). Ledger reuses existing byte counters; network vs
  disk split reported separately, never merged.
- **Offline fallback**: fetch failure + cache miss = hard error naming the
  tensor (no silent wrong data). Full-local mode (plain `ShardHandles`)
  remains the default; remote is opt-in per run.

## 5. Local-first composition (follow-up, not Phase B)

`TieredHandles(local, remote)`: try local index first, remote on miss.
Lets warm caches seed from partial checkouts. Specified here so Phase B
doesn't paint it out (constructor takes an optional local fallback).

## 6. Measurement contract (cold vs warm NEVER blended)

| Metric              | Cold cache | Warm cache |
| ------------------- | ---------: | ---------: |
| Network bytes       |          ✓ |          ✓ |
| Disk bytes          |          ✓ |          ✓ |
| VRAM / RAM          |          ✓ |          ✓ |
| First-token latency |          ✓ |          ✓ |
| Decode tok/s        |          ✓ |          ✓ |
| Cache hit rate      |          — |          ✓ |
| Unique experts touched / total | ✓ | ✓ |

Headline metric (per workload): `fetched_bytes / unique_experts / total_bytes`.

## 7. Acceptance gates

1. **Correctness**: bit-identical tensors vs local `ShardHandles` over
   sampled keys (mock Range server in CI; live Hub in nightly).
2. **Warm parity**: reference workload tok/s within noise of the mmap path.
3. **Cold reporting**: first-token latency + network bytes published
   separately; no blended headline.
4. **No-executor-diff**: `git diff` on serving code must be empty
   (consumer-only change; verified by the existing 59-test suite).
5. **Failure clarity**: offline + uncached = named hard error (tested).

## 8. Test plan (CPU-only, no network in CI)

Mock HTTP server with Range support serving a fake sharded model:
fetch exactness vs direct read; second call performs zero HTTP (cache hit);
eviction under tiny cap refetches correctly; offline mode raises the named
error; manifest reuse skips header round-trips. Live-Hub checks run
manually, never in CI.
