# DGX Spark Performance Optimization Guide

**Version:** 1.0
**Date:** 2026-02-18
**Hardware:** NVIDIA DGX Spark (Grace Blackwell GB10, 128GB unified LPDDR5x, 273 GB/s bandwidth)
**Application:** AI Ready RAG v0.5.0

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Root Cause Analysis](#root-cause-analysis)
3. [Ollama Settings](#1-ollama-environment-settings)
4. [Model Context Tuning](#2-model-context-window-tuning)
5. [DGX Spark CPU/Thread Tuning](#3-dgx-spark-cputhread-tuning)
6. [Application Concurrency Settings](#4-application-concurrency-settings)
7. [Qdrant Performance Settings](#5-qdrant-performance-settings)
8. [Memory Budget](#6-memory-budget-128gb)
9. [Implementation Priority](#7-implementation-priority)
10. [Verification](#8-verification)
11. [Sources](#sources)

---

## Problem Statement

During bulk document uploads (100+ files), the server becomes unresponsive. Chat queries timeout, the API returns HTTP 504/timeout errors, and document processing stalls. The system recovers after the queue drains but is unusable during peak ingestion.

## Root Cause Analysis

The primary bottleneck is **Ollama's default `NUM_PARALLEL=1`** combined with the application firing 100+ concurrent embedding requests via `asyncio.gather`. Each document triggers multiple Ollama calls:

1. **Embedding generation** — one call per chunk (100+ chunks per document)
2. **Summary generation** — one LLM call per document
3. **LLM auto-tagging** — one LLM call per document
4. **Chat queries** — concurrent user requests

With `NUM_PARALLEL=1`, Ollama queues all but one request. The httpx client times out at 60s before Ollama processes the queued requests. The problem is **concurrency starvation**, not memory or GPU capacity.

---

## 1. Ollama Environment Settings

These are the highest-impact changes. Apply via systemd override.

| Variable | Default | Recommended | Rationale |
|---|---|---|---|
| `OLLAMA_NUM_PARALLEL` | `1` | `4` | Allow 4 concurrent inference requests. 128GB unified memory handles this easily with q8_0 KV cache. |
| `OLLAMA_MAX_LOADED_MODELS` | auto (3) | `3` | Keep all 3 models loaded: qwen3:8b (~5GB), nomic-embed-text (~275MB), deepseek-r1:14b (~9GB). Eliminates model swap latency. |
| `OLLAMA_FLASH_ATTENTION` | `0` | `1` | Free performance boost on Blackwell GPU. Reduces memory usage for attention computation. No downside on this hardware. |
| `OLLAMA_KV_CACHE_TYPE` | `f16` | `q8_0` | Cuts KV cache memory ~50% with minimal quality loss. Use `q8_0` not `q4_0` to preserve RAG citation quality. |
| `OLLAMA_MULTIUSER_CACHE` | `false` | `true` | RAG sends similar system prompts across sessions. Shares prefix cache across parallel slots, reducing redundant computation. |
| `OLLAMA_KEEP_ALIVE` | `5m` | `24h` | Dedicated appliance — keep models hot. Model load/unload is the most expensive operation. |
| `OLLAMA_GPU_OVERHEAD` | `0` | `536870912` | Reserve 512MB for Qdrant, system overhead, and CUDA allocations on the unified memory bus. |
| `OLLAMA_LOAD_TIMEOUT` | `5m` | `10m` | deepseek-r1:14b takes time to load. Prevents timeout during cold start. |

### Apply via systemd

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d

sudo tee /etc/systemd/system/ollama.service.d/override.conf << 'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=4"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
Environment="OLLAMA_MULTIUSER_CACHE=true"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="OLLAMA_GPU_OVERHEAD=536870912"
Environment="OLLAMA_LOAD_TIMEOUT=10m"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

### Verify

```bash
# Confirm settings applied
curl -s http://localhost:11434/api/version
# Check loaded models
ollama ps
```

---

## 2. Model Context Window Tuning

Models default to large context windows that waste KV cache memory. The RAG application uses far less.

| Model | Default ctx | Recommended | Rationale |
|---|---|---|---|
| qwen3:8b | 32768 | **8192** | RAG token budget: 6000 context + 1500 history + 500 system = 8K max |
| nomic-embed-text | 8192 | **2048** | Chunks are ~200 tokens with 40 overlap. 2K is plenty. |
| deepseek-r1:14b | 65536 | **16384** | Used for auto-tag classification, not full documents. 16K is sufficient. |

### Create optimized model variants

```bash
# Chat model — reduced context, pinned to performance cores
ollama create qwen3-rag -f - <<'EOF'
FROM qwen3:8b
PARAMETER num_ctx 8192
PARAMETER num_thread 10
PARAMETER num_gpu 999
EOF

# Embedding model — minimal context for chunk-sized inputs
ollama create nomic-embed-rag -f - <<'EOF'
FROM nomic-embed-text
PARAMETER num_ctx 2048
PARAMETER num_thread 10
EOF

# Reasoning model — moderate context for classification tasks
ollama create deepseek-r1-rag -f - <<'EOF'
FROM deepseek-r1:14b
PARAMETER num_ctx 16384
PARAMETER num_thread 10
PARAMETER num_gpu 999
EOF
```

### Update application config

In `.env` on the Spark:

```bash
CHAT_MODEL=qwen3-rag
EMBEDDING_MODEL=nomic-embed-rag
EXCEL_CLASSIFICATION_MODEL=qwen3-rag
EXCEL_REASONING_MODEL=deepseek-r1-rag
```

---

## 3. DGX Spark CPU/Thread Tuning

The DGX Spark has an asymmetric CPU: **10 Cortex-X925 performance cores** + **10 Cortex-A725 efficiency cores**. Mixing them degrades performance because the efficiency cores run at lower clock speeds, causing thread synchronization stalls.

| Variable | Default | Recommended | Rationale |
|---|---|---|---|
| `OMP_NUM_THREADS` | auto (20) | `10` | Use only the 10 high-performance Cortex-X925 cores. Per NVIDIA porting guide. |
| `OMP_THREAD_LIMIT` | auto | `1` | Already set in `start-servers.sh`. Prevents Tesseract OpenMP threads from stacking with ARQ workers. |
| `OMP_SCHEDULE` | static | `guided` | Better workload balancing for heterogeneous core architecture. |
| `CUDA_VISIBLE_DEVICES` | auto | `0` | Explicit single GPU. Prevents library confusion on ARM/unified memory. |
| `MALLOC_ARENA_MAX` | auto (8*cores) | `4` | Limits glibc memory arena fragmentation for long-running Python services. |
| `PYTORCH_CUDA_ALLOC_CONF` | unset | `expandable_segments:True` | Reduces CUDA memory fragmentation for PyTorch ops (Docling, fastembed). |

### Add to `.env`

```bash
OMP_NUM_THREADS=10
OMP_SCHEDULE=guided
CUDA_VISIBLE_DEVICES=0
MALLOC_ARENA_MAX=4
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

These are loaded by `start-servers.sh` via `export $(grep -v "^#" .env | grep -v "^$" | xargs)`.

---

## 4. Application Concurrency Settings

### Config changes (`config.py` / `.env`)

| Setting | Current | Recommended | Rationale |
|---|---|---|---|
| `arq_max_jobs` | `2` | `3` | With `OLLAMA_NUM_PARALLEL=4`, 3 concurrent document processing jobs is safe. |
| `max_concurrent_processing` (spark) | `2` | `3` | Match `arq_max_jobs`. 20 CPU cores handle 3 concurrent Docling/OCR pipelines. |
| `rag_timeout_seconds` | `30` | `60` | Under load, Ollama queues requests. Need more headroom before timeout. |
| `warming_max_concurrent_queries` | `2` | `3` | With `NUM_PARALLEL=4`, warming uses 3 slots, leaving 1 for interactive chat. |

### Critical code fix: Embedding thundering herd

The `_embed_batch_with_retry` method in `vector_service.py` fires ALL chunk embeddings simultaneously via `asyncio.gather`. With 100 chunks, this creates 100 concurrent Ollama requests against `NUM_PARALLEL=4`, causing massive queuing and timeouts.

**Fix:** Add a semaphore to limit concurrent Ollama embedding calls:

```python
# In VectorService.__init__:
self._embed_semaphore = asyncio.Semaphore(4)  # Match OLLAMA_NUM_PARALLEL

# New throttled method:
async def _throttled_embed(self, text: str) -> list[float]:
    async with self._embed_semaphore:
        return await self.embed(text)

# In _embed_batch_with_retry, replace:
#   tasks = [self.embed(text) for text in texts]
# with:
    tasks = [self._throttled_embed(text) for text in texts]
    return await asyncio.gather(*tasks)
```

This limits concurrent Ollama calls to 4, matching the server's parallel capacity. The remaining requests wait on our side (zero-cost) instead of timing out in Ollama's queue.

---

## 5. Qdrant Performance Settings

| Setting | Current | Recommended | Rationale |
|---|---|---|---|
| `hnsw_ef_construct` | `100` | `200` | Higher build quality for better recall. Index is built once during upload; query speed matters more. |
| `hnsw_m` | `16` | `16` | Keep at 16. Sweet spot per Qdrant docs. Higher values risk broken HNSW graphs. |
| `hnsw_ef` (search) | `128` | `128` | Good default for RAG precision. Increase to 256 only if recall is insufficient. |
| Quantization | None | **Scalar int8** | Reduces vector memory 75%, speeds up distance calculations. With 128GB RAM, use `always_ram: true` for rescoring. |
| `on_disk` | `false` | `false` | 128GB unified memory — keep everything in RAM. |
| `indexing_threshold` | `20000` | `50000` | Delay HNSW rebuilds during bulk upload. Rebuild after batch completes. |

### Apply quantization to existing collection

```bash
curl -X PUT http://localhost:6333/collections/documents \
  -H "Content-Type: application/json" \
  -d '{
    "quantization_config": {
      "scalar": {
        "type": "int8",
        "always_ram": true
      }
    },
    "optimizers_config": {
      "indexing_threshold": 50000
    }
  }'
```

### For new collection creation

```python
await client.create_collection(
    collection_name="documents",
    vectors_config=models.VectorParams(
        size=768,
        distance=models.Distance.COSINE,
        on_disk=False,
    ),
    hnsw_config=models.HnswConfigDiff(
        m=16,
        ef_construct=200,
    ),
    quantization_config=models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            always_ram=True,
        ),
    ),
    optimizers_config=models.OptimizersConfigDiff(
        default_segment_number=2,
        indexing_threshold=50000,
        memmap_threshold=200000,
    ),
)
```

---

## 6. Memory Budget (128GB)

| Component | Estimated Usage |
|---|---|
| qwen3:8b (q4_K_M) + KV cache (8K ctx, 4 slots, q8_0) | ~7 GB |
| nomic-embed-text + KV cache (2K ctx, 4 slots) | ~0.5 GB |
| deepseek-r1:14b (q4_K_M) + KV cache (16K ctx, 4 slots, q8_0) | ~12 GB |
| Qdrant (vectors + HNSW index, ~100K chunks, int8 quantization) | ~1 GB |
| Docling + Tesseract OCR (3 concurrent) | ~3 GB |
| Python / FastAPI / Redis / ARQ | ~1 GB |
| OS + GPU overhead reservation | ~2.5 GB |
| **Total estimated** | **~27 GB** |
| **Headroom remaining** | **~101 GB** |

The system has massive headroom. The bottleneck is concurrency management, not memory capacity.

---

## 7. Implementation Priority

### Phase 1: Immediate (no code changes)

1. Set Ollama environment variables via systemd override
2. Create optimized model variants with reduced `num_ctx`
3. Update `.env` with thread tuning and model names
4. Restart Ollama and application

**Expected impact:** 4x throughput improvement, eliminates timeouts during bulk upload.

### Phase 2: Code fixes

5. Add embedding semaphore to `vector_service.py` (prevents thundering herd)
6. Increase `arq_max_jobs` and `max_concurrent_processing` to 3
7. Increase `rag_timeout_seconds` to 60

**Expected impact:** Smooth concurrent processing, no more embedding timeouts.

### Phase 3: Qdrant tuning

8. Apply scalar int8 quantization
9. Increase `hnsw_ef_construct` to 200
10. Set `indexing_threshold` to 50000

**Expected impact:** Faster search, reduced memory footprint, faster bulk indexing.

---

## 8. Verification

After applying changes, verify with:

```bash
# 1. Ollama settings active
curl -s http://localhost:11434/api/version

# 2. All models loaded
ollama ps
# Should show 3 models loaded simultaneously

# 3. Health check responds under load
curl -s http://localhost:8502/api/health | python3 -m json.tool

# 4. Bulk upload test (10 files)
# Upload 10 mixed-format files, verify:
# - Server remains responsive (health check < 2s)
# - No embedding timeouts in logs
# - All documents reach "ready" status
# - Chat queries work during processing

# 5. Monitor during bulk upload
watch -n 5 'ollama ps; echo "---"; curl -s http://localhost:8502/api/health | python3 -c "import sys,json; print(json.load(sys.stdin)[\"status\"])"'
```

---

## Sources

- [Ollama Performance Tuning: GPU Optimization Techniques](https://collabnix.com/ollama-performance-tuning-gpu-optimization-techniques-for-production/)
- [Ollama envconfig/config.go (canonical env var reference)](https://github.com/ollama/ollama/blob/main/envconfig/config.go)
- [Ollama FAQ](https://docs.ollama.com/faq)
- [Ollama Modelfile Reference](https://docs.ollama.com/modelfile)
- [Ollama NVIDIA DGX Spark Performance Blog](https://ollama.com/blog/nvidia-spark-performance)
- [How Ollama Handles Parallel Requests](https://www.glukhov.org/post/2025/05/how-ollama-handles-parallel-requests/)
- [K/V Cache Quantization in Ollama](https://mitjamartini.com/posts/ollama-kv-cache-quantization/)
- [NVIDIA DGX Spark Hardware Overview](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
- [DGX Spark Porting Guide - Optimization](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/optimization.html)
- [New Software Optimizations for DGX Spark (NVIDIA Developer Blog)](https://developer.nvidia.com/blog/new-software-and-model-optimizations-supercharge-nvidia-dgx-spark/)
- [LMSYS DGX Spark In-Depth Review](https://lmsys.org/blog/2025-10-13-nvidia-dgx-spark/)
- [ARM DGX Spark LLM Performance Guide](https://learn.arm.com/learning-paths/laptops-and-desktops/dgx_spark_llamacpp/1_gb10_introduction/)
- [Qdrant Performance Optimization Guide](https://qdrant.tech/documentation/guides/optimize/)
- [Qdrant Indexing Optimization for Bulk Uploads](https://qdrant.tech/articles/indexing-optimization/)
