# Claude API vs Local Ollama Models
## Data Security & Performance Analysis

**Date:** 2026-02-19
**Applies to:** AI Ready RAG — VE-RAG-System

---

## Overview

AI Ready RAG is architected for **air-gap deployment** on NVIDIA DGX Spark using local Ollama models. This document analyzes the trade-offs involved in replacing local Ollama inference with the Anthropic Claude API, covering data security, privacy, and quality/performance dimensions.

This is not a binary choice. A hybrid approach — local embeddings with API-powered LLM inference — is a viable middle path for non-air-gapped deployments.

---

## 1. Data Security Comparison

### 1.1 Data Residency

| Dimension | Ollama (Local) | Claude API |
|-----------|---------------|------------|
| **Where data is processed** | On-premises (DGX Spark) | Anthropic's AWS/GCP infrastructure (primarily US) |
| **Document content leaves premises** | ❌ Never | ✅ Yes — every RAG query sends retrieved chunks as context |
| **Air-gap compatible** | ✅ Yes | ❌ No |
| **Data residency control** | Complete | Geographic region selection (US, EU, APAC) via cross-region inference |
| **Network dependency** | None | Internet connectivity required |

**Key implication:** In the current architecture, every chat query sends 5–8 document chunks to the LLM. With the Claude API, those chunks — which may contain D&O policy terms, financial statements, HR data, or client information — are transmitted to Anthropic's infrastructure on every request.

### 1.2 Training Data Usage

| Account Type | Used for Model Training | Notes |
|---|---|---|
| **Ollama (local)** | ❌ Never — model runs locally | Complete isolation |
| **Claude API (commercial)** | ❌ No by default | Must explicitly opt in to Developer Partner Program |
| **Claude.ai Enterprise** | ❌ No by default | Governed by Commercial Terms of Service |
| **Claude.ai Teams** | ⚠️ Yes by default | Must opt out at claude.ai/settings/data-privacy-controls |
| **Claude.ai Free/Pro** | ✅ Yes by default | Consumer terms apply |

**For corporate deployments, only the commercial API or Enterprise plan provides contractual no-training guarantees without manual opt-out.**

### 1.3 Data Retention

| Model | Retention Period |
|-------|-----------------|
| **Ollama (local)** | Zero — no external logging |
| **Claude API (standard)** | 7 days |
| **Claude API (Zero Data Retention)** | Immediate discard after processing — available via API contract with Anthropic sales |
| **Claude.ai Enterprise** | 30 days default; configurable minimum |
| **Claude.ai Teams** | 30 days (or 5 years if training opt-in is active) |

### 1.4 Compliance & Certifications

| Certification | Ollama (Local) | Claude API (Anthropic) |
|---|---|---|
| **SOC 2 Type II** | Depends on your infrastructure | ✅ Anthropic certified |
| **ISO 27001:2022** | Depends on your infrastructure | ✅ Anthropic certified |
| **ISO/IEC 42001** | N/A | ✅ Anthropic certified |
| **HIPAA BAA** | Depends on your infrastructure | ✅ Available on Enterprise |
| **GDPR / SCCs** | Your responsibility | ✅ DPA with Standard Contractual Clauses included |
| **Data Processing Agreement** | N/A | ✅ Auto-included with commercial terms |

### 1.5 Access Control & Audit

Both approaches support the system's pre-retrieval access control architecture (tag-based filtering before vector search). This layer is application-level and independent of the LLM backend.

| Control | Ollama (Local) | Claude API |
|---------|---------------|------------|
| **Pre-retrieval tag filtering** | ✅ Enforced at application layer | ✅ Enforced at application layer (same) |
| **Prompt/response audit logging** | Application-level only | Anthropic retains logs per retention policy |
| **Who can see your prompts** | No one (local) | Anthropic (per their privacy policy; never sold or shared) |
| **Contractual data protection** | N/A | DPA — customer is data controller, Anthropic is processor |

---

## 2. Performance & Quality Benefits

### 2.1 Model Capability Comparison

| Capability | `qwen3-rag` (local 8B) | `deepseek-r1-rag` (local 14B) | Claude Sonnet 4.6 (API) |
|---|---|---|---|
| **Parameters** | 8 billion | 14 billion | ~200B+ (undisclosed) |
| **Context window** | 8,192 tokens | 16,384 tokens | **200,000 tokens** |
| **Reasoning depth** | Good | Strong | Excellent |
| **Instruction following** | Good | Good | Excellent |
| **Citation accuracy** | Good (with RAG tuning) | Good | Excellent |
| **Multi-document synthesis** | Limited by context | Moderate | Native at full scale |
| **Hallucination rate** | Moderate | Lower | Low |
| **Structured output** | Good | Good | Excellent |
| **Latency** | 5–20s (local GPU) | 15–40s (local GPU) | 2–8s (API) |
| **Cost** | Hardware amortized | Hardware amortized | Per-token billing |

### 2.2 Context Window Impact on RAG Quality

The most significant quality difference is the **context window**. The current system works around local model limitations through:
- `rag_max_chunks_per_doc = 5` (recently increased from 3)
- `rag_total_context_chunks = 8`
- `rag_max_context_tokens = 6,000`
- Query expansion and intent boosting to improve retrieval precision
- Hallucination checks and confidence scoring

With Claude's **200,000-token context window**, most of these constraints become optional:

| Current Limitation | With Local 8B Model | With Claude API |
|---|---|---|
| Chunks per document | 5 max | Could send all 171 chunks of a D&O policy |
| Total context chunks | 8 max | 100+ chunks feasible |
| Context token budget | 6,000 tokens | 180,000+ tokens available for context |
| Multi-document comparison | Limited | Full documents side-by-side |
| Deep policy extraction (Q6) | Partial (3-5 exclusions found) | Comprehensive (full policy analyzed) |
| Cross-document synthesis | Requires careful chunking | Natural — all docs in one context |

**Practical example:** The D&O policy document with 171 chunks and 45 sections mentioning exclusions could be passed in its entirety to Claude, eliminating the retrieval precision problem entirely for single-document deep-dive queries.

### 2.3 Specific Use Case Improvements

#### Insurance Policy Analysis
- **Local:** Retrieves 5 most similar chunks; misses clauses not semantically close to query
- **API:** Full policy in context; no retrieval gaps; comprehensive exclusions, conditions, and endorsements surfaced

#### Financial Document Q&A (Excel)
- **Local:** `qwen2.5:7b` for classification, `deepseek-r1:14b` for reasoning — two separate models, two VRAM slots
- **API:** Single model handles classification and reasoning with superior accuracy; no VRAM contention

#### Cross-Document Comparison
- **Local:** Limited to 8 chunks across all documents; struggles with "compare policy A vs policy B"
- **API:** Both documents fully in context; direct comparison natural

#### Auto-Tagging
- **Local:** `qwen3-rag` (just fixed from `qwen3:8b`) — adequate for simple tag matching
- **API:** Better semantic understanding of document content; more accurate tag suggestions, especially for nuanced categories

### 2.4 Embedding Models

Embeddings are less impacted by the API vs local distinction. `nomic-embed-rag` (local) performs well for semantic search. The primary quality lever for retrieval is model quality, not embedding model choice. Anthropic does not currently offer a standalone embedding API.

**Recommendation:** Keep local embeddings regardless of LLM choice. Embeddings are small (274MB), fast, and have no meaningful quality gap vs cloud alternatives at this scale.

---

## 3. Architecture Options

### Option A: Full Local (Current — Air-Gap)
```
User Query → Local Embeddings → Qdrant → qwen3-rag (Ollama) → Response
```
- ✅ Air-gap compliant
- ✅ Zero data egress
- ✅ No per-query cost
- ❌ Limited context window
- ❌ Lower reasoning quality on complex policy questions

### Option B: Full API (Cloud Deployment)
```
User Query → Local Embeddings → Qdrant → Claude API → Response
```
- ✅ Best quality and context window
- ✅ Commercial DPA, no training on data
- ✅ Lower latency than local 14B model
- ❌ Not air-gap compatible
- ❌ Per-token cost
- ❌ Document chunks transmitted externally

### Option C: Hybrid (Recommended for Non-Air-Gap)
```
User Query → Local Embeddings (nomic-embed-rag) → Qdrant → Claude API → Response
             ↑ stays local                        ↑ stays local
```
- ✅ Only retrieved chunks transmitted (not full documents)
- ✅ Best reasoning quality
- ✅ Embeddings and access control fully local
- ✅ Commercial DPA protections
- ❌ Retrieved chunk content still transmitted per query
- ❌ Not air-gap compatible

### Option D: Tiered by Query Type
```
Simple queries     → qwen3-rag (local, fast, free)
Complex policy Q&A → Claude API (higher quality, per-token cost)
```
Requires query complexity classifier (could use intent detection already in the system).

---

## 4. Cost Considerations

### Local Ollama (Current)
- **Upfront:** DGX Spark hardware (~$3,000–$10,000 depending on SKU)
- **Per-query cost:** $0 (hardware amortized)
- **Operational:** Power, maintenance
- **Scaling:** Hardware-bound

### Claude API (Sonnet 4.6 pricing as of 2026)
- **Input:** ~$3 per million tokens
- **Output:** ~$15 per million tokens
- **Typical RAG query:** ~2,000–6,000 input tokens + ~500 output tokens ≈ **$0.01–0.025 per query**
- **At 1,000 queries/day:** ~$10–25/day, ~$300–750/month

For enterprise document Q&A at moderate volume, API cost is typically far below the value of improved answer quality.

---

## 5. Recommendation by Deployment Scenario

| Scenario | Recommended Approach | Rationale |
|---|---|---|
| **DGX Spark / air-gap production** | Local Ollama (`qwen3-rag`) | Non-negotiable — data cannot leave premises |
| **Cloud pilot / demo environment** | Claude API (Hybrid Option C) | Best quality for demos; DPA protects data |
| **Regulated industry (insurance, finance)** | Local Ollama OR Enterprise API + legal review | Data residency and regulatory requirements vary; get legal sign-off before using API with client documents |
| **Internal knowledge base (non-client data)** | Claude API | Low sensitivity; quality uplift worth it |
| **Complex policy analysis at scale** | Claude API with ZDR | 200K context eliminates chunking limitations |
| **Cost-sensitive high-volume** | Local Ollama | Per-query cost adds up at scale |

---

## 6. Data Security Summary for Corporate Use

If evaluating the Claude API for corporate deployment, the key questions to answer with your legal/IT teams:

1. **What plan?** Commercial API or Enterprise — not Teams. Teams uses consumer terms.
2. **DPA signed?** Auto-included with commercial terms; verify with Anthropic account team.
3. **Data residency required?** Use geographic cross-region inference to specify US/EU processing.
4. **Maximum privacy needed?** Request Zero Data Retention (ZDR) add-on — immediate discard, no logging.
5. **Client data in documents?** Get legal review on whether transmitting insurance policy excerpts or financial data to a third-party API is permissible under client agreements.
6. **HIPAA relevance?** Request Business Associate Agreement (BAA) from Anthropic.

---

## 7. Current System Configuration

For reference, the current production defaults on DGX Spark:

```
Chat model:          qwen3-rag (local, 5.2GB, 8K context)
Embedding model:     nomic-embed-rag (local, 274MB)
Auto-tagging model:  → inherits chat_model (qwen3-rag)
Summary model:       → inherits chat_model (qwen3-rag)
Context chunks:      8 max total, 5 max per document
Context tokens:      6,000 max
VRAM preload:        qwen3-rag on startup, keep_alive 24h
```

Switching to the Claude API requires changing `CHAT_MODEL` in the environment and replacing the Ollama inference calls in `rag_service.py` with the Anthropic SDK. The retrieval pipeline (embeddings, Qdrant, chunking, access control) remains unchanged.

---

*Document maintained by the AI Ready RAG team. Last updated: 2026-02-19.*
