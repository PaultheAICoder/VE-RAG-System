# VE-RAG-System Competitive Assessment: Insurance Document Extraction

**Date:** February 18, 2026
**Version:** 1.0
**Prepared by:** AI Architecture Review Team
**Classification:** Internal — Confidential

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Document Corpus Analyzed](#2-document-corpus-analyzed)
3. [VE-RAG-System Current Capabilities](#3-ve-rag-system-current-capabilities)
4. [Extraction Quality by Document Type](#4-extraction-quality-by-document-type)
5. [Open Source Competitive Landscape](#5-open-source-competitive-landscape)
6. [Commercial / Closed Source Competitive Landscape](#6-commercial--closed-source-competitive-landscape)
7. [Air-Gap Compatibility Matrix](#7-air-gap-compatibility-matrix)
8. [Insurance-Specific Extraction Analysis](#8-insurance-specific-extraction-analysis)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Gap Priority Matrix](#10-gap-priority-matrix)
11. [Recommended Roadmap](#11-recommended-roadmap)
12. [Pricing Analysis](#12-pricing-analysis)
13. [Conclusions](#13-conclusions)
14. [Sources](#14-sources)

---

## 1. Executive Summary

This assessment evaluates VE-RAG-System's ability to extract meaningful, repeatable data from HOA (Homeowners Association) insurance documents and compares it against 22 open-source and commercial solutions. The evaluation focuses on extraction accuracy, structured data repeatability, insurance domain understanding, and air-gap deployment compatibility.

### Key Findings

- **VE-RAG-System composite score: 6.3/10** — strong deployment model and cost profile, with meaningful gaps in extraction sophistication and query capability for insurance documents.
- **No tool — open source or commercial — solves insurance domain understanding out of the box** (except Indico Data at $100K+/yr). VE-RAG already has more insurance domain awareness than most competitors through its intent patterns, auto-tagging strategy, and ACORD 25 field-group chunking.
- **Docling remains the best air-gap-compatible parser**. Its 97.9% table accuracy, MIT license, and existing integration make it the right core choice.
- **Phase 1 quick wins (1-2 weeks)** can raise the composite score from 6.3 to ~7.2, closing nearly half the gap to best-in-class without new model training or infrastructure changes.
- **Reducto on-premise** is the top commercial recommendation for supplementary extraction of complex tables, with proven air-gap deployment in government environments.

---

## 2. Document Corpus Analyzed

**Source:** 18 PDFs across 3 HOA insurance client folders

### Client Folders

| Client | Subfolder Structure | Document Count |
|--------|-------------------|----------------|
| Bethany Terrace (12-13) | 24 NB/Policy, 25 Renewal/Policy, 25 Renewal/Quote, 25 Renewal/Sub, Docs | 8 |
| Cervantes (12-01) | 25 Renewal/Policy, 25 Renewal/Quote, 25 Renewal/Sub | 6 |
| Walnut Creek (10-23) | 24 NB/Policy, 25 Renewal/Policy, 25 Renewal/Quote | 4 |

### Document Types

| Document Type | Count | Size Range | Characteristics |
|---------------|-------|-----------|-----------------|
| **Coverage Summaries** | 3 | 68-88 KB | Multi-column comparison tables (prior year vs. renewal vs. competitor). Currency values, yes/no fields, "No Coverage"/"Not Offered" categorical values, premium breakdowns. |
| **D&O Policies** | 4 | 300-864 KB | Directors & Officers insurance policies. Multi-page legal documents with declarations, endorsements, exclusions. |
| **LIO Policies** | 2 | 1.7-1.8 MB | Liability Insurance Organization policies. 50+ page policy documents with table of contents, coverage forms, conditions. |
| **Package Policies** | 1 | 2.1 MB | Combined coverage policy. Largest document in the corpus. Multiple coverage sections consolidated. |
| **Certificates of Insurance** | 3 | 112-132 KB | ACORD-style certificate forms with standardized field positions. Producer info, insured info, coverage limits. |
| **Loss Run Reports** | 1 | 32 KB | Claims history report. Policy numbers, insured name/address, date ranges, claim status ("No Claims on File"). |
| **Reserve Studies** | 1 | 852 KB | Financial planning document for HOA reserves. Component inventories, funding tables, projections. |
| **CC&Rs** | 1 | 1.2 MB | Covenants, Conditions & Restrictions. Dense legal document governing HOA operations. |
| **Unit Owner Letters** | 1 | 388 KB | Certificate holder notification letter. |
| **Coverage Summary (Quote)** | 1 | 72 KB | Detailed tabular comparison with sections: Property, Earthquake, Building Ordinance, General Liability, Workers Comp, D&O, Crime, Umbrella, Annual Pricing. |

### Critical Document: Coverage Summary Structure

The Coverage Summary is the most critical and most challenging document type. Example structure from Bethany Terrace:

```
BETHANY TERRACE HOMEOWNERS ASSOCIATION
Policy Date 12/13/2025 to 12/13/2026
Company/Agent: Associs/Sara Eanni

                              LIO 2024-2025    LIO 2025-2026    Phily 2025-2026
Property Coverage
  Building Limit              $133,000.00      $133,000.00      $100,000.00
  Community Personal Property $50,000.00       $50,000.00       No Coverage
  Deductible Amount           $5,000.00        $5,000.00        $2,500.00

General Liability
  Occurrence Limit            $1,000,000.00    $1,000,000.00    $1,000,000.00
  Aggregate Limit             $2,000,000.00    $2,000,000.00    $2,000,000.00

Directors & Officers (C.N.A)
  Coverage Limit              $1,000,000.00    $1,000,000.00    $1,000,000.00
  Cyber Coverage              $50,000.00       $50,000.00       $50,000.00

Crime
  Employee Dishonesty         $250,000.00      $250,000.00      $100,000.00
  Social Engineering Limit    $50,000.00       $50,000.00       $10,000.00

Annual Pricing
  Package                     $2,177.00        $2,389.00        $2,076.00
  Directors & Officers        $1,675.00        $1,850.00        $1,556.00
  Total Premium               $3,852.00        $4,239.00        $3,987.00
```

**Extraction challenges:** Multi-column alignment, section headers spanning all columns, currency values, "No Coverage"/"Not Offered"/"See Option"/"Quote upon request" categorical values, nested sub-rows.

---

## 3. VE-RAG-System Current Capabilities

### Composite Score: 6.3/10

| Dimension | Score | Weight | Weighted | Key Finding |
|-----------|-------|--------|----------|-------------|
| Extraction Accuracy | 6/10 | 25% | 1.50 | Docling TableFormer ACCURATE is strong, but table rows fragment across chunks. Only ACORD 25 has template extraction. No insurance NER. |
| Repeatability | 7/10 | 15% | 1.05 | Forms pipeline is deterministic. Standard RAG path depends on LLM (non-deterministic). No content-hash caching. |
| Semantic Understanding | 5/10 | 20% | 1.00 | 18 insurance intent patterns, but no coverage ontology, no negative extraction, no cross-document reasoning beyond tag-based filtering. |
| Chunking Quality | 7/10 | 10% | 0.70 | HybridChunker + ACORD 25 field-group rechunking are solid. Coverage summary tables can still split across chunks. |
| Query Capability | 5/10 | 15% | 0.75 | Simple lookups work well. Cross-document comparison and numerical aggregation are weak. |
| Deployment Model | 9/10 | 10% | 0.90 | Fully air-gapped. All open-source. DGX Spark ready. |
| Total Cost of Ownership | 8/10 | 5% | 0.40 | $0 licensing. Infrastructure = DGX Spark hardware only. |
| **Composite** | | | **6.30** | |

### Architecture Strengths

1. **Forms Pipeline (`ingestkit-forms`)**: Template matching via layout fingerprinting, multi-strategy extraction (PDF widget → OCR → VLM fallback), ACORD 25 field-group rechunking, PII redaction, confidence-based rejection. This is more sophisticated than most commercial alternatives.

2. **Hybrid Search**: Dense + BM25 sparse retrieval via Qdrant, with pre-retrieval access control (tag-based filtering), intent-based boosting, and recency weighting.

3. **Insurance-Aware Auto-Tagging**: The `insurance_agency.yaml` strategy defines 14 document types, 10 coverage line topics, and carrier entity extraction with aliases. Path-based rules auto-tag by client, year, stage, and entity.

4. **Curated Q&A System**: Short-circuits RAG for known questions with 100% deterministic answers.

5. **Air-Gap Architecture**: Every component (Ollama, Qdrant, Docling, Tesseract/PaddleOCR) runs locally with zero cloud dependencies.

### Architecture Weaknesses

1. **No insurance-specific NER**: Policy numbers, coverage limits, deductibles, and premiums are treated as plain text — not as typed entities.

2. **No SQL query routing**: Numerical questions ("total premium", "compare limits") go through the LLM instead of direct SQL aggregation against `forms_data.db`.

3. **No negative extraction**: Cannot determine "what is NOT covered" or identify coverage gaps against a standard HOA program.

4. **Limited form templates**: Only ACORD 25 has field-group definitions. ACORD 24, 27, 28, 80 are not supported.

5. **Table chunking fragmentation**: Coverage summary tables can split across 200-token chunks, destroying column relationships.

6. **3-chunk-per-document limit**: 50+ page D&O and package policies get only 3 chunks in RAG context, insufficient for comprehensive coverage.

---

## 4. Extraction Quality by Document Type

| Document Type | Laptop (SimpleChunker) | Spark (DoclingChunker) | Spark + Forms Pipeline | Key Issues |
|---|---|---|---|---|
| **Coverage Summaries** | 2/10 | 4/10 | N/A (not a form) | Table column alignment destroyed by chunking; no row-preservation logic; no section-aware chunking |
| **ACORD Certificates** | 2/10 | 5/10 | **8/10** | Excellent with forms pipeline; field-group rechunking preserves structure; PII redaction works |
| **D&O Policies** | 4/10 | 6/10 | N/A | Good text extraction but 3-chunk limit on 50+ page docs; no section-aware retrieval |
| **LIO Policies** | 3/10 | 5/10 | N/A | Large files may timeout; chunk overlap insufficient for cross-referencing coverage sections |
| **Package Policies** | 3/10 | 5/10 | N/A | 2.1 MB combined policy; multiple coverage sections compete for 3-chunk limit |
| **Loss Run Reports** | 3/10 | 5/10 | N/A | Tabular claims data loses structure; no date/amount parsing |
| **Reserve Studies** | 4/10 | 6/10 | N/A | Text-heavy sections extract well; financial tables lose structure |
| **CC&Rs** | 5/10 | 7/10 | N/A | Text extraction works; Docling preserves section headings |
| **Unit Owner Letters** | 5/10 | 7/10 | N/A | Mostly text; straightforward extraction |

### Critical Finding: Coverage Summary Gap

The Coverage Summary is the **most important document for daily insurance operations** and scores only **4/10** on Spark. The core issue: Docling's TableFormer correctly detects the table structure, but the HybridChunker then fragments it into 200-token chunks that break the column alignment. A row like:

```
Employee Dishonesty  |  $250,000  |  $250,000  |  $100,000
```

May land in a different chunk than its column headers (`LIO 2024-2025 | LIO 2025-2026 | Phily 2025-2026`), making the values meaningless.

---

## 5. Open Source Competitive Landscape

### Comparison Matrix (1-10, weighted for insurance document use case)

| Solution | Table Extraction | Form Recognition | Repeatability | Insurance Domain | RAG Chunking | Air-Gap | License | Weighted Total |
|---|---|---|---|---|---|---|---|---|
| **Docling** (current) | 8 | 3 | 9 | 2 | 9 | 10 | MIT | **73** |
| **Unstructured** | 7 | 3 | 8 | 2 | 7 | 8 | Apache 2.0 | 63 |
| **Camelot/Tabula** | 7 | 1 | 9 | 1 | 1 | 10 | MIT | 62 |
| **LangChain** | 3 | 2 | 4 | 2 | 5 | 9 | MIT | 59 |
| **LlamaIndex (OSS)** | 4 | 2 | 5 | 2 | 6 | 4 | MIT | 55 |
| **Haystack** | 3 | 2 | 4 | 2 | 6 | 8 | Apache 2.0 | 55 |
| **MinerU** | 9 | 3 | 8 | 2 | 5 | 7 | AGPL-3.0 | 55 |
| **docTR** | 2 | 2 | 8 | 1 | 1 | 9 | Apache 2.0 | 52 |
| **Marker** | 6 | 3 | 6 | 2 | 3 | 7 | GPL-3.0 | 51 |
| **Chandra** | 8 | 6 | 7 | 2 | 3 | 7 | GPL-3.0 | 51 |
| **MegaParse** | 4 | 3 | 5 | 2 | 4 | 3 | Apache 2.0 | 44 |

*Weighting: Table Extraction (2x), Form Recognition (1.5x), Repeatability (1.5x), Chunking (1.5x), Air-Gap (1.5x), all others (1x).*

### Solution Profiles

#### Docling (IBM / LF AI & Data Foundation) — **Current Choice, Recommended to Keep**
- 42,000+ GitHub stars, 1.5M PyPI downloads/month
- TableFormer ML-based table structure recognition with 97.9% accuracy
- HybridChunker preserves semantic coherence, keeping table rows together
- Granite-Docling-258M vision-language model (Apache 2.0) adds end-to-end conversion
- Donated to LF AI & Data Foundation (April 2025) for long-term governance
- **Gap**: No native ACORD form template recognition; treats forms as generic documents

#### Unstructured.io — **Not Recommended (marginal improvement)**
- Element-typed output (Table, NarrativeText, etc.) enables targeted processing
- Overall table score of 0.844 in benchmarks
- **Blocker**: 51 seconds per page processing time; complex table accuracy only 75%

#### Camelot/Tabula — **Recommended as Supplementary Table Extractor**
- Purpose-built for table extraction with Lattice (line-based) and Stream (whitespace-based) modes
- Direct DataFrame output for structured data processing
- MIT license, lightweight, no GPU required
- **Limitation**: Only works with text-based PDFs (not scanned); tables only, no full document processing

#### MinerU (OpenDataLab) — **Watch List (license concern)**
- 1.2B parameter VLM achieving 90.67 on OmniDocBench (best overall benchmark score)
- Outperforms Gemini 2.5 Pro on document parsing benchmarks
- **Blocker**: AGPL-3.0 license (copyleft, may be incompatible with proprietary deployment)

#### Chandra OCR (Datalab) — **Watch List (form checkbox support)**
- Successor to Surya and Marker, built on fine-tuned Qwen-3-VL
- Form support including checkboxes (directly relevant to ACORD forms)
- **Blocker**: GPL-3.0 license; relatively new (late 2025)

#### Key Insight: No OSS Tool Solves Insurance Domain Understanding

All solutions score 2/10 or below on insurance domain understanding. None understand:
- Policy numbers, coverage limits, deductibles as semantic concepts
- The relationship between "Named Insured" and coverage applicability
- That "$1,000,000 Occurrence / $2,000,000 Aggregate" is a coverage structure
- ACORD form field semantics

This gap is best addressed at the RAG query/retrieval layer (which VE-RAG's `forms_query_service.py` already does) rather than at the parser layer.

---

## 6. Commercial / Closed Source Competitive Landscape

### Comparison Matrix

| Solution | Table | Forms | Repeatability | Insurance Domain | Air-Gap | Python SDK | Price/Page | Weighted Score |
|---|---|---|---|---|---|---|---|---|
| **Indico Data** | 8 | 9 | 8 | **10** | YES | 5 | $100K+/yr | 8.0 |
| **Reducto** | 9 | 8 | 7 | 5 | **YES** | 9 | ~$0.01 | 7.7 |
| **Azure AI Doc Intelligence** | 8 | 9 | 8 | 7 | Partial | 8 | $0.008-0.027 | 7.3 |
| **ABBYY FineReader Engine** | 8 | 8 | 9 | 6 | **YES** | 6 | ~$0.01 | 7.1 |
| **Docling** (baseline) | 7 | 4 | 8 | 2 | **YES** | 10 | Free | 6.7 |
| **Unstract** (open source) | 6 | 6 | 7 | 5 | **YES** | 7 | Free | 6.5 |
| **Tungsten (Kofax)** | 7 | 8 | 8 | 7 | YES | 4 | $50K+/yr | 6.3 |
| **Sensible.so** | 7 | 9 | 9 | **9** | NO | 8 | Usage-based | 6.0 |
| **LlamaParse** | 7 | 6 | 6 | 3 | Uncertain | 9 | $0.003 | 5.9 |
| **Mathpix** | 8 | 5 | 8 | 2 | Partial | 7 | $0.01 | 5.4 |
| **AWS Textract** | 7 | 7 | 8 | 5 | NO | 9 | $0.01-0.015 | 5.1 |
| **Google Document AI** | 7 | 7 | 8 | 5 | NO | 8 | $0.01-0.065 | 5.0 |

*Weighted formula: Air-Gap (25%) + Table Extraction (15%) + Insurance Domain (15%) + Repeatability (10%) + Form Recognition (10%) + RAG Chunking (10%) + Integration (10%) + Pricing (5%)*

### Top Recommendations

#### Tier 1: Recommended for Air-Gapped Insurance RAG

**1. Reducto — TOP COMMERCIAL RECOMMENDATION**
- Vision-first hybrid pipeline (computer vision + OCR + VLM + Agentic OCR)
- Raised $75M Series B (Oct 2025); proven in air-gapped government environments (Scale AI case study)
- Fully containerized on-premise deployment with active self-hosted changelog
- RAG-native output designed specifically for LLM consumption
- Estimated $0.008-0.012/page on-premise at volume
- **Risk**: Young company (founded 2023); enterprise pricing opaque; requires GPU

**2. Azure AI Document Intelligence (Disconnected Containers)**
- Most mature cloud-provider on-premise option
- Custom models trainable on ACORD forms with as few as 5 labeled samples
- Disconnected containers for Read, Layout, Invoice, and Custom models
- **Caveat**: 100K pages/month minimum commitment; custom model training requires cloud; initial license provisioning needs internet

#### Tier 2: Strong But With Caveats

**3. ABBYY FineReader Engine SDK**
- Most proven air-gap story (decades of government/defense/insurance deployment)
- 100% deterministic output — critical for repeatability
- No GPU required
- **Caveat**: Traditional OCR (not vision-model based); poor Python integration; no RAG chunking

**4. Indico Data**
- Unmatched insurance domain: 20,000+ insurance terms, 900+ document types, 70+ languages
- Pre-built agents for intake, triage, summarization, validation
- Full bare-metal air-gap deployment
- **Caveat**: $100K+/yr; massive overkill for a RAG extraction use case; platform-oriented, not a simple API

#### Tier 3: Not Recommended

| Solution | Reason for Exclusion |
|---|---|
| AWS Textract | Cloud-only. No air-gap path. |
| Google Document AI | Cloud-only. No air-gap path. |
| Sensible.so | Cloud-only. (Would be ideal otherwise for ACORD forms.) |
| LlamaParse | Air-gap status uncertain. Non-deterministic output. |
| Mathpix | Minimum volume too high. No insurance domain knowledge. |
| Tungsten (Kofax) | Too heavyweight. Poor Python integration. $50K+/yr. |

---

## 7. Air-Gap Compatibility Matrix

| Solution | Air-Gap Ready | Deployment Model | GPU Required | Notes |
|---|---|---|---|---|
| **Docling** | YES | Python library | No (CPU ok) | Currently integrated. Fully local. |
| **Reducto** | YES | Docker containers | Yes | Proven in gov air-gap environments. On-prem changelog shows active dev. |
| **ABBYY FineReader** | YES | Native binary/Docker | No | Decades of on-premise deployment. No phone-home after activation. |
| **Indico Data** | YES | Bare metal/private cloud | Yes | Fully self-contained. SOC 2, HIPAA compliant. |
| **Unstract** | YES | Docker (open source) | Yes | Apache 2.0. Works with local Ollama. |
| **Camelot/Tabula** | YES | Python library | No | Lightweight table extraction. MIT license. |
| **Azure AI Doc Intel** | PARTIAL | Disconnected containers | No | Select models only. 100K pages/month min. Training requires cloud. |
| **LlamaParse** | UNCERTAIN | VPC deployment | Unknown | Enterprise VPC option. True air-gap not confirmed by vendor. |
| **Mathpix** | PARTIAL | On-prem PDF cloud | Unknown | Targets massive scale (tens of millions pages). |
| **AWS Textract** | NO | Cloud-only | N/A | No on-premise option exists. |
| **Google Document AI** | NO | Cloud-only | N/A | GDC theoretically possible but Doc AI not available on it. |
| **Sensible.so** | NO | Cloud-only | N/A | No self-hosted option. |

---

## 8. Insurance-Specific Extraction Analysis

### 8.1 Insurance-Specific Tools Evaluated

| Tool | Strengths | Air-Gap | Relevance |
|------|-----------|---------|-----------|
| **ACORD Transcriber** | 4,700+ versions of 800+ ACORD forms; canonical authority | Cloud SaaS | Reference only |
| **Sensible.so** | SenseML template language; 150+ prebuilt insurance parsers | Cloud API | Design patterns to emulate |
| **SortSpoke** | 5x faster ACORD extraction; handles carrier variations | Cloud SaaS | Not air-gap compatible |
| **Unstract/LLMWhisperer** | Layout-preserving OCR; checkbox/radio capture; 300+ languages | **On-Prem** | Strong candidate for scanned docs |
| **Apryse Template SDK** | Deterministic coordinate-based ACORD extraction | Self-hosted | Template-based; guarantees repeatability |

### 8.2 Table Extraction for Coverage Summaries

Coverage Summaries are the most critical and most challenging document. Recommended extraction pipeline:

**Stage 1: Layout-Preserving Parse (Docling)**
```python
table_options = TableStructureOptions(
    do_cell_matching=True,
    mode="accurate",  # NOT "fast" — accuracy critical for financial data
)
```

**Stage 2: Structured Extraction via Pydantic Schema**
```python
class CoverageLineItem(BaseModel):
    coverage_type: str              # "Property", "General Liability", "D&O"
    sub_type: str | None = None     # "Per Occurrence", "Aggregate"
    values: dict[str, str]          # {"LIO 2024-2025": "$1,000,000", "Phily": "Not Offered"}

class CoverageSection(BaseModel):
    name: str                       # "Property Coverage", "General Liability"
    line_items: list[CoverageLineItem]
    premiums: dict[str, str] | None = None

class CoverageSummary(BaseModel):
    document_title: str
    carriers: list[str]             # Column headers
    sections: list[CoverageSection]
```

**Stage 3: Currency Validation & Normalization**
```python
COVERAGE_STATUS_MAP = {
    "no coverage": "NO_COVERAGE",
    "not offered": "NOT_OFFERED",
    "see option": "SEE_OPTION",
    "quote upon request": "QUOTE_UPON_REQUEST",
    "included": "INCLUDED",
    "n/a": "NOT_APPLICABLE",
}
```

### 8.3 Template-Based Extraction for Repeatability

**Approaches to guarantee deterministic output:**

| Approach | Determinism | Implemented? | Notes |
|----------|------------|-------------|-------|
| Layout fingerprint + cache | High | Partial (forms pipeline) | Extend to cache extraction results by `(file_hash, template_version)` |
| Coordinate-based zone extraction | 100% | Partial (PDF widgets) | Define extraction zones by page coordinates for ACORD forms |
| Template match first, LLM fallback | High for matched | Yes | Current `FormsProcessingService` architecture |
| Content hash caching | 100% for cached | No | Add content-aware caching to avoid re-extraction |

### 8.4 Recommended Chunking by Document Type

| Document Type | Chunking Strategy | Rationale |
|---|---|---|
| **Coverage Summaries** | Section-based with header injection | Each coverage section (Property, GL, D&O, Crime) becomes one chunk. Column headers injected into every chunk. |
| **ACORD Certificates** | Field-group (already implemented) | ACORD_25_GROUPS maps fields to logical groups. Extend to ACORD 24, 27, 80. |
| **D&O/LIO Policies** | Section-based with declarations priority | Declarations page = 1 chunk. Each endorsement = 1 chunk. Exclusions = 1 chunk per section. |
| **Loss Run Reports** | Row-based | Each claim = 1 chunk with full metadata (policy #, date, status, amounts). |
| **Reserve Studies** | Component + financial table | Component inventory by category. Entire funding table as single chunk. |
| **CC&Rs** | Standard semantic (HybridChunker) | Text-heavy; current approach works well. |

### 8.5 Multi-Vector Retriever Pattern

For table-heavy documents, implement the multi-vector retriever pattern:

1. **Summary vectors**: LLM-generated natural language summaries of each table section (optimized for semantic search)
2. **Raw content store**: Actual table data (markdown/structured text) stored as retrieval payload
3. **Retrieval flow**: Query matches summary → raw table content passed to LLM for answer synthesis

This is critical because embedding `"$1,000,000 | $2,000,000 | No Coverage"` produces poor vectors, but embedding `"The General Liability coverage has a per-occurrence limit of one million dollars"` produces excellent vectors.

### 8.6 Hybrid Architecture: Field-Level Extraction + RAG

| Query Type | Best Approach | Example |
|---|---|---|
| Factual/numeric lookup | SQL against `forms_data.db` | "What is the GL limit?" |
| Comparative/aggregation | SQL with GROUP BY | "Total premium across all lines" |
| Interpretive/policy text | RAG vector search | "Does the D&O policy cover cyber?" |
| Coverage gap analysis | Checklist comparison | "What coverages are missing?" |
| Cross-document synthesis | Multi-hop RAG | "Compare D&O across all clients" |

```
Document Upload → Template Match (ingestkit-forms)
                       |
              ┌────────┴────────┐
              ↓                 ↓
          SQL DB            Vector Store
       (exact fields)     (semantic chunks)
              ↓                 ↓
         ┌────────────────────────┐
         │    Query Router        │
         │  (intent detection)    │
         │                        │
         │  Factual → SQL first   │
         │  Interpretive → RAG    │
         │  Comparative → SQL+RAG │
         └────────────────────────┘
```

---

## 9. Evaluation Framework

### 9.1 Evaluation Dimensions

| Dimension | Sub-Dimensions | Weight |
|---|---|---|
| **Extraction Accuracy** | Table structure (25%), Form fields (25%), Entity recognition (20%), Currency parsing (15%), Cross-reference resolution (15%) | 25% |
| **Repeatability** | Deterministic output (40%), Template consistency (30%), Temperature control (15%), Version stability (15%) | 15% |
| **Semantic Understanding** | Insurance domain (30%), Cross-document reasoning (30%), Temporal reasoning (20%), Negative extraction (20%) | 20% |
| **Chunking Quality** | Table preservation (35%), Context preservation (30%), Granularity balance (20%), Cross-chunk continuity (15%) | 10% |
| **Query Capability** | Simple lookup (15%), Intra-doc comparison (20%), Cross-doc synthesis (30%), Aggregation (20%), Gap analysis (15%) | 15% |
| **Deployment Model** | Air-gap (40%), Hardware requirements (25%), Licensing (20%), Operational simplicity (15%) | 10% |
| **Total Cost of Ownership** | Infrastructure (30%), License/API (25%), Customization effort (25%), Maintenance (20%) | 5% |

### 9.2 Test Scenarios

#### Tier 1: Simple Lookup (Baseline)

| # | Query | Type | Success Criteria |
|---|-------|------|------------------|
| Q1 | "What is Bethany Terrace's D&O coverage limit?" | Single-document, single-field | Exact dollar amount with source citation |
| Q2 | "Who is the insurance producer/agent for Bethany Terrace?" | Form field extraction | Correct producer name, contact info |

#### Tier 2: Single-Document Comparison

| # | Query | Type | Success Criteria |
|---|-------|------|------------------|
| Q3 | "Compare the total premium between LIO 2024-2025 and LIO 2025-2026 for Bethany Terrace" | Multi-column table comparison | Both premiums, delta calculated, YoY change |
| Q4 | "List all coverage types and per-occurrence limits from Bethany Terrace's current coverage summary" | Full table extraction | All coverage lines with correct limits, no hallucinated rows |

#### Tier 3: Cross-Document Synthesis

| # | Query | Type | Success Criteria |
|---|-------|------|------------------|
| Q5 | "Which of the three clients has the highest D&O coverage limit?" | Cross-document comparison | Correct ranking with amounts and citations |
| Q6 | "Compare Crime coverage limits across all carrier quotes for Bethany Terrace" | Multi-column extraction | All carriers listed with Crime limits |

#### Tier 4: Negative / Gap Analysis

| # | Query | Type | Success Criteria |
|---|-------|------|------------------|
| Q7 | "What coverages does Bethany Terrace NOT have compared to a standard HOA program?" | Negative extraction + domain knowledge | Lists missing coverages with reasoning |
| Q8 | "Are there gaps in the Additional Insured endorsements across Bethany Terrace's policies?" | Cross-document endorsement analysis | Identifies which policies lack AI endorsements |

#### Tier 5: Complex Reasoning and Aggregation

| # | Query | Type | Success Criteria |
|---|-------|------|------------------|
| Q9 | "What is the total annual insurance cost for Bethany Terrace across all lines, and how does it compare to prior year?" | Multi-document aggregation | Accurate sum, percentage change |
| Q10 | "Based on the loss run history, which coverage lines have highest claims frequency, and do current limits appear adequate?" | Multi-document analysis + domain expertise | Claims frequency analysis, adequacy assessment |

### 9.3 VE-RAG Expected Performance by Test Scenario

| Scenario | Current Score | After Phase 1 | After Phase 2 | Best-in-Class |
|---|---|---|---|---|
| Q1: Simple lookup | 8/10 | 9/10 | 9/10 | 10/10 |
| Q2: Form field | 7/10 (with forms) | 8/10 | 9/10 | 9/10 |
| Q3: Single-doc comparison | 4/10 | 6/10 | 8/10 | 9/10 |
| Q4: Full table extraction | 4/10 | 5/10 | 8/10 | 9/10 |
| Q5: Cross-doc comparison | 3/10 | 5/10 | 7/10 | 8/10 |
| Q6: Multi-column extraction | 4/10 | 6/10 | 8/10 | 9/10 |
| Q7: Negative extraction | 2/10 | 6/10 | 7/10 | 8/10 |
| Q8: Endorsement gap analysis | 2/10 | 3/10 | 6/10 | 8/10 |
| Q9: Total cost aggregation | 3/10 | 7/10 | 8/10 | 9/10 |
| Q10: Claims + adequacy | 2/10 | 3/10 | 5/10 | 8/10 |

---

## 10. Gap Priority Matrix

| Rank | Gap | VE-RAG Now | Best-in-Class | Impact | Difficulty | Air-Gap OK |
|------|-----|-----------|---------------|--------|------------|------------|
| **1** | SQL query routing for numerical questions | No SQL queries at retrieval | Route aggregation to `forms_data.db` | 9/10 | 3-4 days | Yes |
| **2** | Coverage gap / negative extraction | Not supported | Coverage checklist comparison system | 8/10 | 2-3 days | Yes |
| **3** | Temperature=0 for extraction queries | Configurable but not enforced | Query-type-aware temperature routing | 6/10 | 1 day | Yes |
| **4** | Currency/number normalization | LLM handles formatting | Post-extraction regex normalization | 6/10 | 2-3 days | Yes |
| **5** | Additional ACORD form templates | Only ACORD 25 | All major ACORD certificates | 8/10 | 2-3 weeks | Yes |
| **6** | Insurance entity NER | LLM-only entity recognition | SpaCy NER with insurance entities | 7/10 | 1-2 weeks | Yes |
| **7** | Multi-hop retrieval agent | Single retrieval pass | Iterative retrieve-reason-retrieve | 7/10 | 2-3 weeks | Yes |
| **8** | Cross-reference resolution | Not supported | Section/reference linking | 6/10 | 2-3 weeks | Yes |
| **9** | Hierarchical chunking | Flat chunk list | Nested hierarchy with parent context | 6/10 | 1-2 weeks | Yes |
| **10** | Insurance domain LLM fine-tuning | Generic qwen3:8b | LoRA adapter on insurance corpus | 8/10 | 4-6 weeks | Yes (pre-deploy) |
| **11** | Policy entity knowledge graph | None | Lightweight graph DB | 7/10 | 4-6 weeks | Yes |
| **12** | Table-aware chunking with column metadata | Docling + HybridChunker | Table chunks with column-name index | 5/10 | 1-2 weeks | Yes |

**All 12 gaps are air-gap compatible.** No external services required.

---

## 11. Recommended Roadmap

### Phase 1: Quick Wins (1-2 Weeks)
**Theme:** Leverage existing infrastructure for immediate quality improvements.
**Projected Score:** 6.3 → ~7.2

| Task | Effort | Impact | Details |
|------|--------|--------|---------|
| **SQL query routing** | 3-4 days | +1.5 composite | Extend `FormsQueryService` to execute SQL aggregation against `forms_data.db` for numerical intents. Add intent patterns: "total premium", "sum of all", "how much". |
| **Coverage checklist** | 2-3 days | +1.0 composite | Create YAML defining standard HOA coverages. When gap analysis detected, compare extracted coverages against checklist. Return gaps as synthetic SearchResults. |
| **Temperature=0 enforcement** | 1 day | +0.3 composite | Add `query_type` classifier. Set temperature=0 for factual queries. Modify `_run_rag_pipeline()` to override per query type. |
| **Currency normalization** | 2-3 days | +0.3 composite | Post-extraction regex normalizer: `$1M` → `$1,000,000`. Handle `N/A`/`Included`/`Waived` categorical values. |

### Phase 2: Medium Term (1-2 Months)
**Theme:** Expand form template coverage and add multi-hop reasoning.
**Projected Score:** ~7.2 → ~8.0

| Task | Effort | Impact | Details |
|------|--------|--------|---------|
| **ACORD form templates** | 2-3 weeks | +0.5 composite | Register ACORD 24, 27, 28, 80. Create layout fingerprints and field group definitions. Test data already exists in `test_data/`. |
| **Insurance NER** | 1-2 weeks | +0.4 composite | SpaCy NER model with entity types: CARRIER, INSURED, POLICY_NUMBER, NAIC_CODE, COVERAGE_TYPE, DOLLAR_AMOUNT, DATE_RANGE, AGENT. Run at ingestion, store in Qdrant payload. |
| **Multi-hop retrieval** | 2-3 weeks | +0.3 composite | Iterative retrieve-reason-retrieve loop (max 3 hops). Use existing `query_expansion` as reformulation engine. |
| **LLM tool use (calculator)** | 1 week | +0.2 composite | Add calculator tool to LLM prompt. Parse tool calls and execute locally. Deterministic arithmetic. |
| **Hierarchical chunking** | 1-2 weeks | +0.2 composite | Two-level chunks: Level 1 = section (2048 tokens), Level 2 = paragraph (512 tokens). Include parent section when paragraph matches. |

### Phase 3: Long Term (3-6 Months)
**Theme:** Domain specialization and knowledge graph integration.
**Projected Score:** ~8.0 → ~8.5+

| Task | Effort | Impact | Details |
|------|--------|--------|---------|
| **Coverage Summary structured extractor** | 3-4 weeks | High | Pydantic schema extraction → dedicated SQL table at ingest. Multi-vector retriever pattern (summary vectors + raw table payloads). |
| **Insurance domain LoRA** | 4-6 weeks | High | Fine-tune qwen3:8b on insurance corpus. Focus: coverage terminology, policy structure, exclusion interpretation. Validate on 10 test scenarios. |
| **Policy entity knowledge graph** | 4-6 weeks | Medium | Lightweight graph: Client → Policy → Coverage → Carrier. Populate at ingestion using NER + rule-based extraction. Enable graph queries. |
| **Cross-reference resolution** | 2-3 weeks | Medium | Detect patterns ("See Attached Schedule", "Per Underlying Policy"). Create links in knowledge graph. Fetch linked content at retrieval. |
| **Automated quality regression** | 2-3 weeks | Medium | Golden dataset of 10 test scenarios. Automated regression on model/template/parameter changes. Alert on >5% score degradation. |

### Optional: Commercial Enhancement

If budget allows, add **Reducto on-premise** (~$0.01/page) as a supplementary extractor for complex tables. Route document types by complexity:

```
Document Upload → Document Classifier
                       |
    ┌──────────────────┼──────────────────┐
    ↓                  ↓                  ↓
  Docling           Reducto         ingestkit-forms
  (text docs)    (complex tables)    (ACORD forms)
  CC&Rs, policies  Coverage summaries  Certificates
  Loss runs        Reserve studies
```

---

## 12. Pricing Analysis

| Solution | Per-Page (Cloud) | Per-Page (On-Prem) | Min Commitment | Annual @ 50K pages/mo | Air-Gap |
|---|---|---|---|---|---|
| **Docling** (current) | Free | Free | None | $0 | YES |
| **Unstract** | Free | Free (infra only) | None | $0 + GPU | YES |
| **LlamaParse** | $0.003 | Custom | Custom | $1,800+ | Uncertain |
| **ABBYY Engine SDK** | N/A | ~$0.005-0.020 | Perpetual license | $3,000-12,000 + license | YES |
| **Reducto** | $0.015 | ~$0.008-0.012 | Custom | $4,800-7,200 (est.) | YES |
| **Azure AI Doc Intel** | $0.001-0.065 | $0.008-0.027 | 100K pages/mo | $4,800-16,200 | Partial |
| **AWS Textract** | $0.010-0.015 | N/A | None | $6,000-9,000 | NO |
| **Google Doc AI** | $0.010-0.065 | N/A | None | $6,000-39,000 | NO |
| **Tungsten (Kofax)** | N/A | Enterprise | $50K-200K+ | $50,000+ | YES |
| **Indico Data** | N/A | Private offer | Enterprise | $100,000+ (est.) | YES |

---

## 13. Conclusions

### What VE-RAG Does Well

1. **Air-gap architecture** is a genuine competitive advantage. Only 4 commercial solutions can truly match it (Reducto, ABBYY, Indico, Tungsten), and none at VE-RAG's price point ($0).
2. **Forms pipeline** with template matching, field-group rechunking, and structured SQL storage is architecturally superior to most generic RAG systems.
3. **Insurance-aware auto-tagging and intent detection** provide domain awareness that no open-source tool offers.
4. **Hybrid search** (dense + BM25 + access control + intent boosting + recency weighting) is a sophisticated retrieval stack.

### Where VE-RAG Falls Short

1. **Coverage Summary extraction** (the most important document type) scores only 4/10 due to table fragmentation in chunking.
2. **No structured query path** for numerical questions — everything goes through the LLM.
3. **No negative extraction** — cannot answer "what's missing?"
4. **Limited form templates** — only ACORD 25 is supported.

### Strategic Recommendation

The fastest path to production-quality insurance document extraction is:

1. **Phase 1 quick wins** (1-2 weeks, $0 cost) close nearly half the gap to best-in-class.
2. **Phase 2 template expansion** (1-2 months) broadens coverage to all major HOA insurance document types.
3. **Optional Reducto on-premise** provides the single biggest extraction quality jump for complex tables at ~$0.01/page.

The air-gap requirement and $0 licensing cost are VE-RAG's strongest competitive advantages. No commercial solution matches both simultaneously.

---

## 14. Sources

### Benchmarks & Comparisons
- [PDF Data Extraction Benchmark 2025 — Procycons](https://procycons.com/en/blogs/pdf-data-extraction-benchmark/)
- [PDF Table Extraction Showdown — BoringBot](https://boringbot.substack.com/p/pdf-table-extraction-showdown-docling)
- [Docling vs LlamaParse vs Unstructured vs Reducto — Reducto](https://llms.reducto.ai/document-parser-comparison)
- [Best LLM-Ready Document Parsers 2025 — Reducto](https://llms.reducto.ai/best-llm-ready-document-parsers-2025)
- [State of PDF Parsing — Applied AI](https://www.applied-ai.com/briefings/pdf-parsing-benchmark/)
- [12 Open-Source PDF Parsing Tools Comparison](https://liduos.com/en/ai-develope-tools-series-2-open-source-doucment-parsing.html)
- [Unstructured Parsing Quality Benchmarks](https://unstructured.io/blog/unstructured-leads-in-document-parsing-quality-benchmarks-tell-the-full-story)

### Open Source Tools
- [Docling — GitHub](https://github.com/docling-project/docling) (MIT, 42K+ stars)
- [IBM Granite-Docling Announcement](https://www.ibm.com/new/announcements/granite-docling-end-to-end-document-conversion)
- [Docling AAAI 2025 Paper](https://research.ibm.com/publications/docling-an-efficient-open-source-toolkit-for-ai-driven-document-conversion)
- [Unstructured — GitHub](https://github.com/Unstructured-IO/unstructured) (Apache 2.0)
- [Marker — GitHub](https://github.com/datalab-to/marker) (GPL-3.0)
- [MinerU — GitHub](https://github.com/opendatalab/MinerU) (AGPL-3.0)
- [Chandra OCR — GitHub](https://github.com/datalab-to/chandra) (GPL-3.0)
- [docTR — GitHub](https://github.com/mindee/doctr) (Apache 2.0)
- [MegaParse — GitHub](https://github.com/QuivrHQ/MegaParse) (Apache 2.0)
- [Camelot — GitHub](https://github.com/atlanhq/camelot) (MIT)
- [pdfplumber — GitHub](https://github.com/jsvine/pdfplumber) (MIT)

### Commercial Solutions
- [Reducto — Series B Announcement](https://www.prnewswire.com/news-releases/reducto-raises-75m-series-b-to-define-the-future-of-ai-document-intelligence-302581462.html)
- [Reducto — On-Premise Changelog](https://docs.reducto.ai/onprem/changelog)
- [Azure AI Document Intelligence — Disconnected Containers](https://learn.microsoft.com/en-us/azure/ai-services/document-intelligence/containers/disconnected)
- [ABBYY FineReader Engine SDK](https://www.simpleocr.com/product/abbyy-finereader-ocr-sdk-on-premise/)
- [Indico Data — Agentic Platform Launch](https://www.prnewswire.com/news-releases/indico-data-launches-industrys-first-agentic-decisioning-platform-purpose-built-for-insurance-302466802.html)
- [Sensible.so — Insurance Solutions](https://www.sensible.so/solutions/insurance)
- [LlamaParse V2 — LlamaIndex](https://www.llamaindex.ai/blog/introducing-llamaparse-v2-simpler-better-cheaper)

### Insurance-Specific
- [ACORD Transcriber](https://www.acordsolutions.com/solutions/transcriber)
- [SortSpoke — ACORD Forms](https://sortspoke.com/solutions/document-types/acord-forms)
- [Unstract/LLMWhisperer](https://unstract.com/llmwhisperer/)
- [Apryse — ACORD Form Extraction](https://apryse.com/blog/acord-form-extraction-dot-net)
- [Best OCR for Insurance 2025 — Unstract](https://unstract.com/blog/best-ocr-for-insurance-document-processing-automation/)
- [Docsumo — Insurance Data Extraction](https://www.docsumo.com/blogs/data-extraction/insurance-industry)

### RAG Architecture
- [LangChain — Multi-Vector Retriever for Tables](https://blog.langchain.com/semi-structured-multi-modal-rag/)
- [LangChain — Benchmarking RAG on Tables](https://blog.langchain.com/benchmarking-rag-on-tables/)
- [HybridRAG — Knowledge Graphs + Vector Retrieval](https://arxiv.org/abs/2408.04948)
- [One-Shot Template Matching for Document Capture](https://arxiv.org/pdf/1910.10037)
- [Best Chunking Strategies for RAG 2025 — Firecrawl](https://www.firecrawl.dev/blog/best-chunking-strategies-rag-2025)
- [Chunking for RAG Best Practices — Unstructured](https://unstructured.io/blog/chunking-for-rag-best-practices)
