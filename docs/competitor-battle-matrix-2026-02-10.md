# Competitor Battle Matrix (Scored 0-5)

**Date:** February 10, 2026  
**Companion file:** `docs/competitor-analysis-2026-02-10.md`  
**Purpose:** Numeric competitor scoring for pricing/positioning decisions over the next 6 months.

## Scoring Model

Scoring scale (per criterion):
- `5` = strong advantage for your target segment
- `3` = neutral/mixed
- `1` = weak fit
- `0` = no fit

Weighted criteria (aligned to your direction):
- `SMB Affordability` (weight 40%)
- `Deployment Fit (On-Prem/Private VPC)` (weight 30%)
- `Air-Gapped Readiness` (weight 20%)
- `Ops Simplicity for Small Teams` (weight 10%)

Weighted score formula:

`Total = (Affordability*0.40) + (Deployment*0.30) + (Airgap*0.20) + (Ops*0.10)`  

Max total score = `5.0`

## Scored Matrix

| Competitor | Type | Affordability (40%) | Deployment (30%) | Air-Gap (20%) | Ops Simplicity (10%) | Weighted Total | SMB Cost Flag |
|---|---|---:|---:|---:|---:|---:|---|
| OpenSearch | OSS stack | 5 | 5 | 5 | 2 | **4.7** | SMB-friendly |
| Qdrant | Direct/self-host infra | 5 | 5 | 5 | 3 | **4.8** | SMB-friendly |
| Weaviate | Self-hosted RAG infra | 4 | 5 | 4 | 3 | **4.1** | SMB-friendly |
| Haystack (deepset OSS) | OSS stack | 5 | 5 | 5 | 2 | **4.7** | SMB-friendly |
| LangChain | OSS stack | 5 | 5 | 5 | 2 | **4.7** | SMB-friendly |
| LlamaIndex | OSS stack | 5 | 5 | 5 | 2 | **4.7** | SMB-friendly |
| deepset AI Platform | Direct | 2 | 4 | 3 | 4 | **2.9** | Mixed / likely above SMB target |
| Elastic | Direct | 1 | 5 | 5 | 2 | **2.9** | High SMB cost |
| AWS Kendra | Cloud-first adjacent | 3 | 1 | 0 | 4 | **1.9** | Mixed |
| Azure AI Search | Cloud-first adjacent | 3 | 1 | 0 | 4 | **1.9** | Mixed |
| Vertex AI Search | Cloud-first adjacent | 3 | 1 | 0 | 4 | **1.9** | Mixed |
| SearchBlox | Direct | 1 | 5 | 5 | 3 | **3.0** | High SMB cost |
| IBM watsonx suite | Direct/adjacent | 1 | 4 | 3 | 3 | **2.3** | High SMB cost |
| Sinequa | Direct | 1 | 5 | 4 | 2 | **2.7** | High SMB cost |
| Mindbreeze InSpire | Direct | 1 | 5 | 4 | 3 | **2.8** | High SMB cost |

## Ranked By Weighted Total

1. Qdrant - 4.8  
2. OpenSearch - 4.7  
3. Haystack - 4.7  
4. LangChain - 4.7  
5. LlamaIndex - 4.7  
6. Weaviate - 4.1  
7. SearchBlox - 3.0  
8. deepset AI Platform - 2.9  
9. Elastic - 2.9  
10. Mindbreeze InSpire - 2.8  
11. Sinequa - 2.7  
12. IBM watsonx suite - 2.3  
13. AWS Kendra - 1.9  
14. Azure AI Search - 1.9  
15. Vertex AI Search - 1.9  

## How To Use This Matrix

- Near-term battle focus (highest risk in your segment):
  - Qdrant
  - OpenSearch
  - Weaviate
  - OSS orchestration stacks (Haystack/LangChain/LlamaIndex)
- Keep enterprise suite competitors in monitoring, but treat them as lower price-pressure risks in SMB.
- Treat cloud-first vendors as adjacent threats when buyers relax air-gap constraints.

## Notes

- Scores are decision-support estimates based on publicly available deployment/pricing signals as of February 10, 2026.
- Where vendor pricing is custom-only, affordability scores were penalized for SMB predictability risk.
- This matrix is intentionally weighted to your current strategy and can be reweighted by segment (SMB vs public sector) later.
