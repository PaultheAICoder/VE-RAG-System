# Competitor Analysis - Enterprise Knowledge RAG (Air-Gapped)

**Analysis date:** February 10, 2026  
**Strategy window:** March 16, 2026 to September 16, 2026  
**Scope:** United States (starting West Coast / Portland, OR), SMB + mid-market + public sector (maybe), deployment focus on fully on-prem and private VPC  

## Inputs Confirmed

- Primary goals: positioning/messaging, feature gaps, pricing/packaging, GTM
- Competitor types: direct air-gapped platforms, self-hostable general RAG platforms, open-source stacks, plus adjacent cloud-first threats
- Time horizon: next 6 months
- Pricing posture: cost leader
- Pricing model preference: hybrid + flat annual
- Minimum margin target: 35%
- Sales motion: mixed
- Key differentiators: air-gapped security, lower total cost, customer service
- Primary buyer role: operations leader
- Pain points: document search time, onboarding/training speed, process consistency, turnover knowledge loss
- Product maturity: MVP; beta starts March 16, 2026
- Success metric: 6 paid pilots by September 16, 2026
- Target pricing: $850/month per engagement, 6-month pilot, 50% conversion target
- Max CAC per paid pilot: $1,500
- Budget through Sep 16, 2026: <$10k
- Ranking priority: SMB affordability first, then on-prem/VPC fit
- SMB affordability threshold: under $1k/month
- Competitors above threshold: include and mark as high SMB cost

## Project Baseline (From This Repo)

Current product baseline aligns with true air-gapped architecture:
- Local model inference via Ollama
- Local vector storage via Qdrant
- FastAPI backend + React frontend
- Enterprise controls including role-based access and tag-based retrieval filtering

This gives a real differentiation advantage versus cloud-only RAG offerings.

## Deliverable 1: Ranked 15-Competitor Shortlist

Ranking logic:
1. SMB affordability first (<$1k/month)
2. On-prem/private VPC deployment fit
3. Active vendor presence with 2025-2026 evidence

| Rank | Competitor | Type | Deployment Fit (On-Prem/VPC) | SMB Cost Signal | Why It Matters | Evidence |
|---|---|---|---|---|---|---|
| 1 | OpenSearch | Open-source stack | Strong on-prem/self-managed | SMB-friendly | Free Apache-2.0, mature self-host path | https://opensearch.org/downloads/ |
| 2 | Qdrant | Direct/self-hosted RAG infra | Strong (hybrid/private/air-gapped) | SMB-friendly start | Explicit private cloud + air-gapped support messaging | https://qdrant.tech/pricing/ |
| 3 | Weaviate | Self-hosted general RAG infra | Strong (self-managed + BYOC) | SMB-friendly start | Low entry plus customer-managed deployment | https://weaviate.io/pricing , https://docs.weaviate.io/deploy/aws |
| 4 | Haystack (deepset OSS) | Open-source stack | Strong (self-host) | SMB-friendly | Apache-2.0 RAG framework, active ecosystem | https://github.com/deepset-ai/haystack |
| 5 | LangChain | Open-source stack | Strong (self-host) | SMB-friendly | MIT framework, broad ecosystem and integrations | https://github.com/langchain-ai/langchain |
| 6 | LlamaIndex | Open-source stack | Strong (self-host) | SMB-friendly | MIT framework focused on data-connected RAG/agents | https://github.com/run-llama/llama_index |
| 7 | deepset AI Platform | Direct | Medium-strong (cloud + custom deployment) | Mixed (free dev, enterprise custom) | Explicit RAG platform positioning | https://www.deepset.ai/pricing |
| 8 | Elastic | Direct | Strong self-managed/on-prem | High SMB cost risk | Enterprise search/security trust and footprint | https://www.elastic.co/subscriptions |
| 9 | AWS Kendra | Adjacent cloud-first threat | Weak on-prem, strong AWS | Mixed usage-based | Cloud-native option inside existing AWS estates | https://aws.amazon.com/kendra/pricing/ |
| 10 | Azure AI Search | Adjacent cloud-first threat | Weak on-prem, strong Azure | Mixed usage-based | Strong Microsoft ecosystem pull | https://azure.microsoft.com/en-us/pricing/details/search/ |
| 11 | Vertex AI Search | Adjacent cloud-first threat | Weak on-prem, strong GCP | Mixed usage-based | Fast adoption path in Google Cloud orgs | https://cloud.google.com/generative-ai-app-builder/pricing |
| 12 | SearchBlox | Direct | Strong on-prem/cloud | High SMB cost | Explicit enterprise RAG + private deployment story | https://www.searchblox.com/pricing |
| 13 | IBM watsonx (suite) | Direct/adjacent | Strong (on-prem + cloud options) | Likely high SMB cost | Enterprise trust channel + broad platform reach | https://www.ibm.com/products/watsonx-assistant/pricing |
| 14 | Sinequa | Direct | Strong (on-prem/private cloud/SaaS) | High SMB cost | Regulated-enterprise knowledge/search positioning | https://www.sinequa.com/product/ |
| 15 | Mindbreeze InSpire | Direct | Strong (on-prem/cloud-native) | High SMB cost | Enterprise KM/search with published annual pricing | https://www.mindbreeze.com/pricing |

### Concise Notes By Competitor

- OpenSearch: strongest low-cost baseline alternative for technical teams.
- Qdrant: very relevant because it is both self-hostable and explicitly air-gap-compatible.
- Weaviate: credible low-entry infrastructure option with managed and self-managed paths.
- Haystack/LangChain/LlamaIndex: DIY stack pressure, especially for engineering-led SMB buyers.
- deepset: more packaged than OSS, but price/deployment often sales-led.
- Elastic: high capability, but likely price/complexity mismatch for SMB budget-first buyers.
- AWS/Azure/Google: adjacent cloud threats where “good enough + existing procurement” wins.
- SearchBlox/Sinequa/Mindbreeze/IBM: direct enterprise competitors, generally higher SMB cost.

## Deliverable 2: Feature/Pricing Matrix

| Vendor | On-Prem | Private VPC/BYOC | Air-Gapped Story | Pricing Transparency | SMB < $1k/mo Friendly? | Competitive Risk to 6-Pilot Goal |
|---|---|---|---|---|---|---|
| OpenSearch | Yes | Yes | Yes (self-managed) | High | Yes | Medium |
| Qdrant | Yes | Yes | Yes (explicit) | Medium-High | Yes (starts free) | High |
| Weaviate | Yes | Yes | Yes via self-managed | Medium | Yes (entry-level) | High |
| Haystack | Yes | Yes | Yes | High (OSS) | Yes | Medium |
| LangChain | Yes | Yes | Yes | High (OSS) | Yes | Medium |
| LlamaIndex | Yes | Yes | Yes | High (OSS) | Yes | Medium |
| deepset | Custom | Yes (custom) | Possible in custom setups | Medium | Mixed | Medium-High |
| Elastic | Yes | Yes | Yes | Medium | Usually no | Medium |
| SearchBlox | Yes | Yes | Yes | High | No (high SMB cost) | Medium |
| IBM watsonx | Yes | Yes | Possible | Medium | Usually no | Low-Medium |
| Sinequa | Yes | Yes | Strong enterprise security posture | Low (custom) | No | Low-Medium |
| Mindbreeze | Yes | Yes | Yes | High | No | Low-Medium |
| AWS Kendra | No | AWS only | No | High | Sometimes | Medium |
| Azure AI Search | No | Azure only | No | Medium | Sometimes | Medium |
| Vertex AI Search | No | GCP only | No | High | Sometimes | Medium |

## Pricing Detail Addendum

Concrete public pricing values, monthly normalization, and confidence flags are in:

- `docs/competitor-pricing-snapshot-2026-02-10.md`

## Strategic Readout (What This Means For Your GTM)

- Positioning should lead with: **air-gapped security + predictable low cost + strong customer service**.
- Your biggest near-term pressure is from:
  - low-cost self-host stacks (OpenSearch/Qdrant/Weaviate + OSS orchestration), and
  - cloud bundle inertia (AWS/Azure/GCP) where buyers already have cloud commitments.
- Premium enterprise suites are important for long-term differentiation but are less likely to win on price in your target SMB affordability band.

## Recommended Next Deliverables

1. Numeric scored matrix (0-5 per criterion) for all 15 competitors.
2. Top-5 battlecards (objection handling, win themes, landmines).
3. Pilot GTM playbook aligned to 6 paid pilots by Sep 16, 2026.
