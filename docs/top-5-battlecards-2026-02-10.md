# Top-5 Battlecards (SMB + On-Prem/VPC)

**Date:** February 10, 2026  
**Context:** Supports `docs/competitor-analysis-2026-02-10.md` and `docs/competitor-battle-matrix-2026-02-10.md`  
**Goal:** Help win 6 paid pilots by September 16, 2026.

## 1) Qdrant-Centered DIY Stack

### Why Buyers Consider It
- Strong air-gapped and private deployment story
- Low starting cost
- High performance vector retrieval

### Typical Objection
- "We can build this ourselves with Qdrant and keep costs low."

### Win Themes
- Faster time-to-value than DIY integration
- Lower operational risk for non-specialist teams
- Service layer, governance, and support reduce hidden engineering cost

### Landmines To Avoid
- Do not attack Qdrant quality; your own system depends on similar infrastructure patterns
- Do not position "cheap" alone; position "predictable total cost with support"

### Proof You Need In Pilots
- Time-to-first-answer benchmark
- Admin/operator hours saved per week
- Retrieval quality and permission-filter correctness under real documents

## 2) OpenSearch DIY Stack

### Why Buyers Consider It
- Open-source and broadly known
- No license lock-in
- Fits existing self-host ops patterns

### Typical Objection
- "OpenSearch is free, so why pay for a platform?"

### Win Themes
- "Free software" is not free operations
- Pre-integrated document processing, retrieval workflow, and user experience
- Faster launch for operations teams without a full data/ML platform team

### Landmines To Avoid
- Do not oversell that OpenSearch cannot do RAG; it can, with effort
- Avoid vague claims; quantify setup and maintenance overhead

### Proof You Need In Pilots
- End-to-end deployment timeline vs DIY estimate
- Ongoing maintenance effort comparison
- User adoption/usage metrics from pilot teams

## 3) Weaviate-Centered Stack

### Why Buyers Consider It
- Good self-host and managed options
- Modern vector-native positioning
- Developer-friendly ecosystem

### Typical Objection
- "Weaviate gives us flexibility and we can grow into it."

### Win Themes
- Your offer is tuned for operations leaders, not just developers
- Better cost predictability for SMB engagement model
- Service-backed rollout and onboarding support

### Landmines To Avoid
- Avoid "feature checklist wars" on low-level vector capabilities
- Do not ignore procurement concern about total ownership burden

### Proof You Need In Pilots
- Time from contract to production pilot start
- Admin training time
- Ticket volume / support intervention trend

## 4) LangChain DIY Build

### Why Buyers Consider It
- Massive ecosystem and mindshare
- Highly flexible orchestration patterns
- Minimal upfront license cost

### Typical Objection
- "Our engineer can prototype this in a week with LangChain."

### Win Themes
- Prototype speed is not production reliability
- Governance, access control, and operations support are critical in enterprise workflows
- Your platform reduces fragility from custom chain sprawl

### Landmines To Avoid
- Do not dismiss developer productivity claims
- Avoid framing that flexibility is bad; frame unmanaged flexibility as risk

### Proof You Need In Pilots
- Stability metrics (failed runs, response consistency)
- Access-control test pass rates
- Cost of support incidents avoided

## 5) Haystack DIY Build

### Why Buyers Consider It
- Open-source RAG framework with enterprise relevance
- Strong retrieval pipeline capabilities
- Self-host friendly

### Typical Objection
- "Haystack already gives us the RAG pipeline we need."

### Win Themes
- Your differentiated value is packaged operations + service + predictable pricing
- Faster deployment for non-ML-heavy organizations
- Better fit for teams optimizing operational outcomes, not framework assembly

### Landmines To Avoid
- Avoid purely technical comparison that ignores buyer role (operations leader)
- Do not skip change-management and onboarding narrative

### Proof You Need In Pilots
- Operator adoption metrics
- Reduction in document search time
- Onboarding acceleration outcomes

## Cross-Competitor Talk Track (Use In Sales Calls)

- Core message:
  - "You can build many RAG stacks. The question is how fast you can deploy, govern, and operate it with predictable cost in an air-gapped environment."
- Cost message:
  - "We are priced for SMB reality ($850/month engagement model) and backed by direct support."
- Risk message:
  - "DIY lowers license cost but often raises integration and support risk."

## Quick Discovery Questions To Qualify Deals

1. Do you have internal engineering bandwidth to own RAG operations long-term?
2. Is fully air-gapped runtime a hard requirement or a preference?
3. How much process disruption can you tolerate during rollout?
4. Do you need fixed/predictable monthly cost for approvals?
5. Is speed-to-value more important than maximum technical flexibility?

## Recommended Pilot KPIs (To Strengthen Battlecards)

1. Time-to-first-production-answer
2. Weekly hours saved in document search and onboarding
3. User activation/adoption rate
4. Retrieval precision with access-control compliance
5. Support ticket volume and resolution time
