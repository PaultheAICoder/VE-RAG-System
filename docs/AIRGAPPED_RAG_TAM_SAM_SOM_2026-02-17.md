# Air-Gapped Enterprise RAG Market Sizing (TAM/SAM/SOM)

Date: 2026-02-17  
Scope: VE-RAG-System growth outlook for demo -> pilot horizon

## 1) Context from Internal Competitor Files
Reviewed:
- `docs/competitor-analysis-2026-02-10.md`
- `docs/competitor-battle-matrix-2026-02-10.md`
- `docs/competitor-pricing-snapshot-2026-02-10.md`
- `docs/competitor-ui-demo-library-2026-02-10.md`

Key assumptions already defined internally:
- Price target: **$850/month** per engagement
- Pilot structure: **6-month paid pilot**
- Conversion goal: **50%**
- CAC ceiling: **$1,500 per paid pilot**
- Budget through pilot horizon: **< $10k**
- Positioning: **air-gapped security + lower total cost + customer service**

---

## 2) Enterprise RAG / Sovereign AI Market Status (Quant Signals)

### Core market demand indicators
- **Global RAG market (2025): $1.94B**, projected to **$9.86B by 2030** (38.4% CAGR).  
  Source: MarketsandMarkets PR release via PRNewswire.
- **Sovereign cloud IaaS (2026): $80.4B**, +35.6% YoY. Governments + regulated industries + critical infrastructure are named primary buyers.  
  Source: Gartner (Feb 2026).
- **North America sovereign cloud IaaS (2026): $16.4B**.  
  Source: Gartner (Feb 2026).
- **Sovereign AI opportunity by 2030: ~$600B**, with up to **40% of AI workloads** potentially moving to sovereign environments in public/regulated sectors.  
  Source: McKinsey (Dec 2025/Jan 2026 period).
- Survey directional evidence: **>50% of AI workloads** already sit in private cloud + on-prem combinations, with security/compliance as top drivers.  
  Source: GTT study press release (2025).

### Why this matters for VE-RAG-System
The demand center is shifting from generic “AI enthusiasm” to **deployable, governed, private AI**. Your architecture (local models + local vector + RBAC/tag filtering + optional no-internet operation) is aligned with this trend.

---

## 3) TAM / SAM / SOM Model

## 3.1 TAM (Top-Down)
Method:
1. Start from RAG market baseline (2025 = $1.94B, CAGR 38.4%).
2. Estimate air-gapped/private-hosted subset as **20% to 40%** of RAG spend.
   - Lower bound reflects conservative private deployment share.
   - Upper bound informed by sovereign-workload trend (up to 40% in relevant sectors).

RAG total projection:
- 2026: **$2.68B**
- 2027: **$3.71B**
- 2028: **$5.13B**
- 2029: **$7.10B**
- 2030: **$9.86B**

Air-gapped/private-hosted RAG TAM (global):
- 2026: **$0.54B to $1.07B**
- 2030: **$1.97B to $3.94B**

If we apply Gartner sovereign-cloud regional split as a proxy (North America ~20.4% in 2026), then NA air-gapped RAG TAM in 2026 is roughly:
- **$0.11B to $0.22B**

Interpretation:
- Even conservative top-down sizing implies a **nine-figure regional opportunity** and a **high-growth category**.

---

## 3.2 SAM (Serviceable Available Market)
Given your current product maturity and pricing posture, a realistic SAM should focus on:
- regulated/document-heavy organizations,
- with clear compliance/security sensitivity,
- and willingness to adopt sub-$25k annual software engagements.

Useful U.S. demand anchors (not all immediately serviceable):
- FDIC-insured institutions: **4,421** (Q2 2025)
- U.S. hospitals: **6,100** (AHA 2026 fast facts)
- U.S. local governments: **90,837** (2022 Census of Governments)
- U.S. community water systems: about **49,000** (EPA)

These total >150k entities, but your practical early-service coverage is much smaller.

Working SAM assumptions (near-term):
- Addressable accounts in first expansion wave: **5,000 to 12,000 organizations**
- Effective ACV target (pilot -> production blended): **$10.2k to $18k**

Estimated SAM:
- **$51M to $216M ARR-equivalent**

Interpretation:
- Your SAM is large enough to scale meaningfully without needing enterprise mega-deals first.

---

## 3.3 SOM (Serviceable Obtainable Market)
Capacity-constrained by current budget and motion.

From your stated constraints:
- Max paid pilots with current CAC cap/budget profile is roughly aligned with your goal of **6 paid pilots** in current window.

Base-case SOM (next 12 months):
- 12 paid pilots/year at $850/mo for 6 months -> **$61.2k pilot revenue**
- 50% conversion -> 6 converted customers
- Converted ARR at $10.2k each -> **$61.2k ARR run-rate**
- Total first-year revenue envelope (pilot billings + annualized converted run-rate effect): **~$120k range**

Conservative case:
- 6 pilots/year, 50% conversion -> **$30.6k ARR run-rate**

Aggressive case (requires channel/partner boost):
- 24 pilots/year, 50% conversion -> **$122.4k ARR run-rate**

Interpretation:
- Near-term SOM is **go-to-market limited**, not market-size limited.

---

## 4) Growth Outlook Relative to Release Plan

## Next 30 days (demo)
Primary objective: maximize conversion confidence, not feature breadth.

Growth levers:
1. Tight vertical demo packs (insurance, legal, property management) with real docs.
2. Security narrative first: air-gap, local inference, data residency, auditability.
3. Time-to-value metric in demo: "search-to-answer time" and "onboarding speed".

## Next 45 days (pilot)
Primary objective: pilot throughput and referenceability.

Growth levers:
1. Standardized pilot offer (6 months, fixed scope, fixed success KPIs).
2. Weekly pilot health scorecards (adoption, query success, latency, fallback rate).
3. Reference pipeline: convert 1-2 pilots into case-study-grade proof.

---

## 5) What This Implies for Planning

1. **Market headroom is strong** (TAM/SAM are not the bottleneck).
2. **Execution bottleneck is GTM capacity + proof velocity**.
3. Winning path over next 6 months is:
   - high-conviction niche messaging,
   - repeatable pilot package,
   - rigorous conversion operations.

---

## 6) Risks in the Model

- RAG market reports vary widely by analyst methodology.
- Sovereign/private-cloud spend is broader than RAG alone; this is a proxy, not a one-to-one mapping.
- SAM depends heavily on your qualification criteria (compliance need, IT maturity, budget owner).
- SOM assumes CAC discipline and 50% conversion hold in real pipeline conditions.

---

## 7) Sources

- Gartner: Sovereign cloud IaaS spending forecast (Feb 9, 2026)  
  https://www.gartner.com/en/newsroom/press-releases/2026-02-09-gartner-says-worldwide-sovereign-cloud-iaas-spending-will-total-us-dollars-80-billion-in-2026
- Gartner: GenAI spending forecast 2025 (Mar 31, 2025)  
  https://www.gartner.com/en/newsroom/press-releases/2025-03-31-gartner-forecasts-worldwide-genai-spending-to-reach-644-billion-in-2025
- IDC: AI spending to $632B by 2028 (via BusinessWire release)  
  https://www.businesswire.com/news/home/20240819177906/en/Worldwide-Spending-on-Artificial-Intelligence-Forecast-to-Reach-%24632-Billion-in-2028-According-to-a-New-IDC-Spending-Guide
- McKinsey: Sovereign AI agenda and market estimate  
  https://www.mckinsey.com/capabilities/tech-and-ai/our-insights/tech-forward/the-sovereign-ai-agenda-moving-from-ambition-to-reality
- GTT private cloud/AI workload survey release  
  https://www.gtt.net/about-us/press-releases/gtt-study-private-cloud-adoption-accelerates-as-enterprises-prioritize-security-compliance-and-now-ai/
- RAG market sizing release (MarketsandMarkets via PRNewswire)  
  https://www.prnewswire.com/news-releases/retrieval-augmented-generation-rag-market-worth-9-86-billion-by-2030--marketsandmarkets-302580695.html
- SBA Office of Advocacy small business FAQs (2026)  
  https://advocacy.sba.gov/2026/02/03/frequently-asked-questions-about-small-business-2026/
- FDIC Q2 2025 banking profile release  
  https://www.fdic.gov/news/press-releases/2025/fdic-insured-institutions-reported-return-assets-113-percent-and-net
- AHA Fast Facts on U.S. Hospitals, 2026  
  https://www.aha.org/statistics/fast-facts-us-hospitals
- Census of Governments count (quoted in St. Louis Fed explainer)  
  https://www.stlouisfed.org/publications/regional-economist/2024/march/local-governments-us-number-type
- EPA public water systems overview  
  https://www.epa.gov/dwreginfo/information-about-public-water-systems


---

## 8) Pricing Update (User-Preferred Model)

Updated commercial model:
- One-time startup fee: **$4,000**
- Monthly license: **$850**
- Monthly onsite hardware fee: **$150**
- Total monthly recurring charge: **$1,000/month**

Unit economics (per customer):
- 6-month pilot contract value: **$10,000** (`$4,000 + $1,000 x 6`)
- 12-month first-year contract value: **$16,000** (`$4,000 + $1,000 x 12`)
- Steady-state recurring ARR (excluding startup fee): **$12,000/customer/year**

Updated SOM-style revenue scenarios (12-month view):

| Scenario | Paid Pilots / Year | Pilot Revenue (6-month pilots) | Conversions @50% | Recurring ARR Run-Rate from Converted Customers |
|---|---:|---:|---:|---:|
| Conservative | 6 | **$60,000** | 3 | **$36,000 ARR** |
| Base | 12 | **$120,000** | 6 | **$72,000 ARR** |
| Aggressive | 24 | **$240,000** | 12 | **$144,000 ARR** |

Interpretation with this pricing:
- Your pilot cash collection improves meaningfully vs pure monthly pricing due to the upfront startup fee.
- Growth still depends primarily on pilot volume + conversion rate, not TAM constraints.
- Recurring scale is driven by the **$12k ARR per converted customer** baseline; startup fees improve near-term cash but are non-recurring.


---

## 9) Conservative Growth Forecast (2 / 5 / 10 Years)

Assumptions (explicit):
1. Pricing stays fixed at:
   - Startup fee: `$4,000` (one-time)
   - Recurring: `$1,000/month` (`$850 license + $150 onsite hw`)
2. Pilot contract duration: `6 months` (`$10,000` per pilot).
3. New paid pilots in Year 1: `6`.
4. Paid pilot volume growth:
   - `+15% YoY` in Years 2-5
   - `+8% YoY` in Years 6-10
5. Pilot-to-recurring conversion: `50%` (at end of pilot year).
6. Annual recurring customer retention: `88%` (12% churn).
7. No price increases, no upsell/cross-sell, no enterprise tier expansion.
8. Annual recognized recurring revenue is calculated from recurring customers at start-of-year (conservative recognition).

Derived unit values:
- Pilot revenue per new customer: `$10,000`
- Recurring ARR per converted customer: `$12,000/year`

### Forecast Results

| Horizon | New Paid Pilots (that year) | Annual Pilot Revenue | Recurring Customers (end of year) | ARR Run-Rate (end of year) | Annual Revenue (Pilot + Recognized Recurring) |
|---|---:|---:|---:|---:|---:|
| Year 2 | 6.90 | $69,000 | 6.09 | $73,080 | $105,000 |
| Year 5 | 10.50 | $104,956 | 16.49 | $197,848 | $258,193 |
| Year 10 | 15.43 | $154,260 | 35.42 | $425,014 | $531,958 |

### Readout
- Under conservative assumptions, growth is steady but moderate.
- Year-10 ARR run-rate is about **$425k** without any price lift or upsell.
- The biggest levers are still pilot volume and conversion rate; pricing changes matter less than pipeline throughput in this model.


---

## 10) Revised 2 / 5 / 10-Year Forecast (Conservative vs Realistic vs Aggressive)

This section supersedes the prior single-scenario forecast.

Common pricing assumptions (all scenarios):
- Startup fee: `$4,000` one-time
- Recurring fee: `$1,000/month` (`$850 license + $150 onsite hw`)
- Pilot revenue per new customer (6 months): `$10,000`
- ARR per converted recurring customer: `$12,000`
- No price increases or upsell modeled (flat pricing)

Model mechanics:
- Annual revenue = `pilot revenue + recurring revenue recognized from start-of-year recurring customers`
- ARR run-rate (end of year) = `end-of-year recurring customers * $12,000`

### Scenario Assumptions

| Scenario | Year-1 Paid Pilots | Pilot Growth (Years 2-5) | Pilot Growth (Years 6-10) | Conversion | Retention |
|---|---:|---:|---:|---:|---:|
| Conservative | 6 | 15% | 8% | 50% | 88% |
| Realistic | 12 | 22% | 12% | 58% | 92% |
| Aggressive | 24 | 30% | 18% | 65% | 94% |

### Forecast Comparison (Years 2, 5, 10)

| Scenario | Horizon | New Paid Pilots (that year) | Annual Pilot Revenue | Recurring Customers (end of year) | ARR Run-Rate (end of year) | Annual Revenue |
|---|---|---:|---:|---:|---:|---:|
| Conservative | Year 2 | 6.90 | $69,000 | 6.09 | $73,080 | $105,000 |
| Conservative | Year 5 | 10.49 | $104,940 | 16.48 | $197,817 | $258,182 |
| Conservative | Year 10 | 15.42 | $154,192 | 35.38 | $424,543 | $531,497 |
| Realistic | Year 2 | 14.64 | $146,400 | 14.89 | $178,733 | $229,920 |
| Realistic | Year 5 | 26.58 | $265,840 | 47.41 | $568,946 | $683,145 |
| Realistic | Year 10 | 46.85 | $468,501 | 126.51 | $1,518,112 | $1,764,192 |
| Aggressive | Year 2 | 31.20 | $312,000 | 34.94 | $419,328 | $499,200 |
| Aggressive | Year 5 | 68.55 | $685,464 | 129.09 | $1,549,094 | $1,764,647 |
| Aggressive | Year 10 | 156.82 | $1,568,176 | 435.13 | $5,221,586 | $5,821,802 |

### Quick Readout
- Conservative path stays sub-$1M annual revenue through Year 10.
- Realistic path crosses ~$1.5M ARR run-rate by Year 10.
- Aggressive path reaches multi-million ARR by Year 10, but requires sustained high pilot throughput and strong retention.


---

## 11) Sensitivity Analysis (Requested)

Sensitivity base case:
- Uses the **Realistic** scenario growth assumptions from Section 10.
- Tests conversion and retention variation around baseline:
  - Baseline conversion: `58%`
  - Baseline retention: `92%`
- Measures impact on **Year-10 ARR run-rate**.

### Year-10 ARR Sensitivity Matrix

| Retention \ Conversion | 48% | 58% (Baseline) | 68% |
|---|---:|---:|---:|
| 87% | $1,084,014 | $1,309,850 | $1,535,686 |
| 92% (Baseline) | $1,256,369 | $1,518,112 | $1,779,855 |
| 97% | $1,470,919 | $1,777,360 | $2,083,802 |

### Sensitivity Readout
- Conversion and retention both materially affect long-term ARR.
- Around the realistic baseline, a **+10 point conversion improvement** (58% -> 68%) increases Year-10 ARR by about **$261,743**.
- A **+5 point retention improvement** (92% -> 97%) increases Year-10 ARR by about **$259,248**.
- Combined improvement (68% conversion + 97% retention) lifts Year-10 ARR to about **$2.08M**.

Practical implication:
- Your highest-return growth investments are likely in:
  1. Pilot qualification and success criteria (improves conversion)
  2. Onboarding + support playbooks (improves retention)

