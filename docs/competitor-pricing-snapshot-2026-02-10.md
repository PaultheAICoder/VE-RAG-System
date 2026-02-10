# Competitor Pricing Snapshot (Public Pricing Evidence)

**As of:** February 10, 2026  
**Purpose:** Add concrete pricing visibility for the 15-competitor set.  
**Important:** Prices can change; verify before quoting in sales conversations.

## Pricing Summary Table

| Competitor | Public Pricing Signal | Approx Monthly Equivalent | SMB Affordability vs <$1k/mo | Notes | Source |
|---|---|---:|---|---|---|
| OpenSearch (OSS / AWS service) | OSS is free; AWS service is pay-as-you-go with no minimum/setup fee | Varies | Potentially SMB-friendly for small workloads | Cost depends on instance/storage/transfer | https://aws.amazon.com/opensearch-service/pricing/ |
| Qdrant | Managed Cloud starts at $0 (1GB free cluster) | $0+ | SMB-friendly start | Hybrid/Private Cloud are custom | https://qdrant.tech/pricing/ |
| Weaviate | Flex starts at $45/mo; Premium starts at $400/mo | $45+ / $400+ | SMB-friendly entry | BYOC and larger deployments are sales-led | https://weaviate.io/pricing |
| Haystack (OSS) | Open-source framework (no license fee) | $0 license + infra | SMB-friendly if team can self-manage | Real cost is engineering/ops time | https://github.com/deepset-ai/haystack |
| LangChain / LangSmith | Developer $0/seat; Plus $39/seat/mo | $0-$39+/seat + usage | SMB-friendly entry | Enterprise/self-host options are custom | https://www.langchain.com/pricing |
| LlamaIndex / LlamaCloud | Free $0; Starter $50/mo; Pro $500/mo; Enterprise custom | $0/$50/$500 | SMB-friendly at low tiers | Credit-based usage model; higher volume increases cost | https://www.llamaindex.ai/pricing |
| deepset AI Platform | Studio $0; Enterprise custom | $0+ | Mixed | Production deployment pricing is custom | https://www.deepset.ai/pricing |
| Elastic | Usage-based pricing via calculator; no fixed starter price on main page | Varies | Often above SMB at production scale | Resource-based + support tier effects | https://www.elastic.co/pricing/ |
| AWS Kendra | GenAI Enterprise base index: $0.32/hour (example = $230.4/mo) | ~$230.4/mo base + extras | Can start under $1k, may rise quickly | Additional storage/query units and connectors add cost | https://aws.amazon.com/kendra/pricing/ |
| Azure AI Search | Pricing table present but public page currently shows `$-` placeholders in captured view | Not reliably extractable from page snapshot | Unclear from captured page | Use Azure calculator/quote for exact regional price | https://azure.microsoft.com/en-us/pricing/details/search/ |
| Vertex AI Search | Standard Edition $1.50/1,000 queries; Enterprise Edition $4.00/1,000 queries | Usage-based | Can be SMB-friendly at low volume | Configurable plan also has minimum commitment model | https://cloud.google.com/generative-ai-app-builder/pricing |
| SearchBlox | Self-managed: $25,000/yr single server; $75,000/yr HA. Managed plans include $24k/$36k/$48k per year tiers | ~$2,000-$6,250+/mo | High SMB cost | Strong on-prem story but above your target price band | https://www.searchblox.com/pricing |
| IBM watsonx (family signal) | watsonx.ai Essentials starts $0; Standard starts $1,050/mo | $0+ / $1,050+ | Usually above SMB target at production tier | watsonx Orchestrate page is tiered but no numeric price shown in captured page | https://www.ibm.com/products/watsonx-ai/pricing , https://www.ibm.com/products/watsonx-orchestrate/pricing |
| Sinequa | No public numeric pricing found on product page | Custom | High SMB cost risk | Sales-led enterprise pricing | https://www.sinequa.com/product/ |
| Mindbreeze InSpire | Starts at EUR 83,000/yr (USD 103,700/yr) for small org package | ~$8,641/mo (USD basis) | High SMB cost | Published annual pricing; higher tiers are sales-led | https://www.mindbreeze.com/pricing |

## Normalized Price Notes (For Internal Use)

- SearchBlox self-managed single server: `$25,000/yr = ~$2,083/mo`
- SearchBlox self-managed HA cluster: `$75,000/yr = ~$6,250/mo`
- SearchBlox managed Hybrid Search tier: `$24,000/yr = ~$2,000/mo`
- Mindbreeze published USD starter: `$103,700/yr = ~$8,641/mo`
- AWS Kendra base index example: `$0.32/hr x 720h = $230.4/mo` (before add-ons)

## Confidence Labels

- **High confidence:** Qdrant, Weaviate, deepset, LangChain/LangSmith, LlamaIndex, AWS Kendra, SearchBlox, Mindbreeze, IBM watsonx.ai
- **Medium confidence:** OpenSearch total monthly (usage-based), Vertex AI Search total monthly (depends on traffic mix), Elastic total monthly (calculator-driven)
- **Low/opaque confidence:** Azure AI Search exact unit pricing from public page snapshot (`$-` placeholders), Sinequa numeric pricing (sales-led/custom)

