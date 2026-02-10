# Snowflake Cortex Search Pricing Estimate

**Date:** February 10, 2026  
**Scope:** Reasonable-cost estimate using public Snowflake pricing mechanics and explicit assumptions.

## Assumptions

- Region: AWS US West (Oregon)
- Credit price scenarios: `$2`, `$3`, `$4` per credit
- Embedding model: `snowflake-arctic-embed-m-v1.5` at `0.03 credits / 1M tokens`
- Serving rate: `6.3 credits / GB-month` indexed data (example rate from Snowflake docs)
- Average row/chunk payload:
  - Text: `1,000 bytes`
  - Embedding: `768 dims x 4 bytes = 3,072 bytes`
  - Total indexed bytes per row: `4,072 bytes`
- Monthly updates: `5%` of rows
- Warehouse for refresh/indexing:
  - SMB: `20` XS hours/month
  - Growth: `40` XS hours/month
  - Large: `120` XS hours/month

## Monthly Recurring Estimate (Cortex Search-Oriented)

| Scenario | Rows Indexed | Estimated Monthly Credits (Recurring) | USD @ $2/credit | USD @ $3/credit | USD @ $4/credit |
|---|---:|---:|---:|---:|---:|
| SMB pilot | 500,000 | ~33.1 | ~$66 | ~$99 | ~$132 |
| Growth | 2,000,000 | ~92.2 | ~$184 | ~$277 | ~$369 |
| Large | 10,000,000 | ~381.0 | ~$762 | ~$1,143 | ~$1,524 |

## One-Time Initial Indexing Add-On (First Month)

| Scenario | Initial Embedding Credits | USD @ $2 | USD @ $3 | USD @ $4 |
|---|---:|---:|---:|---:|
| 500k rows | 4.5 | $9 | $13.5 | $18 |
| 2M rows | 18 | $36 | $54 | $72 |
| 10M rows | 90 | $180 | $270 | $360 |

## Interpretation

- Snowflake can start low at small scale, but total cost rises with:
  - indexed data volume,
  - query volume,
  - refresh frequency,
  - warehouse activity,
  - and contracted credit price.
- For SMB buyers, Snowflake can be competitive at low usage but may exceed a strict sub-$1k target as volume grows.

## Caveats

- This is an estimate, not a quote.
- Final cost depends on contracted credit rates and architecture choices.
- Additional platform costs can apply (storage, cloud services, orchestration, app layer).

## Sources

- Cortex Search costs and components:  
  https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-costs
- Cortex Search overview and model usage context:  
  https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-search/cortex-search-overview
- Snowflake service consumption / credit reference:  
  https://www.snowflake.com/wp-content/uploads/2017/07/CreditConsumptionTable.pdf
