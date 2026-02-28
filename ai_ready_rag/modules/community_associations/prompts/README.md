# Community Associations Module — Prompt Overrides

This directory holds module-level prompt overrides for the Community Associations (CA) vertical.

## How Prompt Resolution Works

`PromptResolver` searches for prompt `.txt` files in this order (first match wins):

1. `tenant-instances/{tenant_id}/prompts/{name}.txt` — tenant-specific override
2. `ai_ready_rag/modules/community_associations/prompts/{name}.txt` — **this directory**
3. `ai_ready_rag/prompts/{name}.txt` — core defaults

## Overriding Enrichment Prompts for CA

To customize synopsis extraction for insurance-aware CA document handling, place a file here:

```
ai_ready_rag/modules/community_associations/prompts/enrichment_synopsis.txt
```

This file will be used instead of the core `enrichment_synopsis.txt` for any
`ClaudeEnrichmentService` instantiated with `module_id="community_associations"`.

**Example use case**: The CA module may need to extract HOA-specific fields (reserve study
amounts, board meeting references, CC&R section numbers) that the generic core prompt does
not include.

## Existing Prompts in This Directory

| File | Purpose |
|------|---------|
| `appraisal.txt` | Property appraisal document analysis |
| `board_minutes.txt` | Board meeting minutes extraction |
| `ccr_bylaws.txt` | CC&R and bylaws parsing |
| `reserve_study.txt` | Reserve study financial analysis |
| `unit_owner_letter.txt` | Unit owner correspondence parsing |

## Per-Tenant Overrides

Individual tenants can further override any prompt by placing a file at:

```
tenant-instances/{tenant_id}/prompts/enrichment_synopsis.txt
```

Tenant overrides take priority over both module and core prompts.
