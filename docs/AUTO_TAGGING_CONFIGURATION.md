# Auto-Tagging Configuration Guide

This guide explains how to configure automatic document tagging in VE-RAG. Auto-tagging assigns tags to uploaded documents based on folder structure and document content, eliminating manual tag selection.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [How It Works](#2-how-it-works)
3. [Enable Auto-Tagging](#3-enable-auto-tagging)
4. [Choose a Strategy](#4-choose-a-strategy)
5. [Create a Custom Strategy](#5-create-a-custom-strategy)
6. [Upload Documents](#6-upload-documents)
7. [Manage Auto-Created Tags](#7-manage-auto-created-tags)
8. [Approval Mode](#8-approval-mode)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Quick Start

```bash
# 1. Add to .env
AUTO_TAGGING_ENABLED=true
AUTO_TAGGING_STRATEGY=generic

# 2. Restart the server
sudo systemctl restart ve-rag   # or your restart method

# 3. Upload with folder path preserved
#    The system automatically tags based on folder structure
```

That's it for basic usage. Read on for strategy selection and customization.

---

## 2. How It Works

Auto-tagging runs in two stages:

| Stage | When | Source | Cost |
|-------|------|--------|------|
| **Path-based** | At upload time (instant) | Folder path | Zero |
| **LLM-based** | After document processing (async) | Document content | 1 LLM call per document |

**Example:** Uploading `Bethany Terrace (12-13)/24 NB/Quote/CNA/D&O Quote.pdf` with the `insurance_agency` strategy produces:

| Tag | Source |
|-----|--------|
| `client:bethany-terrace` | Path (folder level 0) |
| `year:2024-2025` | Path (folder level 1) |
| `stage:quote` | Path (folder level 2) |
| `entity:cna` | Path (folder level 3) |
| `doctype:quote` | LLM (content classification) |
| `topic:do` | LLM (coverage line detection) |

All auto-tags use a `namespace:value` format (e.g., `client:bethany-terrace`) to avoid collisions with existing manual tags like `hr` or `finance`.

---

## 3. Enable Auto-Tagging

Add these settings to your `.env` file:

```env
# Required
AUTO_TAGGING_ENABLED=true
AUTO_TAGGING_STRATEGY=generic            # See Section 4 for options

# Optional (defaults shown)
AUTO_TAGGING_PATH_ENABLED=true           # Extract tags from folder paths
AUTO_TAGGING_LLM_ENABLED=true            # Classify content with LLM
AUTO_TAGGING_LLM_MODEL=qwen3:8b          # Ollama model for classification
AUTO_TAGGING_REQUIRE_APPROVAL=false      # Set true to require admin review
AUTO_TAGGING_CONFIDENCE_THRESHOLD=0.7    # Auto-apply LLM tags above this
AUTO_TAGGING_CREATE_MISSING_TAGS=true    # Auto-create tags that don't exist yet
```

Restart the server after changing `.env`.

**Verify** auto-tagging is active:

```bash
curl http://localhost:8502/api/health | jq '.auto_tagging'
```

Expected output:
```json
{
  "enabled": true,
  "strategy": "generic",
  "strategy_name": "Generic",
  "path_enabled": true,
  "llm_enabled": true
}
```

---

## 4. Choose a Strategy

Strategies are YAML files in `data/auto_tag_strategies/`. Each defines how folder paths map to tags and how the LLM classifies documents.

### Built-In Strategies

| Strategy ID | Industry | What It Extracts |
|-------------|----------|------------------|
| `generic` | Any | Client name from top folder, basic document types |
| `insurance_agency` | Insurance brokerage | Client, policy year, workflow stage, carrier, coverage lines |
| `law_firm` | Legal practice | Client, matter stage, opposing party, practice areas |
| `construction` | Construction/project mgmt | Project, phase, subcontractor/vendor, trade/discipline |

### Select a Strategy

Set the strategy ID in `.env`:

```env
AUTO_TAGGING_STRATEGY=insurance_agency
```

### Preview a Strategy

To see what a strategy will extract, use the API:

```bash
curl http://localhost:8502/api/admin/auto-tagging/strategies/insurance_agency | jq
```

Or read the YAML directly:

```bash
cat data/auto_tag_strategies/insurance_agency.yaml
```

---

## 5. Create a Custom Strategy

If none of the built-in strategies match your folder structure, create a custom one.

### Step 1: Copy a template

```bash
cp data/auto_tag_strategies/generic.yaml data/auto_tag_strategies/my_company.yaml
```

### Step 2: Edit the strategy header

```yaml
strategy:
  id: "my_company"
  name: "My Company"
  description: "Custom strategy for our document management"
  version: "1.0"
```

### Step 3: Define tag namespaces

Namespaces are the categories of tags your strategy produces. The four universal namespaces (`client`, `year`, `doctype`, `stage`) are recommended. Add `entity` and `topic` if your industry has domain-specific entities.

```yaml
namespaces:
  client:
    display: "Client"
    color: "#6366f1"
    description: "Customer or project name"
  year:
    display: "Year"
    color: "#f59e0b"
    description: "Document year or period"
  doctype:
    display: "Document Type"
    color: "#10b981"
    description: "Classification of document"
```

### Step 4: Define path rules

Path rules extract tags from folder names. Each rule targets a specific folder depth (level).

Given a file at: `Acme Corp/2025/Contracts/Smith_Agreement.pdf`
- Level 0 = `Acme Corp`
- Level 1 = `2025`
- Level 2 = `Contracts`

```yaml
path_rules:
  # Level 0: Client name
  - namespace: "client"
    level: 0
    pattern: "^(.+)$"          # Capture entire folder name
    capture_group: 1
    transform: "slugify"        # "Acme Corp" → "acme-corp"

  # Level 1: Year (if folder is a 4-digit year)
  - namespace: "year"
    level: 1
    pattern: "^(20\\d{2})$"    # Match "2024", "2025", etc.
    capture_group: 1

  # Level 2: Workflow stage (mapped from folder names)
  - namespace: "stage"
    level: 2
    mapping:
      "Contracts": "contract"
      "Invoices": "billing"
      "Reports": "reporting"
      "Archive": "archived"
```

**Available transforms:**

| Transform | Example Input | Example Output |
|-----------|--------------|----------------|
| `slugify` | `"Acme Corp"` | `"acme-corp"` |
| `year_range` | `"24"` | `"2024-2025"` |
| `lowercase` | `"CNA"` | `"cna"` |
| *(none)* | `"Quote"` | `"Quote"` |

**Optional rule fields:**

- `parent_match`: Only apply rule if the parent folder matches a regex. Example: extract carrier name only under `Quote/` or `Sub/` folders.
- `mapping`: Use instead of `pattern` for static folder name → tag value maps.

### Step 5: Define document types

These are the valid values for the `doctype:` namespace. The LLM picks from this list.

```yaml
document_types:
  contract:
    display: "Contract"
    description: "Legal agreement or contract"
    keywords: ["contract", "agreement", "terms", "msa"]
  invoice:
    display: "Invoice"
    description: "Billing document"
    keywords: ["invoice", "payment", "billing", "amount due"]
  report:
    display: "Report"
    description: "Analysis or summary report"
    keywords: ["report", "analysis", "summary", "findings"]
  unknown:
    display: "Unknown"
    description: "Could not classify"
    keywords: []
```

### Step 6: Configure LLM prompt (optional)

Customize the prompt the LLM receives for content classification:

```yaml
llm_prompt: |
  Classify this document. Return JSON only.

  Document filename: {filename}
  First 2000 characters of content:
  {content_preview}

  Return this exact JSON structure:
  {{
    "document_type": "<one of: {document_type_ids}>",
    "year_start": "<YYYY or null>",
    "year_end": "<YYYY or null>",
    "confidence": <0.0 to 1.0>
  }}
```

The `{document_type_ids}` placeholder is auto-filled from your `document_types` keys.

### Step 7: Activate your strategy

```env
AUTO_TAGGING_STRATEGY=my_company
```

Restart the server.

### Step 8: Test with dry-run

```bash
python -m ai_ready_rag.cli.bulk_upload \
    --path "/path/to/test/folder/" \
    --strategy my_company \
    --dry-run
```

This prints the tags that *would* be applied without uploading anything.

---

## 6. Upload Documents

### Single File (UI)

When auto-tagging is enabled, the upload dialog no longer requires manual tag selection. Provide the source path if uploading via API:

```bash
curl -X POST http://localhost:8502/api/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/document.pdf" \
  -F "source_path=Acme Corp/2025/Contracts/document.pdf"
```

### Bulk Upload (CLI)

Upload an entire customer folder:

```bash
python -m ai_ready_rag.cli.bulk_upload \
    --path "/path/to/Acme Corp/" \
    --strategy insurance_agency \
    --api-url http://localhost:8502
```

Upload all customer folders in a parent directory:

```bash
python -m ai_ready_rag.cli.bulk_upload \
    --path "/path/to/Customer_Projects/" \
    --strategy insurance_agency \
    --recursive \
    --api-url http://localhost:8502
```

**Useful flags:**

| Flag | Description |
|------|-------------|
| `--dry-run` | Preview tags without uploading |
| `--recursive` | Walk all subdirectories |
| `--skip-images` | Skip .jpg/.png files |
| `--skip-emails` | Skip .msg files |
| `--only-pdf` | Upload only PDF files |
| `--manual-tags hr,finance` | Add manual tags to all files |

---

## 7. Manage Auto-Created Tags

When `AUTO_TAGGING_CREATE_MISSING_TAGS=true` (default), the system creates tags as needed. For example, uploading a new client folder "Cedar Ridge" auto-creates the `client:cedar-ridge` tag.

### Assign auto-tags to users

Users only see documents with tags assigned to their profile. After auto-tagging creates `client:` tags, assign them to the appropriate users:

```bash
# List all client tags
curl http://localhost:8502/api/tags?prefix=client: \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Assign client tags to a user
curl -X POST http://localhost:8502/api/users/{user_id}/tags \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tag_ids": ["<client:bethany-terrace-tag-id>", "<client:cedar-ridge-tag-id>"]}'
```

### View tag facets

See how many documents exist per namespace:

```bash
curl http://localhost:8502/api/tags/facets \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## 8. Approval Mode

For sensitive deployments, enable approval mode so auto-tags are *suggested* rather than applied:

```env
AUTO_TAGGING_REQUIRE_APPROVAL=true
```

In this mode:
- **Path-based tags** with confidence = 1.0 are still auto-applied
- **LLM tags** above the confidence threshold (0.7) are auto-applied
- **LLM tags** between 0.4 and 0.7 are saved as *suggestions*
- **LLM tags** below 0.4 are saved as suggestions only

### Review suggestions

```bash
# View pending suggestions for a document
curl http://localhost:8502/api/documents/{doc_id}/tag-suggestions \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# Approve specific suggestions
curl -X POST http://localhost:8502/api/documents/{doc_id}/tag-suggestions/approve \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"suggestion_ids": ["suggestion-id-1", "suggestion-id-2"]}'

# Bulk approve across documents
curl -X POST http://localhost:8502/api/documents/tag-suggestions/approve-batch \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"suggestion_ids": ["id1", "id2", "id3"]}'
```

---

## 9. Troubleshooting

### Auto-tags not appearing

1. **Check `.env`:** Ensure `AUTO_TAGGING_ENABLED=true` is set
2. **Check health endpoint:** `curl http://localhost:8502/api/health | jq '.auto_tagging'`
3. **Check strategy exists:** Verify the YAML file is in `data/auto_tag_strategies/`
4. **Check source_path:** Path-based tags require `source_path` on upload. The bulk CLI provides this automatically.

### LLM tags not appearing

1. **Check LLM is enabled:** `AUTO_TAGGING_LLM_ENABLED=true`
2. **Check Ollama model:** Ensure the model in `AUTO_TAGGING_LLM_MODEL` is pulled (`ollama list`)
3. **Check document status:** LLM tagging runs after chunking. Wait for `status=ready`.
4. **Check confidence:** Tags below the confidence threshold may be in suggestions (see approval mode).

### Path rules not matching

Use `--dry-run` with the bulk CLI to debug:

```bash
python -m ai_ready_rag.cli.bulk_upload \
    --path "/path/to/folder/" \
    --strategy my_company \
    --dry-run
```

Common issues:
- **Wrong level number:** Level 0 is the first folder in the *relative* path, not the absolute path
- **Regex not matching:** Test your pattern at regex101.com. Remember YAML requires escaping backslashes (`\\d` not `\d`)
- **Missing parent_match:** If a rule has `parent_match`, it only fires when the parent folder matches

### Strategy YAML errors

If the strategy fails to load, the system falls back to `generic`. Check server logs for YAML parsing errors. Common mistakes:
- Indentation (use spaces, not tabs)
- Missing quotes around strings with special characters
- Duplicate keys in mappings

---

## Reference: Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_TAGGING_ENABLED` | `false` | Master switch |
| `AUTO_TAGGING_STRATEGY` | `generic` | Strategy ID (YAML filename without extension) |
| `AUTO_TAGGING_PATH_ENABLED` | `true` | Enable path-based tag extraction |
| `AUTO_TAGGING_LLM_ENABLED` | `true` | Enable LLM content classification |
| `AUTO_TAGGING_LLM_MODEL` | `qwen3:8b` | Ollama model for classification |
| `AUTO_TAGGING_REQUIRE_APPROVAL` | `false` | Require admin review for low-confidence tags |
| `AUTO_TAGGING_CONFIDENCE_THRESHOLD` | `0.7` | Auto-apply LLM tags at or above this confidence |
| `AUTO_TAGGING_CREATE_MISSING_TAGS` | `true` | Auto-create tags not yet in the database |
