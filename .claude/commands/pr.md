---
description: Execute a "main stays green" PR workflow for this monorepo
argument-hint: [pr-number]
---

# PR Workflow

Goal: Execute a "main stays green" PR workflow for this monorepo.

## Inputs
- A PR number (e.g. `/pr 123`) or a diff/branch description.

## Workflow
1) Summarize scope: backend, frontend, or fullstack.
2) If fullstack: generate/update the Contract Artifact before proposing code changes.
3) Produce a patch plan broken into small commits.
4) List the exact verification gates to run locally:
   - backend pytest (scoped if possible)
   - frontend lint/build/test
5) Provide a PR description template:
   - What / Why / How
   - Contract changes (if any)
   - How to verify
   - Risks / Rollback

## Output format
- Checklist + suggested commit messages + commands.
