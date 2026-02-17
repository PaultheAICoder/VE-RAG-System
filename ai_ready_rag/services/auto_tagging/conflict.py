"""Conflict resolution, guardrail enforcement, and provenance for auto-tagging.

Resolves namespace-specific conflicts between path-based and LLM-based tags,
enforces max_tags_per_doc guardrails, and builds provenance JSON for the
auto_tag_source column.
"""

from __future__ import annotations

from typing import TypedDict

from ai_ready_rag.services.auto_tagging.classifier import ClassificationResult
from ai_ready_rag.services.auto_tagging.models import AutoTag

# Namespace authority table: which source wins on conflict
PATH_AUTHORITATIVE = {"client", "year", "stage"}
LLM_AUTHORITATIVE = {"doctype", "topic", "entity"}


class ConflictRecord(TypedDict):
    """Record of a single namespace conflict between path and LLM tags."""

    namespace: str
    path_value: str
    llm_value: str
    winner: str  # "path" | "llm"
    reason: str


class Provenance(TypedDict, total=False):
    """Provenance JSON matching spec Section 8.5 schema."""

    strategy_id: str
    strategy_version: str
    path_candidates: list[dict]
    llm_candidates: list[dict]
    conflicts: list[ConflictRecord]
    applied: list[str]
    discarded: list[str]
    suggested: list[str]
    truncated: list[str]


def resolve_conflicts(
    path_tags: list[AutoTag],
    llm_result: ClassificationResult,
    confidence_threshold: float,
) -> tuple[list[AutoTag], list[AutoTag], list[ConflictRecord]]:
    """Resolve namespace conflicts between path and LLM tags.

    Args:
        path_tags: Tags from path parsing (already applied at upload).
        llm_result: ClassificationResult from DocumentClassifier.
        confidence_threshold: Minimum confidence for LLM to win conflict.

    Returns:
        Tuple of (winning_llm_tags, losing_path_tags, conflict_records).
        - winning_llm_tags: LLM tags that should be ADDED to document.tags
          (includes non-conflicting and conflict-winners).
        - losing_path_tags: Path tags that should be REMOVED from document.tags
          (only when LLM won a conflict in their namespace).
        - conflict_records: List of conflict records for provenance.
    """
    # Build dict of path tags by namespace
    path_by_ns: dict[str, list[AutoTag]] = {}
    for pt in path_tags:
        path_by_ns.setdefault(pt.namespace, []).append(pt)

    winning_llm: list[AutoTag] = []
    losing_path: list[AutoTag] = []
    conflicts: list[ConflictRecord] = []

    for llm_tag in llm_result.tags:
        ns = llm_tag.namespace
        path_in_ns = path_by_ns.get(ns)

        if not path_in_ns:
            # No conflict: LLM tag passes through
            winning_llm.append(llm_tag)
            continue

        # Conflict exists
        if ns in PATH_AUTHORITATIVE:
            # Path wins unconditionally
            for pt in path_in_ns:
                conflicts.append(
                    ConflictRecord(
                        namespace=ns,
                        path_value=pt.value,
                        llm_value=llm_tag.value,
                        winner="path",
                        reason=f"{ns}: path is authoritative",
                    )
                )
        elif ns in LLM_AUTHORITATIVE:
            if llm_tag.confidence >= confidence_threshold:
                # LLM wins
                winning_llm.append(llm_tag)
                for pt in path_in_ns:
                    if pt not in losing_path:
                        losing_path.append(pt)
                    conflicts.append(
                        ConflictRecord(
                            namespace=ns,
                            path_value=pt.value,
                            llm_value=llm_tag.value,
                            winner="llm",
                            reason=(
                                f"{ns}: LLM confidence {llm_tag.confidence} "
                                f">= threshold {confidence_threshold}"
                            ),
                        )
                    )
            else:
                # LLM below threshold, path wins
                for pt in path_in_ns:
                    conflicts.append(
                        ConflictRecord(
                            namespace=ns,
                            path_value=pt.value,
                            llm_value=llm_tag.value,
                            winner="path",
                            reason=(
                                f"{ns}: LLM confidence {llm_tag.confidence} "
                                f"below threshold {confidence_threshold}, path wins"
                            ),
                        )
                    )
        else:
            # Unknown namespace: default to path wins
            for pt in path_in_ns:
                conflicts.append(
                    ConflictRecord(
                        namespace=ns,
                        path_value=pt.value,
                        llm_value=llm_tag.value,
                        winner="path",
                        reason=f"{ns}: unknown namespace, path wins by default",
                    )
                )

    return winning_llm, losing_path, conflicts


def enforce_guardrail(
    manual_tag_names: set[str],
    path_tags: list[AutoTag],
    llm_tags: list[AutoTag],
    max_tags: int,
) -> tuple[list[AutoTag], list[AutoTag], list[str]]:
    """Enforce max_tags_per_doc guardrail.

    Priority order: manual first, then path (by rule order), then LLM (by
    confidence descending).

    Args:
        manual_tag_names: Set of manual tag names (always kept).
        path_tags: Path-derived AutoTag objects (kept after manual).
        llm_tags: LLM-derived AutoTag objects (kept last, sorted by confidence desc).
        max_tags: Maximum total tags allowed.

    Returns:
        Tuple of (kept_path_tags, kept_llm_tags, truncated_tag_names).
    """
    manual_count = len(manual_tag_names)
    budget = max_tags - manual_count

    if budget <= 0:
        # No room for any auto tags
        truncated = [at.tag_name for at in path_tags] + [at.tag_name for at in llm_tags]
        return [], [], truncated

    # Keep path tags up to budget
    kept_path: list[AutoTag] = []
    truncated: list[str] = []

    for pt in path_tags:
        if len(kept_path) < budget:
            kept_path.append(pt)
        else:
            truncated.append(pt.tag_name)

    remaining_budget = budget - len(kept_path)

    # Sort LLM tags by confidence descending, keep up to remaining budget
    sorted_llm = sorted(llm_tags, key=lambda t: t.confidence, reverse=True)
    kept_llm: list[AutoTag] = []

    for lt in sorted_llm:
        if len(kept_llm) < remaining_budget:
            kept_llm.append(lt)
        else:
            truncated.append(lt.tag_name)

    return kept_path, kept_llm, truncated


def build_provenance(
    strategy_id: str,
    strategy_version: str,
    path_tags: list[AutoTag],
    llm_result: ClassificationResult | None,
    conflicts: list[ConflictRecord],
    applied_tag_names: list[str],
    discarded_tag_names: list[str],
    suggested_tag_names: list[str],
    truncated_tag_names: list[str] | None = None,
) -> dict:
    """Build provenance JSON for auto_tag_source column.

    Returns dict matching spec Section 8.5 schema.
    """
    path_candidates = [
        {"namespace": t.namespace, "value": t.value, "confidence": t.confidence} for t in path_tags
    ]

    llm_candidates: list[dict] = []
    if llm_result is not None:
        all_llm_tags = (
            list(llm_result.tags) + list(llm_result.suggested) + list(llm_result.discarded)
        )
        llm_candidates = [
            {"namespace": t.namespace, "value": t.value, "confidence": t.confidence}
            for t in all_llm_tags
        ]

    provenance: dict = {
        "strategy_id": strategy_id,
        "strategy_version": strategy_version,
        "path_candidates": path_candidates,
        "llm_candidates": llm_candidates,
        "conflicts": list(conflicts),
        "applied": applied_tag_names,
        "discarded": discarded_tag_names,
        "suggested": suggested_tag_names,
    }

    if truncated_tag_names:
        provenance["truncated"] = truncated_tag_names

    return provenance
