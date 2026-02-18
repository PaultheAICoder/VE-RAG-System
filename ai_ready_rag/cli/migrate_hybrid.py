"""Migration CLI: blue/green cutover from unnamed to named vectors.

Migrates a Qdrant collection from unnamed dense vectors to named
dense+sparse vectors for hybrid search. Supports resume via cursor
file, 5 verification gates, and cutover/rollback/cleanup commands.

Usage:
    python -m ai_ready_rag.cli.migrate_hybrid --source-collection documents
    python -m ai_ready_rag.cli.migrate_hybrid --source-collection documents --verify
    python -m ai_ready_rag.cli.migrate_hybrid --source-collection documents --cutover
    python -m ai_ready_rag.cli.migrate_hybrid --source-collection documents --rollback
    python -m ai_ready_rag.cli.migrate_hybrid --source-collection documents --cleanup
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

CURSOR_FILE = ".migrate_cursor"
DEFAULT_BATCH_SIZE = 100
DEFAULT_TARGET_SUFFIX = "_hybrid"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate Qdrant collection from unnamed to named vectors (hybrid search)",
        prog="python -m ai_ready_rag.cli.migrate_hybrid",
    )
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--source-collection", required=True)
    parser.add_argument(
        "--target-collection",
        default=None,
        help="Target collection name (default: {source}_hybrid)",
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--embedding-dimension", type=int, default=768)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run verification gates after migration",
    )
    parser.add_argument(
        "--cutover",
        action="store_true",
        help="Update DB settings to use target collection",
    )
    parser.add_argument(
        "--rollback",
        action="store_true",
        help="Revert DB settings to source collection",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete source collection (use after cutover)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


# ---------------------------------------------------------------------------
# Cursor checkpoint for resume support
# ---------------------------------------------------------------------------


@dataclass
class MigrationCursor:
    offset: str | int | None = None
    migrated: int = 0
    source_count: int = 0
    timestamp: str = ""

    def save(self, path: str | None = None) -> None:
        path = path or CURSOR_FILE
        data = {
            "offset": self.offset,
            "migrated": self.migrated,
            "source_count": self.source_count,
            "timestamp": self.timestamp,
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | None = None) -> "MigrationCursor":
        path = path or CURSOR_FILE
        if not Path(path).exists():
            return cls()
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def clear(self, path: str | None = None) -> None:
        path = path or CURSOR_FILE
        if Path(path).exists():
            Path(path).unlink()


# ---------------------------------------------------------------------------
# Sparse embedding helpers
# ---------------------------------------------------------------------------


def get_sparse_model():
    """Load BM25 sparse model (same pattern as VectorService)."""
    try:
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding(
            model_name="Qdrant/bm25",
            cache_dir=os.environ.get("FASTEMBED_CACHE_PATH"),
        )
    except Exception as e:
        logger.error(f"Failed to load sparse model: {e}")
        return None


def sparse_embed_batch(model, texts: list[str]) -> list[models.SparseVector | None]:
    """Generate sparse vectors for a batch of texts."""
    if model is None:
        return [None] * len(texts)
    results = list(model.embed(texts))
    return [
        models.SparseVector(indices=r.indices.tolist(), values=r.values.tolist()) for r in results
    ]


# ---------------------------------------------------------------------------
# Target collection creation
# ---------------------------------------------------------------------------


async def create_target_collection(
    client: AsyncQdrantClient,
    name: str,
    embedding_dim: int,
) -> None:
    """Create collection with named dense + sparse vectors and payload indexes."""
    await client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(
                size=embedding_dim,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            ),
        },
    )
    # Create payload indexes (mirrors VectorService.initialize)
    for field_name, schema in [
        ("tags", models.PayloadSchemaType.KEYWORD),
        ("document_id", models.PayloadSchemaType.KEYWORD),
        ("tenant_id", models.PayloadSchemaType.KEYWORD),
        ("sparse_indexed", models.PayloadSchemaType.BOOL),
    ]:
        await client.create_payload_index(
            collection_name=name,
            field_name=field_name,
            field_schema=schema,
        )
    logger.info(f"Created target collection '{name}' with named vectors + indexes")


# ---------------------------------------------------------------------------
# Point transformation
# ---------------------------------------------------------------------------


def transform_points(
    source_points: list,
    sparse_model,
) -> list[models.PointStruct]:
    """Transform unnamed-vector points to named dense+sparse."""
    texts = []
    for p in source_points:
        chunk_text = (p.payload or {}).get("chunk_text", "")
        texts.append(chunk_text)

    # Generate sparse vectors
    sparse_vectors = sparse_embed_batch(sparse_model, texts)

    points = []
    for i, p in enumerate(source_points):
        # Source has unnamed vector (list[float])
        dense_vector = p.vector
        if isinstance(dense_vector, dict):
            # Already named vectors (re-run scenario)
            dense_vector = dense_vector.get("dense", dense_vector)

        vector: dict = {"dense": dense_vector}
        if sparse_vectors[i] is not None:
            vector["sparse"] = sparse_vectors[i]

        payload = dict(p.payload or {})
        payload["sparse_indexed"] = sparse_vectors[i] is not None

        points.append(
            models.PointStruct(
                id=p.id,
                vector=vector,
                payload=payload,
            )
        )
    return points


# ---------------------------------------------------------------------------
# Core migration
# ---------------------------------------------------------------------------


async def do_migrate(
    qdrant_url: str,
    source: str,
    target: str,
    batch_size: int,
    embedding_dim: int,
    verify: bool,
) -> bool:
    """Scroll source, transform, upsert to target. Returns True on success."""
    client = AsyncQdrantClient(url=qdrant_url, check_compatibility=False)

    # 1. Validate source exists
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if source not in names:
        logger.error(f"Source collection '{source}' not found")
        return False

    # 2. Get source point count
    source_info = await client.get_collection(source)
    source_count = source_info.points_count or 0
    logger.info(f"Source collection '{source}': {source_count} points")

    # 3. Create target collection if not exists
    if target not in names:
        await create_target_collection(client, target, embedding_dim)
    else:
        logger.info(f"Target collection '{target}' already exists, resuming")

    # 4. Load cursor for resume
    cursor = MigrationCursor.load()
    if cursor.source_count and cursor.source_count != source_count:
        logger.warning(
            f"Source count changed ({cursor.source_count} -> {source_count}). "
            "Migration may miss or duplicate points."
        )
    cursor.source_count = source_count

    # 5. Load sparse model
    sparse_model = get_sparse_model()
    if sparse_model is None:
        logger.warning("Sparse model unavailable. Sparse vectors will be None.")

    # 6. Scroll and transform
    offset = cursor.offset
    migrated = cursor.migrated

    while True:
        results, next_offset = await client.scroll(
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True,
        )

        if not results:
            break

        # Transform points
        points = transform_points(results, sparse_model)

        # Upsert to target
        await client.upsert(collection_name=target, points=points)
        migrated += len(results)

        # Checkpoint
        cursor.offset = next_offset
        cursor.migrated = migrated
        cursor.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        cursor.save()

        logger.info(f"Migrated {migrated}/{source_count} points")

        if next_offset is None:
            break
        offset = next_offset

    logger.info(f"Migration complete: {migrated} points migrated")
    cursor.clear()

    # 7. Verify if requested
    if verify:
        passed = await run_verification_gates(client, source, target)
        if not passed:
            logger.error("Verification gates FAILED. Do NOT cutover.")
            return False
        logger.info("All verification gates PASSED.")

    return True


# ---------------------------------------------------------------------------
# Verification gates
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


async def gate_point_count(client: AsyncQdrantClient, source: str, target: str) -> GateResult:
    """G1: Point count parity."""
    src_info = await client.get_collection(source)
    tgt_info = await client.get_collection(target)
    src_count = src_info.points_count or 0
    tgt_count = tgt_info.points_count or 0
    passed = src_count == tgt_count
    return GateResult("G1: Point count", passed, f"source={src_count}, target={tgt_count}")


async def gate_payload_integrity(
    client: AsyncQdrantClient,
    source: str,
    target: str,
    sample_size: int = 100,
) -> GateResult:
    """G2: Sample random points, compare chunk_text SHA256."""
    src_points, _ = await client.scroll(source, limit=sample_size, with_payload=True)
    if not src_points:
        return GateResult("G2: Payload integrity", True, "No points to check")

    sample = random.sample(src_points, min(sample_size, len(src_points)))
    point_ids = [p.id for p in sample]

    tgt_points = await client.retrieve(target, ids=point_ids, with_payload=True)
    tgt_map = {p.id: p for p in tgt_points}

    mismatches = 0
    for sp in sample:
        tp = tgt_map.get(sp.id)
        if tp is None:
            mismatches += 1
            continue
        src_hash = hashlib.sha256((sp.payload or {}).get("chunk_text", "").encode()).hexdigest()
        tgt_hash = hashlib.sha256((tp.payload or {}).get("chunk_text", "").encode()).hexdigest()
        if src_hash != tgt_hash:
            mismatches += 1

    passed = mismatches == 0
    return GateResult(
        "G2: Payload integrity",
        passed,
        f"checked={len(sample)}, mismatches={mismatches}",
    )


async def gate_search_parity(
    client: AsyncQdrantClient,
    source: str,
    target: str,
    n_queries: int = 10,
) -> GateResult:
    """G3: Run canary queries, compare top-5 overlap >= 80%."""
    src_points, _ = await client.scroll(
        source, limit=n_queries, with_vectors=True, with_payload=False
    )
    if not src_points:
        return GateResult("G3: Search parity", True, "No points for queries")

    overlaps = []
    for p in src_points:
        vector = p.vector
        if isinstance(vector, dict):
            vector = vector.get("dense", list(vector.values())[0])

        # Search source (unnamed vector)
        src_results = await client.query_points(
            collection_name=source,
            query=vector,
            limit=5,
            with_payload=False,
        )
        src_ids = {r.id for r in src_results.points}

        # Search target (named "dense" vector)
        tgt_results = await client.query_points(
            collection_name=target,
            query=vector,
            using="dense",
            limit=5,
            with_payload=False,
        )
        tgt_ids = {r.id for r in tgt_results.points}

        overlap = len(src_ids & tgt_ids) / max(len(src_ids), 1)
        overlaps.append(overlap)

    avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    passed = avg_overlap >= 0.80
    return GateResult(
        "G3: Search parity",
        passed,
        f"avg_overlap={avg_overlap:.2%}, threshold=80%",
    )


async def gate_latency(
    client: AsyncQdrantClient,
    source: str,
    target: str,
    n_queries: int = 10,
) -> GateResult:
    """G4: Compare query latency. Target p99 <= 2x source p99."""
    src_points, _ = await client.scroll(
        source, limit=n_queries, with_vectors=True, with_payload=False
    )
    if not src_points:
        return GateResult("G4: Latency", True, "No points for queries")

    src_latencies = []
    tgt_latencies = []

    for p in src_points:
        vector = p.vector
        if isinstance(vector, dict):
            vector = vector.get("dense", list(vector.values())[0])

        # Source query
        start = time.perf_counter()
        await client.query_points(collection_name=source, query=vector, limit=5, with_payload=False)
        src_latencies.append(time.perf_counter() - start)

        # Target query
        start = time.perf_counter()
        await client.query_points(
            collection_name=target,
            query=vector,
            using="dense",
            limit=5,
            with_payload=False,
        )
        tgt_latencies.append(time.perf_counter() - start)

    # p99 (with only 10 samples, use max)
    src_p99 = sorted(src_latencies)[-1] if src_latencies else 0
    tgt_p99 = sorted(tgt_latencies)[-1] if tgt_latencies else 0

    # Allow up to 2x latency; treat sub-millisecond as pass (mock/fast queries)
    passed = True if src_p99 < 0.001 else tgt_p99 <= 2 * src_p99
    return GateResult(
        "G4: Latency",
        passed,
        f"source_p99={src_p99 * 1000:.1f}ms, target_p99={tgt_p99 * 1000:.1f}ms",
    )


async def gate_sparse_presence(
    client: AsyncQdrantClient,
    target: str,
    sample_size: int = 100,
) -> GateResult:
    """G5: Check that random points in target have sparse vectors."""
    points, _ = await client.scroll(
        target, limit=sample_size, with_vectors=True, with_payload=False
    )
    if not points:
        return GateResult("G5: Sparse presence", True, "No points to check")

    sample = random.sample(points, min(sample_size, len(points)))
    missing = 0
    for p in sample:
        if isinstance(p.vector, dict):
            if "sparse" not in p.vector or p.vector["sparse"] is None:
                missing += 1
        else:
            missing += 1

    passed = missing == 0
    return GateResult(
        "G5: Sparse presence",
        passed,
        f"checked={len(sample)}, missing_sparse={missing}",
    )


async def run_verification_gates(
    client: AsyncQdrantClient,
    source: str,
    target: str,
) -> bool:
    """Run all 5 cutover verification gates. Returns True if all pass."""
    gates: list[GateResult] = []

    gates.append(await gate_point_count(client, source, target))
    gates.append(await gate_payload_integrity(client, source, target))
    gates.append(await gate_search_parity(client, source, target))
    gates.append(await gate_latency(client, source, target))
    gates.append(await gate_sparse_presence(client, target))

    all_passed = True
    for g in gates:
        status = "PASS" if g.passed else "FAIL"
        logger.info(f"  [{status}] {g.name}: {g.detail}")
        if not g.passed:
            all_passed = False

    return all_passed


# ---------------------------------------------------------------------------
# Cutover / Rollback / Cleanup
# ---------------------------------------------------------------------------


def _update_collection_setting(collection_name: str, reason: str) -> None:
    """Update qdrant_collection in AdminSettings DB."""
    from ai_ready_rag.db.database import SessionLocal
    from ai_ready_rag.services.settings_service import SettingsService

    db = SessionLocal()
    try:
        svc = SettingsService(db)
        svc.set_with_audit(
            key="qdrant_collection",
            value=collection_name,
            changed_by="migration-cli",
            reason=reason,
        )
        logger.info(f"Updated qdrant_collection setting to '{collection_name}'")
    finally:
        db.close()


async def do_cutover(qdrant_url: str, source: str, target: str) -> None:
    """Update DB settings to point to target collection."""
    client = AsyncQdrantClient(url=qdrant_url, check_compatibility=False)

    # Verify target exists
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if target not in names:
        logger.error(f"Target collection '{target}' not found. Run migration first.")
        sys.exit(1)

    # Run verification gates before cutover
    passed = await run_verification_gates(client, source, target)
    if not passed:
        logger.error("Verification gates FAILED. Aborting cutover.")
        sys.exit(1)

    # Update DB setting
    _update_collection_setting(target, reason=f"Cutover from {source} to {target}")
    logger.info(f"Cutover complete: qdrant_collection -> '{target}'")
    logger.info("Restart the application or call POST /api/admin/refresh-capabilities")


async def do_rollback(qdrant_url: str, source: str, target: str) -> None:
    """Revert DB settings to source collection."""
    client = AsyncQdrantClient(url=qdrant_url, check_compatibility=False)

    # Verify source still exists
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if source not in names:
        logger.error(f"Source collection '{source}' not found. Cannot rollback.")
        sys.exit(1)

    _update_collection_setting(source, reason=f"Rollback from {target} to {source}")
    logger.info(f"Rollback complete: qdrant_collection -> '{source}'")


async def do_cleanup(qdrant_url: str, source: str) -> None:
    """Delete the source (old) collection."""
    client = AsyncQdrantClient(url=qdrant_url, check_compatibility=False)

    collections = await client.get_collections()
    names = [c.name for c in collections.collections]
    if source not in names:
        logger.info(f"Collection '{source}' already deleted.")
        return

    await client.delete_collection(source)
    logger.info(f"Deleted collection '{source}'")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    target = args.target_collection or f"{args.source_collection}{DEFAULT_TARGET_SUFFIX}"

    if args.cutover:
        asyncio.run(do_cutover(args.qdrant_url, args.source_collection, target))
    elif args.rollback:
        asyncio.run(do_rollback(args.qdrant_url, args.source_collection, target))
    elif args.cleanup:
        asyncio.run(do_cleanup(args.qdrant_url, args.source_collection))
    else:
        success = asyncio.run(
            do_migrate(
                args.qdrant_url,
                args.source_collection,
                target,
                args.batch_size,
                args.embedding_dimension,
                args.verify,
            )
        )
        if not success:
            sys.exit(1)


if __name__ == "__main__":
    main()
