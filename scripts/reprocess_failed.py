#!/usr/bin/env python3
"""Reprocess all failed PDF documents on a VE-RAG-System server.

Connects via REST API, triggers reprocess for each failed document,
then polls until all are done (or timeout).

Usage:
    python scripts/reprocess_failed.py                          # default: Spark
    python scripts/reprocess_failed.py --host localhost          # local dev
    python scripts/reprocess_failed.py --include-pending         # also wait for pending docs
"""

import argparse
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description="Reprocess failed documents (remote)")
    parser.add_argument("--host", default="172.16.20.51", help="Server host (default: Spark)")
    parser.add_argument("--port", default="8502", help="Server port (default: 8502)")
    parser.add_argument("--email", default="admin@test.com", help="Login email")
    parser.add_argument("--password", default="npassword2002!", help="Login password")
    parser.add_argument(
        "--include-pending", action="store_true", help="Also wait for pending docs to finish"
    )
    parser.add_argument(
        "--timeout", type=int, default=900, help="Max wait time in seconds (default: 900)"
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Login
    print(f"Connecting to {base_url}...", end=" ", flush=True)
    r = httpx.post(
        f"{base_url}/api/auth/login", json={"email": args.email, "password": args.password}
    )
    r.raise_for_status()
    token = r.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    print("OK")

    # Get all documents
    r = httpx.get(f"{base_url}/api/documents?per_page=200", headers=headers)
    r.raise_for_status()
    docs = r.json()["documents"]

    failed = [d for d in docs if d["status"] == "failed"]
    pending = [d for d in docs if d["status"] == "pending"]
    ready = [d for d in docs if d["status"] == "ready"]
    processing = [d for d in docs if d["status"] == "processing"]

    print(
        f"\nCurrent status: ready={len(ready)} failed={len(failed)} pending={len(pending)} processing={len(processing)}"
    )

    if not failed:
        print("No failed documents to reprocess.")
        return

    # Reprocess each failed doc
    print(f"\nReprocessing {len(failed)} failed documents...")
    for d in failed:
        r = httpx.post(
            f"{base_url}/api/documents/{d['id']}/reprocess",
            headers=headers,
            timeout=60,
        )
        status = "queued" if r.status_code == 202 else f"HTTP {r.status_code}"
        print(f"  {d['original_filename']}: {status}")

    # Poll until all done
    wait_count = len(failed) + (len(pending) if args.include_pending else 0)
    print(f"\nWaiting for {wait_count} documents to process (timeout: {args.timeout}s)...")

    poll_interval = 15
    for i in range(args.timeout // poll_interval):
        time.sleep(poll_interval)
        r = httpx.get(f"{base_url}/api/documents?per_page=200", headers=headers)
        r.raise_for_status()
        docs = r.json()["documents"]

        counts = {}
        for d in docs:
            counts[d["status"]] = counts.get(d["status"], 0) + 1

        elapsed = (i + 1) * poll_interval
        parts = [f"{s}={c}" for s, c in sorted(counts.items())]
        print(f"  {elapsed:>4}s: {', '.join(parts)}")

        if counts.get("processing", 0) == 0 and counts.get("pending", 0) == 0:
            print("\nAll done!")
            break
    else:
        print(f"\nTimeout after {args.timeout}s")

    # Final summary
    r = httpx.get(f"{base_url}/api/documents?per_page=200", headers=headers)
    r.raise_for_status()
    docs = r.json()["documents"]

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)

    successes = [d for d in docs if d["status"] == "ready"]
    failures = [d for d in docs if d["status"] == "failed"]

    print(f"  Ready:  {len(successes)}")
    print(f"  Failed: {len(failures)}")

    if failures:
        print("\nFailed documents:")
        for d in failures:
            err = (d.get("error_message") or "")[:120]
            print(f"  - {d['original_filename']}: {err}")


if __name__ == "__main__":
    main()
