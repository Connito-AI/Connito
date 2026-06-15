"""Build the per-shard row-count table for C4-en (json.gz).

C4's json.gz files have no footer, so the row count for each shard
must be enumerated. This script downloads each shard and counts lines
in the decompressed JSON-lines stream. Writes the result to:

    connito/shared/data/c4_en_shard_rows.json

Run once at dataset-revision change. Output is consensus-load-bearing
— every validator must use the IDENTICAL table, so the file is
checked into the repo (not regenerated per-validator). Re-run only
when the dataset publisher actually re-uploads C4.

Time: ~hours, depends on bandwidth. Parallelism is intentionally
limited (DEFAULT_WORKERS) to keep HF rate limits happy.

Memory: streaming gunzip + line count, ~constant per shard.

Usage:
    python -m scripts.build_c4_en_shard_row_counts
    python -m scripts.build_c4_en_shard_row_counts --revision <sha>
    python -m scripts.build_c4_en_shard_row_counts --shards 0-9  # debug subset
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "allenai/c4"
CONFIG = "en"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "connito" / "shared" / "data" / "c4_en_shard_rows.json"
DEFAULT_WORKERS = 4


def _list_shards(revision: str) -> list[str]:
    info = HfApi().dataset_info(REPO_ID, revision=revision)
    shards = [
        f.rfilename for f in info.siblings
        if f.rfilename.startswith("en/")
        and f.rfilename.endswith(".json.gz")
        and "train" in f.rfilename.split("/")[-1].lower()
    ]
    return sorted(shards)


def _count_rows(repo_id: str, revision: str, shard_path: str) -> tuple[str, int]:
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{shard_path}"
    req = urllib.request.Request(url, headers={"User-Agent": "connito-c4-shard-counter/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        compressed = r.read()
    count = 0
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as gz:
        for _ in gz:
            count += 1
    return shard_path, count


def _parse_shard_range(spec: str, total: int) -> list[int]:
    if "-" in spec:
        lo, hi = spec.split("-", 1)
        return list(range(int(lo), int(hi) + 1))
    return [int(spec)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--revision", default="main", help="HF dataset revision (commit SHA preferred)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--shards", default=None,
                    help="Optional shard index subset for debug runs, e.g. `0-9` or `0`")
    ap.add_argument("--out", default=str(OUTPUT_PATH))
    args = ap.parse_args()

    print(f"Listing shards for {REPO_ID} @ {args.revision} ...", file=sys.stderr)
    shards = _list_shards(args.revision)
    print(f"  found {len(shards)} train shards", file=sys.stderr)

    if args.shards:
        idxs = _parse_shard_range(args.shards, len(shards))
        shards = [shards[i] for i in idxs]
        print(f"  subset: {len(shards)} shards ({idxs[:3]}..{idxs[-1:]})", file=sys.stderr)

    # Resume support — keep already-counted entries if the output file
    # exists. Lets long runs survive a crash without redoing work.
    existing: dict[str, int] = {}
    if Path(args.out).exists():
        try:
            existing = json.loads(Path(args.out).read_text())
            print(f"  resuming from {len(existing)} prior entries", file=sys.stderr)
        except Exception:
            existing = {}
    todo = [s for s in shards if s not in existing]
    print(f"  {len(todo)} shards to count", file=sys.stderr)

    if not todo:
        print("Nothing to do.", file=sys.stderr)
        return 0

    results = dict(existing)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_count_rows, REPO_ID, args.revision, s): s for s in todo}
        completed = 0
        for fut in as_completed(futures):
            shard = futures[fut]
            try:
                _, count = fut.result()
                results[shard] = count
                completed += 1
                if completed % 16 == 0 or completed == len(todo):
                    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True))
                    print(
                        f"  [{completed}/{len(todo)}] {shard} -> {count}",
                        file=sys.stderr,
                    )
            except Exception as e:
                print(f"  FAILED {shard}: {e}", file=sys.stderr)

    Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"Wrote {len(results)} entries to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
