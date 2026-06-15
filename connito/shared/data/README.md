# Static data tables consumed by `connito.shared`

Files here are loaded via `importlib.resources` and treated as
consensus-load-bearing — every validator must read the IDENTICAL
contents, so any change here must be coordinated network-wide and
gated on a chain epoch.

## `c4_en_shard_rows.json`

Maps `allenai/c4` (`en` config) parquet/json.gz shard paths to row
counts. Required when `DataCfg.eval_source_seeded_shard_pick=True`
because gzip files have no footer and the in-shard offset bound
(`hash % rows_in_this_shard`) needs the actual per-shard count.

**Bootstrap state** — only 4 entries (shards 0, 1, 500, 1023) are
committed. The remaining 1020 must be populated before flipping
`eval_source_seeded_shard_pick=True` for production:

    python -m scripts.build_c4_en_shard_row_counts

Re-run only when `allenai/c4` is re-uploaded (rare). The
`eval_source_revision_pin["allenai/c4"]` SHA in `DataCfg` should be
bumped together with regenerating this table — never one without
the other.
