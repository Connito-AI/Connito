# Dead-code removal

An evidence-based inventory of unreachable code in this repository, produced
before any deletion. Findings are graded by confidence: §1 is what the
accompanying cleanup removes, §2 is what needs a human decision first, and §3
records things that *look* dead but are not — so the next sweep doesn't
re-litigate them.

**Baseline:** `origin/staging` @ `6d7b58e`.

## Relationship to the earlier catalogue

This is not the first pass over this ground. Commit `65a2c3b` on
`chore/comment-cleanup` ("catalogue dead code without removing it") carries a
`DEAD_CODE_ANALYSIS.md` that surveyed the same tree and **deliberately deleted
nothing** — its own words: *"the dead code catalogued below was found and is
reported here, but deliberately left in place at the repo owner's direction."*
That branch is merged to neither `master` nor `staging`.

This document is deliberately named differently so the two do not collide, and
supersedes it only in the sense that the current instruction is to remove
rather than catalogue. **If that earlier direction still stands, this PR should
not merge** — the decision is the owner's, not this document's.

The two analyses were produced independently and agree. Everything the earlier
catalogue listed as callerless — `_cuda_mem_report`, `PhaseResponseLite`,
`search_model_submission_destination`, `SignedModelSubmitMessage`,
`_normalize_hash`/`_hash_bytes`, `hex_to_byte`, `names_for_expert`/
`iter_named_grads`, and 42 unused imports — is confirmed here, and both passes
independently ruled `validate_miner_chain_commit_payload` *out* of scope as
documented back-compat. Where the earlier catalogue was sharper, its finding is
credited inline (see §1.3).

## How this was produced

Text search alone produces both false positives and false negatives here, so
three independent passes were used and cross-checked:

1. **Import graph.** Every `.py` file parsed with `ast`, module-level imports
   resolved to file paths, then forward-reachability computed separately from
   the production entry points (`validator/run.py`, `miner/train.py`,
   `miner/model_io.py`, `sn_owner/dht_init.py`, `sn_owner/phase_service.py`,
   `shared/config.py`), from the tools, and from the test suite.
2. **AST symbol usage.** Every top-level function / class / constant checked
   against real `Name` loads, `Attribute` accesses, `ImportFrom` aliases and
   *string literals* across every file. Docstrings and comments deliberately do
   **not** count as usage — this is what separates a live symbol from one that
   only survives in prose. String literals **do** count, so anything reachable
   through `import_from_string`, `getattr` or a config value stays live.
3. **Full-text sweep** over `.py`, `.yaml`, `.json`, `.md`, `.ipynb`,
   `.toml`, `.txt` and the Dockerfile, to catch references from configuration,
   dashboards, compose files and documentation.

Dynamic-reference mechanisms present in this repo, and how each was handled:

| Mechanism | Where | Handling |
|---|---|---|
| `import_from_string` on a config string | `dataloader.py:544` ← `expert_groups/*/config.yaml:dataset_class` | String literals counted as usage; the referenced classes are in §3 |
| FastAPI route registration | `sn_owner/phase_service.py` `@app.get(...)` | Decorated handlers treated as entry points (§3) |
| `sys.path` + flat-module import | `tools/quantization/*` `from gpu_common import ...` | Checked by hand; `gpu_common` is live (§3) |
| Backend switch constant | `mycelia.py:MODEL_BACKEND` | Read; only one branch exists, `else` raises |
| Prometheus metric names | `observability/grafana/dashboards/*.json` | Grepped by metric name, not symbol (§2) |
| `pytest` name convention | `connito/test/test_*.py` | `test_*` / `Test*` excluded from candidacy |
| Wildcard re-export | `from connito.shared.helper import *` in 3 modules | Bare-`Name` usage is counted repo-wide, so re-exported names cannot hide |

No repository-native dead-code tooling exists (`pyproject.toml` configures
`ruff`, `black` and `mypy`, but CI runs none of them — the only workflow is
`docker-publish.yml`). Nothing new was installed for the analysis.

---

## 1. Safe to remove

Strong evidence of unreachability from every entry point, tool, test,
configuration file and dynamic-dispatch mechanism in the repo.

### 1.1 Whole files

| Item | Type | Location | Evidence | Dependencies / Impact | Recommendation |
|---|---|---|---|---|---|
| `client.py` | File, 133 lines | `connito/shared/client.py` | The last remnant of the removed validator-HTTP distribution path. Unreachable in the import graph from *every* production entry point, tool and test — no module imports it, and reverse-edge lookup returns nobody. `test_remove_http_submission.py` already pins that the rest of that path (`connito.validator.server`, `submit_model`, `_download_checkpoint_from_validator_http`) is gone; this file was missed. Its only two symbols, `download_model` and `CHUNK`, have no AST references outside it. | Sole consumer of `SignedDownloadRequestMessage` and `construct_block_message` — see 1.2. Removes the last import of `requests` in `shared/` outside `cycle.py`. | Remove file |
| `custom_deepseek.py` | File, 194 lines | `connito/shared/modeling/custom_deepseek.py` | The DeepSeek-**V3** prototype, superseded by `custom_deepseek_v2_lite.py`. Backend selection in `mycelia.py` is a hardcoded module constant, `MODEL_BACKEND = "deepseek_v2"`, with exactly one `if` branch and an `else` that raises — no config key, string reference or runtime path can select V3. Never imported by anything. Unmodified since the initial commit (2026-04-07) while the V2-Lite module it parallels is edited continuously (last change 2026-08-17). | Self-contained: imports only `transformers.models.deepseek_v3` and `helper.*`. Nothing else in the tree references `TopkRouter`, `SparseMoeBlock` or `get_moe_model_config` from this module. | Remove file |

### 1.2 Symbols orphaned by 1.1

| Item | Type | Location | Evidence | Dependencies / Impact | Recommendation |
|---|---|---|---|---|---|
| `SignedDownloadRequestMessage` | Class | `connito/shared/schema.py:28` | Only AST references are in `client.py` (2), which 1.1 removes. | — | Remove |
| `construct_block_message()` | Function | `connito/shared/schema.py:49` | Same: only referenced from `client.py`. | — | Remove |
| `SignedModelSubmitMessage` | Class | `connito/shared/schema.py:33` | Zero AST references anywhere, already — the model-submit-over-HTTP path it modelled is gone. | — | Remove |
| `construct_model_message()` | Function | `connito/shared/schema.py:38` | Zero references anywhere. Also visibly abandoned: the docstring promises `model_hash \|\| construct_block_message(...)` but the body returns the hash alone, and the `expert_groups` parameter is never read. | Dropping it removes `schema.py`'s only use of `compile_full_state_dict_from_path` and `get_model_hash`, decoupling `schema` from `checkpoint_helper`. | Remove |
| `SignedMessage` | Class | `connito/shared/schema.py:13` | A base class whose only two referents are the dead subclasses above. Nothing constructs it, and `to_dict`/`from_dict` have no callers. | Becomes dead only once the three above go — a deletion chain, not an independent finding. | Remove |

What survives in `schema.py`: `sign_message`, `verify_message`, `b64url_decode_nopad` — the live signing primitives used by `checkpoints.py`, `model.py` and `inter_validator_connection.py`.

### 1.3 Duplicated implementation

| Item | Type | Location | Evidence | Dependencies / Impact | Recommendation |
|---|---|---|---|---|---|
| `_cuda_mem_report()` | Function | `connito/validator/run.py:276` | **AST-identical** to `connito/shared/memory.py:cuda_mem_report` (verified by comparing `ast.dump` of both bodies with the name normalised — exact match, 27 lines each). The shared copy is the live one, called twice inside `memory.py:cleanup()`, which `run.py` and `model.py` both use. The `run.py` copy has zero callers: it is the pre-centralisation original left behind when `shared/memory.py` was created ("Centralised so both the validator main loop and helpers in `shared` … can reclaim cached allocator memory"). **It was also broken.** Identical AST does not mean identical name resolution: the body calls `log_phase` at `run.py:294`, and `run.py` never imports it, while `memory.py` does (`memory.py:14`). The `run.py` copy would have raised `NameError` on its first call — a latent bug that only unreachability kept hidden. This last point is credited to the earlier catalogue (`65a2c3b`), which caught it first; the initial version of this document described the function as merely a duplicate. | None. `run.py` already imports `cleanup` from `shared.memory`. | Remove |

### 1.4 Orphaned `torch.distributed` expert-group facility

Four functions in `connito/shared/expert_manager.py` form a complete
multi-rank process-group + weight-sync subsystem that nothing calls. The
validator's cross-validator averaging goes through hivemind
(`inter_validator_connection.py`), and the miner's multi-rank path
(`mp.spawn` + NCCL) never builds per-expert process groups.

| Item | Type | Location | Evidence | Dependencies / Impact | Recommendation |
|---|---|---|---|---|---|
| `create_expert_groups()` | Function | `expert_manager.py:379` | Zero AST references. The only textual mentions in the repo are two lines of `train.py:setup_training`'s docstring, which document `expert_groups` and `group_ids` **return values that the function does not return** — its real signature returns seven different items. Prose, not usage. | The two stale docstring lines are removed with it. The rest of that docstring is also stale (it documents `global_model`, `outer_optimizer`, `start_step`, none of which are returned either) — flagged in §4, not fixed here. | Remove |
| `split_into_groups()` | Function | `expert_manager.py:340` | Zero references of any kind, including strings and prose. | Self-contained. | Remove |
| `sync_weights()` | Function | `expert_manager.py:480` | Zero references. Requires `dist.is_initialized()`; no caller ever supplies that context. | Uses `_named_params` / `is_expert_param`, both of which stay live via `populate_global_grads_from_local`. | Remove |
| `sync_expert_weights()` | Function | `expert_manager.py:493` | Zero references. Takes an `expert_groups: Mapping[int, ProcessGroup]` that only the (also dead) `create_expert_groups` could produce. | Same. | Remove |

### 1.5 Orphaned gradient-selection helpers

`connito/validator/inter_validator_connection.py` — a self-contained cluster.
The live packing path is `iter_named_params` → `build_grad_buff_from_model` →
`pack_grads` / `unpack_to_grads`; these four sit beside it, unused.

| Item | Type | Location | Evidence | Dependencies / Impact | Recommendation |
|---|---|---|---|---|---|
| `names_for_expert()` | Function | `inter_validator_connection.py:125` | Zero AST references. | — | Remove |
| `iter_named_grads()` | Function | `inter_validator_connection.py:145` | Zero AST references. Its live sibling `iter_named_params` is used three times and stays. | — | Remove |
| `select_tensors()` | Function | `inter_validator_connection.py:159` | Zero AST references. | Its only caller-of, `name_selected`, becomes dead — see next row. | Remove |
| `name_selected()` | Function | `inter_validator_connection.py:153` | One AST reference, from `select_tensors` alone. Dead once that goes. | Deletion chain. | Remove |

### 1.6 Individually orphaned symbols

| Item | Type | Location | Evidence | Dependencies / Impact | Recommendation |
|---|---|---|---|---|---|
| `split_validation_uids_into_foreground()` | Function | `validator/round_groups.py:639` | Self-declared: *"Deprecated: returns the flat A→B→C concatenation used in earlier drafts. New callers should use `split_foreground_background`."* Zero callers; `split_foreground_background` is the one the tests and `Round.freeze` exercise. | None. | Remove |
| `search_model_submission_destination()` | Function | `shared/cycle.py:416` | Resolves the assigned validator's **axon** so a miner can push its model over the wire — the removed HTTP submission path again. Zero callers; submission is now HF upload + chain commit (`model_io.py`). | None. `get_validator_miner_assignment`, which it calls, stays live (`round.py`). | Remove |
| `PhaseResponseLite` | Class | `shared/cycle.py:140` | A three-field slim twin of the live `PhaseResponse`. Never constructed, never annotated with, never named in a string or a response model. | None. | Remove |
| `_normalize_hash()` | Function | `shared/checkpoints.py:46` | Private, module-local, zero references in its own file or anywhere else. | — | Remove |
| `_hash_bytes()` | Function | `shared/checkpoints.py:56` | Same. | — | Remove |
| `hex_to_byte()` | Function | `shared/helper.py:235` | Zero references. A one-line `bytes.fromhex` wrapper. Re-exported by three `import *` modules, but no bare-`Name` use of it exists anywhere in the tree. | — | Remove |
| `grad_hook()` | Function | `shared/model.py:157` | Zero references. A debug hook that `print`s and raises on non-finite grads; the production guards are in `run.py:aggregate_miner_gradient_change` and are covered by `test_validator_grad_safety.py`. | — | Remove |
| `merge_state_dicts_with_priority()` | Function | `shared/modeling/mycelia.py:368` | Zero references. Merging is done by the streaming loaders (`stream_pretrained_state_dict_to_partial_model`, `stream_safetensors_to_partial_model`). | — | Remove |
| `resolve_hf_token()` | Function | `shared/hf_distribute.py:47` | Zero references. A pure passthrough to `_resolve_token(token, token_env_var)`; every real caller in the module calls `_resolve_token` directly. No `__all__`, no `__init__.py`, so it is not part of any declared package surface. | — | Remove |
| `MAX_CONCURRENT_DOWNLOADS` | Constant | `validator/evaluator.py:704` | Zero references. Concurrency is owned by `BackgroundDownloadWorker`, which does not read it. | Sibling `EVAL_MAX_BATCHES` in the same block **is** live (7 references) and stays. | Remove |
| `EVAL_WORKERS` | Constant | `validator/evaluator.py:705` | Zero references. | Same. | Remove |
| `DOWNLOAD_TIMEOUT_SEC` | Constant | `validator/evaluator.py:706` | Zero references. The live timeout is `ckpt.download_timeout_s` via config. | Same. | Remove |

### 1.7 Unused imports

Bindings never loaded anywhere in their file — verified by AST, with string
literals counted as usage so quoted annotations do not produce false hits.
`from __future__ import annotations` is a compiler directive and is excluded
throughout.

| File | Line | Import |
|---|---|---|
| `connito/shared/chain.py` | 8 | `from typing import Literal` |
| `connito/shared/checkpoints.py` | 5 | `import time` (remaining match is a comment) |
| `connito/shared/checkpoints.py` | 6 | `from collections import Counter` |
| `connito/shared/checkpoints.py` | 31 | `from connito.shared.config import MinerConfig, ValidatorConfig` |
| `connito/shared/checkpoints.py` | 37 | `from connito.shared.expert_manager import ExpertManager` |
| `connito/shared/config.py` | 5 | `import os` |
| `connito/shared/cycle.py` | 40 | `from connito.shared.chain import serve_axon` (the function itself is live — called by `setup_chain_worker`; only this import is unused) |
| `connito/shared/expert_manager.py` | 15 | `from connito.shared.helper import sum_model_gradients` |
| `connito/shared/model.py` | 5 | `import traceback` |
| `connito/shared/model.py` | 14 | `from connito.shared.chain import SignedModelHashChainCommit` |
| `connito/shared/model.py` | 19 | `from connito.shared.checkpoints import build_chain_checkpoints` |
| `connito/shared/model.py` | 32 | `from connito.shared.cycle import PhaseNames, get_blocks_from_previous_phase_from_api` |
| `connito/shared/model.py` | 46 | `from connito.shared.schema import verify_message` |
| `connito/validator/cohort_state.py` | 22 | `from dataclasses import field` |
| `connito/validator/evaluator.py` | 7 | `import os` |
| `connito/validator/inter_validator_connection.py` | 3 | `import os` |
| `connito/validator/inter_validator_connection.py` | 10 | `from typing import Set` |
| `connito/validator/inter_validator_connection.py` | 20 | `from hivemind.utils.timed_storage import TimedStorage` |
| `connito/validator/inter_validator_connection.py` | 26 | `import traceback` |
| `connito/validator/inter_validator_connection.py` | 28 | `import asyncio` |
| `connito/validator/run.py` | 198 | `from datetime import datetime` |

Removed in the same pass because the code that used them is gone (cascade, not independent findings):

| File | Import | Was used only by |
|---|---|---|
| `connito/shared/schema.py` | `asdict`, `dataclass`, `Path`, `compile_full_state_dict_from_path`, `get_model_hash` | the five symbols in §1.2 |
| `connito/shared/expert_manager.py` | `random`, `Iterable`, `torch.distributed as dist` | the four functions in §1.4 (remaining textual hits are inside a pre-existing commented-out block) |
| `connito/validator/inter_validator_connection.py` | `fnmatch` | `name_selected` (§1.5) |

Test-side unused imports, same evidence standard:

| File | Line | Import |
|---|---|---|
| `test_background_submission_validation.py` | 10, 80 | `import asyncio`, `from connito.validator.round import RosterEntry` |
| `test_combined_validator_seed.py` | 32 | `from connito.shared.cycle import PhaseNames` |
| `test_filter_checkpoints_miner_role.py` | 23 | `import pytest` |
| `test_natural_with_fallback_routing.py` | 23 | `import types` |
| `test_round_freeze_groups.py` | 14 | `import pytest` |
| `test_round_progress_publish.py` | 18 | `import threading` |
| `test_telemetry.py` | 3 | `import threading` |
| `test_validate_miner_submission.py` | 23, 25 | `from unittest.mock import patch`, `import pytest` |
| `test_validator_grad_safety.py` | 8, 10 | `import asyncio`, `import types` |

---

## 2. Probably unused — verify before removal

Unreferenced inside this repository, but each has a plausible consumer
*outside* it. None of these are touched by the accompanying cleanup.

| Item | Location | Why it looks dead | What must be verified first |
|---|---|---|---|
| `VALIDATOR_COMMIT_MAX_BYTES`, `MINER_COMMIT_MAX_BYTES`, `MINER_COMMIT_MAX_HF_REPO_ID_CHARS`, `validate_miner_chain_commit_payload()` | `shared/chain.py:33-36,190` | Zero in-repo references. | The code says to keep them: *"Back-compat aliases for callers that imported the old names"* and *"Kept so external imports don't break; new code should use `validate_chain_commit_payload` directly."* That is an explicit statement of intent about consumers this repo cannot see (miner forks pinning older imports). Note the alias set is only **half** dead — `VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS` and `validate_validator_chain_commit_payload` are both used by `run.py` — so this is not a "delete the whole back-compat block" decision. Confirm with the maintainers whether the compatibility window has closed. |
| 13 declared-but-never-updated Prometheus metrics: `DATALOADER_QUEUE_DEPTH`, `MODEL_PARAMETER_COUNT`, `VALIDATOR_ACTIVE_MINER_EVALS`, `VALIDATOR_SCORE_STD`, `VALIDATOR_EVAL_BATCH_COUNT`, `MOE_EXPERTS_ACTIVE`, `MOE_ROUTING_ENTROPY`, `MOE_EXPERT_UTILIZATION`, `CHECKPOINT_SAVE_LATENCY_SECONDS`, `CHECKPOINT_FETCH_LATENCY_SECONDS`, `CHAIN_CYCLE_LATENCY_SECONDS`, `CHAIN_WEIGHT_SET_SUCCESS`, `CHAIN_WEIGHT_SET_FAILURE` | `shared/telemetry.py` | Each is constructed once and never `.set()`/`.inc()`/`.observe()`d, so each exports a constant zero (or nothing). No in-repo reference by symbol *or* by metric-name string: `observability/grafana/dashboards/mycelia.json`, `observability/prometheus.yml` and the two `OBSERVABILITY.md` files mention none of them. | Exported metric names are an operator-facing contract. Validator operators run their own Grafana and alert rules against the `/metrics` endpoint; removing a series silently breaks a dashboard panel or an alert that this repo cannot see. Confirm with the ops owners, then remove metric and declaration together. |
| `expert_groups/exp_dummy/dataset.py` | whole file, 86 lines | Nothing imports it, and — the surprising part — `exp_dummy/config.yaml:17` sets `dataset_class: "expert_groups.exp_math.dataset:StreamingTorchDataset"`, pointing at **exp_math's** class, not its own. So even the group it belongs to does not load it. | It is the annotated "Customer Extension Point" template, and `docs/miner-faq/phases.md` and `docs/exp-legal-migration-plan.md` both describe `expert_groups/{name}/dataset.py` as the customization hook generally. Decide whether it is documentation-by-example (keep, and fix the config to point at itself) or a stale copy (remove). The config mismatch is a real bug either way. |
| `build_partial_model()` | `tools/quantization/gpu_common.py:79` | No sibling script imports it; each `from gpu_common import (...)` list was checked. | `tools/quantization/` is a documented experiment harness whose README records measured results. Confirm no unmerged/local arm depends on it before trimming the shared helper module. |
| `L40S_VRAM_GB` | `tools/quantization/gpu_option_b_budget.py:47` | Assigned, never read. | Same — it encodes a measured hardware fact (`44.4 GB usable, per the OOM report`) that may be intentional documentation. |
| `public_multiaddrs` import in `sn_owner/dht_init.py:7`, and `helper.py:public_multiaddrs` itself | `dht_init.py:7`, `helper.py` | The import is unused, and `dht_init` is the *only* referent of `helper.public_multiaddrs` — so dropping the import orphans the function too. | Deliberately not touched. `sn_owner/` is left alone in this pass: `phase_service.py` has four unused imports of its own (`AllowedHotkeyService`, `mp`, `add_init_peer_id`, `init_dht_and_peer_id`) that read like an unfinished wiring of DHT bootstrap into the phase service. Whether that wiring is abandoned or pending is a question for the owner, and it is not worth guessing at inside a cleanup PR. |
| `expert_groups/exp_math/expert_assignment.json.bk` | file | A `.bk` backup committed in the initial commit; `docs/exp-legal-migration-plan.md:335` already notes it "is in an older" format. | Confirm the newer `expert_assignment.json` fully supersedes it and no rollback procedure references the backup. |

---

## 3. Keep — indirect usage confirmed

Recorded so a future sweep does not re-flag them.

| Item | Why it looks dead | Why it is live |
|---|---|---|
| `sn_owner/phase_service.py`, `read_phase()`, `prev_phase()` | No module imports the file; neither function is called anywhere. | FastAPI route handlers, registered by the `@app.get("/get_phase")` / `@app.get("/previous_phase_blocks")` decorators and served by uvicorn. This *is* the public phase API — `docs/miner-faq/phases.md:33` names `phase_service.py:read_phase` as the source of `https://cycle-api.connito.ai/get_phase`, which every role polls in `wait_till`. **Keep.** |
| `StreamingTorchDataset` in `exp_legal/dataset.py`, `exp_math/dataset.py` | Never imported by any Python file. | Loaded by string: `expert_groups/*/config.yaml:dataset_class` → `dataloader.py:544` `import_from_string(...)`. `exp_math`'s class is loaded by two groups (`exp_math` and, via the mismatch noted in §2, `exp_dummy`). **Keep.** |
| `tools/quantization/gpu_common.py` | Package-level import-graph analysis shows nobody importing it. | All five sibling scripts do `sys.path.insert(0, Path(__file__).parent)` then `from gpu_common import ...` — a flat-module import invisible to a `connito.*`-rooted graph. `banner`, `build_config`, `host_ram_hwm_gb`, `host_ram_now_gb`, `eval_batches`, `drop_gated_sources`, `peak_vram_gb`, `reset_peak`, `vram_gb`, `EVAL_SEED` and `MODEL_PATH` are all consumed this way. **Keep** (except `build_partial_model`, §2). |
| `serve_axon()` | `shared/chain.py:448`; no test touches it. | Called at `chain.py:440` from `setup_chain_worker(serve=True)`, which every role uses. Only the *import* of it in `cycle.py:45` is unused. **Keep the function**, drop the import (§1.7). |
| `is_denied()`, `quantize_linear_modules_()`, `quantized_module_names()` | Flagged by a naive "no external mentions" pass. | Called internally at `quantization.py:331`, `:373` and `:422` respectively, on paths reached from `quantize_model_`, which `run.py` and `train.py` both import. **Keep.** |
| `validator/dedup.py` | Module name never appears as an AST symbol. | Imported by `validator/background_eval_worker.py:39` and `test_dedup_filter.py`. **Keep.** |
| `validator/full_topology_eval.py` | Same. | `FullTopologyEvalBase` imported by `run.py:126`, `test_full_topology_eval_base.py` and `tools/quantization/gpu_graft_verify.py`. **Keep.** |
| `test_eval_source_skip.py`, `test_get_base_model_partial.py` | Named `test_*` but contain no `test_*` functions, so pytest collects nothing from them. | They are standalone verification **scripts** with a `main()` and an `if __name__ == "__main__"` block, run by hand against live HuggingFace data. Not dead, just misnamed for the directory they live in. **Keep.** |
| Every `test_*` / `Test*` in `connito/test/` | No caller. | pytest collects by name convention. **Keep.** |
| `from __future__ import annotations` (≈70 files) | Bound name never loaded. | Compiler directive, not a runtime import. **Keep.** |
| `EVAL_MAX_BATCHES` | Sits in the same "Pipeline Config" block as three dead constants. | 7 live references across `evaluator.py` and `background_eval_worker.py`. **Keep.** |
| `validate_validator_chain_commit_payload()`, `VALIDATOR_COMMIT_MAX_HF_REPO_ID_CHARS` | Same back-compat block as the dead aliases in §2. | Both used by `run.py`; the latter also by `test_hf_distribution_safety.py`. **Keep.** |

---

## 4. Related defects noticed, not fixed here

Out of scope for a dead-code sweep, but worth a follow-up:

- **`train.py:setup_training`'s docstring is stale beyond the two lines this
  cleanup touches.** It documents `global_model`, `outer_optimizer`,
  `start_step`, `expert_groups` and `group_ids` as return values; the function
  returns seven entirely different items. Only the two lines naming the
  now-deleted `create_expert_groups` were removed.
- **`exp_dummy/config.yaml` points `dataset_class` at `exp_math`'s class.**
  Either the config or the file is wrong — see §2.
- **`evaluator.py:711-716` holds a commented-out `load_model_from_path`**
  directly above the live definition of the same name. Commented-out code, not
  dead code; left alone deliberately.

---

## 5. What the accompanying change actually removes

767 lines deleted, 234 added (221 of which are this document), across 26 code
files. Everything in §1, nothing in §2 or §3.

| | Count |
|---|---|
| Files deleted | 2 (`shared/client.py`, `shared/modeling/custom_deepseek.py`) |
| Functions removed | 17 |
| Classes removed | 4 |
| Constants removed | 3 |
| Unused imports removed | 21 production + 12 test + 9 cascade |
| Stale docstring lines removed | 2 (the only prose referencing a deleted symbol) |

## 6. Verification performed

**Static.** After the removals, the three analysis passes were re-run against
the modified tree:

- Every `.py` file still parses (`ast.parse`, 96 files, zero errors).
- The zero-reference symbol set dropped from 45 to 22, and those 22 are
  *exactly* the §2 deferred items (17) plus three test-internal helpers
  (`TEST_PHASE_END_BLOCK`, `_isolate_caches`, `_import_pull_helper`) — i.e.
  nothing in §1 survived, and nothing new was orphaned that isn't accounted
  for.
- The unused-import scan is clean apart from the two `sn_owner/` files that
  were deliberately skipped (§2).
- All 16 touched modules import cleanly, including the three entry points the
  test suite never imports (`validator.run`, `miner.train`, `miner.model_io`).

*Caveat on the tooling:* once this document existed in the repo, the
full-text pass began counting its own prose as "references" and stopped
flagging the §2 items. Every finding above was established before the document
was written; re-runs exclude `DEAD_CODE_ANALYSIS.md` explicitly.

**Dynamic.** `pytest connito/test` (minus the two `main()` scripts, which need
live HuggingFace access):

| | Total | Passed | Failed | Skipped |
|---|---|---|---|---|
| Before | 471 | 465 | 4 | 2 |
| After | 471 | 465 | 4 | 2 |

Identical, and the same four tests fail on both sides:
`test_reserve_owner_share_normal_case`,
`test_reserve_owner_share_owner_already_present`,
`test_checkpoint_cfg_restores_download_concurrency_for_compatibility`,
`test_checkpoint_precision_preservation`. These are **pre-existing failures in
this sandbox, not regressions** — they fail identically on unmodified
`origin/staging`.

**Limitation, stated plainly.** The sandbox has no CUDA, no C++ toolchain
(so `hivemind` cannot build from its pinned git ref) and no Python 3.10/3.12,
so the suite ran on Python 3.14 with `torch 2.13` instead of the pinned
`2.10`, `bittensor==10.5.0`, and a **minimal stub** standing in for
`hivemind`. That accounts for the four failures, and it means the run
validates *import integrity and logic* rather than the pinned dependency
matrix. The suite should be re-run on a normal dev box or CI before merge.
There is no CI test job to fall back on — `.github/workflows/docker-publish.yml`
is the only workflow, and it only builds the image.
