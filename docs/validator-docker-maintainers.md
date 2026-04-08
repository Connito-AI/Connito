# Validator container — maintainer guide

> **Audience:** repo maintainers (Connito core team).
> **Not for:** validator operators — they read
> [validator-docker-operators.md](validator-docker-operators.md).

This doc covers everything that operators **don't** see: how the image
is built, how the release pipeline works, how to test changes locally
before they hit master, how to roll back, and which knobs you can turn
without breaking running validators.

## Why only validators are containerized

Miners do not use this stack. Reasoning:

- Miners have wildly heterogeneous hardware (different GPUs, different
  CUDA versions, different kernel versions). A single container image
  is a poor fit.
- Validators are protocol-critical: a stale validator misses chain
  commits and burns its weights. Auto-update via Watchtower removes the
  human-in-the-loop failure mode.
- Validators are uniform — they all run `connito.validator.run` against
  the same config schema, against the same chain.

If we ever want a miner image, that's a separate Dockerfile under
`connito/miner/docker/`. Don't try to make one image serve both.

## File layout

```
.dockerignore                                        # repo root, applies to any docker build
.github/workflows/docker-publish.yml                 # CI: build + push to GHCR
connito/validator/docker/
  Dockerfile                                         # validator image
  docker-compose.yml                                 # operator-facing compose file
  .env.example                                       # operator-facing env template
docs/
  validator-docker-operators.md                      # operator runbook
  validator-docker-maintainers.md                    # this file
```

The build context for the Dockerfile is the **repo root**, not
`connito/validator/docker/`. The Dockerfile copies `requirements.txt`
and the entire `connito/` package, both of which live at the root. This
is why `.dockerignore` lives at the repo root.

## How the image is built

[`connito/validator/docker/Dockerfile`](../connito/validator/docker/Dockerfile):

1. Base: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. Pin via the
   `CUDA_IMAGE` build arg.
2. System deps: Python 3.10, build tools, git (needed for the VCS-installed
   `hivemind` and `unsloth` deps), curl (for the healthcheck).
3. `pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124`.
   Torch is pinned in `requirements.txt`; the extra index just makes
   sure pip picks up the cu124 binary wheel instead of the CPU one.
   **This is the layer to watch.** It's slow, fat, and the most likely
   thing to break when a dep is bumped.
4. Copy `pyproject.toml`, `README.md`, `LICENSE`, `connito/`.
5. `pip install --no-deps -e .` to install the package itself without
   re-resolving deps.
6. Create `/data/{checkpoints,expert_groups,wandb,logs}` as bind-mount
   targets.
7. `ENTRYPOINT ["python", "-m", "connito.validator.run"]`,
   `CMD ["--path", "/data/configs/validator.yaml"]`.

### Layer caching

The dep-install layer is keyed only on `requirements.txt`. As long as
that file doesn't change, every build reuses the cached pip layer
(~10-12 GB) and only re-runs the source-copy + `pip install -e .` steps.
Expect ~3-5 min for source-only changes, ~30+ min for a cold rebuild.

CI uses the `type=gha` Buildx cache backend so this works across
workflow runs too — see the `cache-from` / `cache-to` lines in the
workflow.

## Release pipeline

[`.github/workflows/docker-publish.yml`](../.github/workflows/docker-publish.yml).

### Triggers

| Trigger | Tags pushed |
|---------|-------------|
| Push to `master` | `latest`, `master-<short-sha>` |
| Push of `vX.Y.Z` tag | `vX.Y.Z` |
| Manual `workflow_dispatch` | Whatever the source ref produces |

The workflow has a `paths:` filter so it only runs when something that
actually ends up in the image changes (validator code, shared code,
sn_owner code, requirements, the Dockerfile itself). Pure
miner/docs/notebook edits skip the build entirely. **If you change a
file that should trigger the build but isn't in the list, add it to
`paths:` — don't disable the filter.**

### Tagging strategy

Operators run `:latest` by default. This means **every push to master is
a production release within 5 minutes**. If you want a staging gate:

1. Change the workflow trigger to `tags: ["v*.*.*"]` only.
2. Have operators set `IMAGE=ghcr.io/connito-ai/connito-validator:stable`
   in their `.env`.
3. Cut releases by tagging `vX.Y.Z` and re-tagging the resulting image
   as `:stable` (manually or via a second workflow).

We currently do not do this — `master` is the release channel. Treat it
accordingly.

### GitHub repo setup (before the first build)

These settings have to be in place **before** the workflow can run even
once. If you're inheriting an existing repo, check them; if you're
setting this up from scratch, do them now.

1. **Enable Actions for the repo.**
   GitHub → repo → **Settings** → **Actions** → **General** → under
   *Actions permissions* select **Allow all actions and reusable
   workflows** (or "Allow OWNER, and select non-OWNER actions" — at a
   minimum the official `docker/*` and `actions/*` actions need to be
   allowed).

2. **Grant the workflow permission to write packages.**
   Same page → scroll to *Workflow permissions* → choose **Read and
   write permissions**, and tick **Allow GitHub Actions to create and
   approve pull requests** if you also want bot-driven updates.

   Strictly speaking, the workflow declares its own per-job
   `permissions:` block:

   ```yaml
   permissions:
     contents: read
     packages: write
     id-token: write
   ```

   …so this works even if the repo default is "Read repository
   contents and packages permissions" — the per-job block escalates.
   But the org-level setting can still **deny** what the workflow asks
   for, so confirm both.

3. **No PAT needed.** The workflow authenticates to GHCR with the
   built-in `secrets.GITHUB_TOKEN` (see the `docker/login-action` step).
   Don't add a `GHCR_TOKEN` secret — it's not required and it's one
   more thing to rotate.

4. **First push creates the package automatically.** You do **not** need
   to pre-create `ghcr.io/connito-ai/connito-validator` in the GHCR UI.
   The first successful run of the workflow will create it. The
   visibility of the new package is **inherited from the repo**:
   - Public repo → public package (operators don't need to log in).
   - Private repo → private package (operators must `docker login ghcr.io`).

5. **Org-level package creation policy.** If the repo lives under a
   GitHub **organization** (not a personal account), and the org has
   restricted package creation, the first workflow run will fail with
   `denied: installation not allowed to Create organization package`.
   Fix: org → **Settings** → **Packages** → enable
   *Container* package creation for this repo (or for all repos).
   Personal-account repos don't have this restriction.

6. **Branch protection on `master` (recommended, not required).** Because
   every push to `master` ships to every operator within ~5 min, you
   probably want at minimum a "require PR + 1 review" rule and "require
   `docker-publish` to pass" before merging. Otherwise a force-push or a
   `git commit -am && git push` straight to master is a production
   deploy with no review.

### GHCR package settings (one-time, after the first successful build)

After the workflow has run for the first time and created the package:

1. GitHub → org or user → **Packages** → `connito-validator`.
2. **Package settings** → **Danger Zone** → **Change visibility** → set
   to **Public** (recommended for SN102, since operators run on
   arbitrary machines and managing per-host PATs is painful) or keep
   **Private** if you want to restrict who can run the validator.
3. **Package settings** → **Manage Actions access** → click **Add
   repository** and add this repo. This is what lets *future* workflow
   runs from this repo continue pushing to the same package after the
   initial creation. Without this step, subsequent pushes fail with
   `denied: permission_denied`.
4. (Private packages only) **Package settings** → **Manage access** →
   add any operator GitHub accounts / teams that need pull access, and
   instruct them to `docker login ghcr.io` on their host as documented
   in [validator-docker-operators.md](validator-docker-operators.md).

### Verifying the pipeline end-to-end

After the GitHub setup above, the smoke test is:

1. Make a trivial change under `connito/validator/` (e.g. add a comment
   to `run.py`) on a feature branch.
2. Open a PR, merge to `master`.
3. Watch the workflow at GitHub → **Actions** → **Build & publish
   validator image**. First run will be slow (~30+ min cold cache);
   subsequent runs are 3-5 min.
4. When green: GitHub → **Packages** → `connito-validator` → confirm
   the new tags `latest`, `master-<sha>` exist and the digests match.
5. From any host with docker installed:
   ```bash
   docker pull ghcr.io/connito-ai/connito-validator:latest
   ```
   should succeed (with `docker login` if private).
6. Bonus: spin up the operator stack on a staging host and confirm
   Watchtower picks up the new digest within `WATCHTOWER_POLL_INTERVAL`
   seconds.

If any of those steps fail, fix it **before** announcing this to
operators — the pipeline working end-to-end on day one is the whole
point of this setup.

## Local dev loop

```bash
# From the repo root:
docker build -f connito/validator/docker/Dockerfile -t connito-validator:dev .

# Or via compose, which uses the same Dockerfile:
cd connito/validator/docker
docker compose build

# Smoke test (CPU-only, no real wallet — just confirm the entrypoint loads):
docker run --rm --entrypoint python connito-validator:dev -c \
  "import connito.validator.run; print('ok')"

# Full local run with a GPU and a fake config:
docker run --rm --gpus all \
  -v $HOME/.bittensor/wallets:/root/.bittensor/wallets:ro \
  -v $PWD/scratch:/data \
  connito-validator:dev --path /data/configs/validator.yaml
```

For iterative dev where you don't want to wait for `docker build` on
every change, mount the source over the installed package:

```bash
docker run --rm --gpus all \
  -v $HOME/.bittensor/wallets:/root/.bittensor/wallets:ro \
  -v $PWD/scratch:/data \
  -v $PWD/connito:/app/connito \
  connito-validator:dev --path /data/configs/validator.yaml
```

## What to test before merging to master

Anything that touches the image is effectively a deploy. Minimum bar:

1. **Build succeeds locally.** `docker build -f connito/validator/docker/Dockerfile .`
   This catches dep resolution failures (the cu124 wheel availability
   thing — see "Known landmines" below).
2. **Entrypoint imports cleanly.** Run the smoke test from the dev loop
   above. Catches missing `__init__.py` exports, circular imports
   introduced by the change, etc.
3. **One full validation cycle on a staging hotkey.** No way around
   this. Start the container against a registered staging wallet, watch
   it complete at least one cycle (`(0)` through `(11)` in
   `connito/validator/run.py`), and confirm the chain commit lands.
4. **Graceful shutdown.** `docker stop` (sends `SIGTERM`) should let the
   `KeyboardInterrupt` handler in `run.py` shut the averagers and
   telemetry poller down within ~10s. If it hangs and gets `SIGKILL`'d,
   investigate before merging — Watchtower will hit the same path on
   every upgrade.

For dep bumps specifically: also do a **cold cache rebuild** locally
(`--no-cache`) so you actually exercise the pip resolver, not the
cached layer.

## Rolling back

Two paths, depending on how bad it is:

### "Soft" rollback — push a fix forward

Revert the offending commit on master, let the workflow rebuild
`:latest`, Watchtower picks it up within 5 min. This is the default. It
works because `:latest` is just whatever the most recent green build
produced.

### "Hard" rollback — pin all hosts to a known-good sha

If the bad image is *actively breaking* validators (e.g. crashloop on
startup), Watchtower will keep restarting them but they'll never come
up. Operators need to:

1. Find the last known-good `master-<sha>` tag in the GHCR UI.
2. Set `IMAGE=ghcr.io/connito-ai/connito-validator:master-<sha>` in
   their `.env`.
3. `docker compose up -d`.

You should **post the known-good sha in the operators channel** the
moment you know there's a regression — don't make 30 operators each go
hunting for it.

Then push the revert through the normal pipeline so `:latest` becomes
healthy again, and operators can drop their `IMAGE` pin at their leisure.

## Knobs you can turn safely

These don't break operators:

- Adding new env vars to `docker-compose.yml` with sensible defaults
  (`${FOO:-default}`).
- Adding new `EXPOSE` lines to the Dockerfile (host networking ignores
  them anyway).
- Changing the base image **patch version** (`12.4.1` → `12.4.2`).
- Adding new entries to `.dockerignore`.

## Knobs that are breaking changes (operators must act)

These need a heads-up in the operator channel:

- **Bumping `CUDA_IMAGE` major/minor** (`12.4` → `12.6`). Requires
  operators to upgrade their NVIDIA driver. Coordinate.
- **Changing the path the validator writes checkpoints to.** Operators
  have `validator.yaml` files pointing at `/data/...` paths. If you
  rename `/data/checkpoints` to `/data/state/checkpoints`, every
  operator's existing checkpoints become orphaned.
- **Renaming env vars in `docker-compose.yml`.** Operators have `.env`
  files with the old names; they'll silently fall back to defaults and
  probably do the wrong thing.
- **Changing `network_mode`.** Going from `host` to bridge networking
  forces every operator to redo their port forwarding.
- **Removing the `com.centurylinklabs.watchtower.enable` label** —
  Watchtower will stop upgrading the container.
- **Bumping torch / hivemind / bittensor major versions.** Always do
  these on a tagged release, never as a silent `master` push, and post
  in the operators channel.

When in doubt: announce it, then push.

## Known landmines

- **`torch==2.10.0` + cu124.** The pin in `requirements.txt` only works
  if PyTorch has actually published a `cu124` wheel for that version on
  the day you build. If it hasn't, the build fails at the pip install
  step. Fixes:
  - Bump `torch` to a version that does have a cu124 wheel.
  - Or change `--extra-index-url` in the Dockerfile to a different
    `cu1xx` index.
  - Always do a `--no-cache` local build when bumping torch.
- **Hivemind installs from a git SHA** (`@4bd43b77`). If that ref
  disappears (force push, repo rename) the build breaks for everyone.
  Mirror it under our org if it ever becomes load-bearing.
- **Unsloth installs from `main`.** Same risk, but only on `linux` per
  the requirements.txt marker. A breaking unsloth change can land in
  master with zero warning. Consider pinning to a SHA before the next
  validator-critical release.
- **`shm_size: 16gb` in `docker-compose.yml`.** PyTorch DataLoader
  workers need a lot of `/dev/shm`. If you drop this, validators on
  smaller hosts will OOM mysteriously inside the dataloader.
- **`network_mode: host` only works on Linux.** macOS and Windows
  Docker Desktop ignore it. Validators are Linux-only anyway, but be
  aware when testing locally on a Mac.

## Observability of the pipeline itself

- **GitHub Actions** is the source of truth for build status. Set up
  branch protection so a failing `docker-publish` blocks merges if you
  want; right now it doesn't.
- **GHCR** shows the published tags and pull counts. You can see how
  many distinct hosts pulled `:latest` in the last day, which is a
  rough proxy for "how many validators run our image."
- **Watchtower notifications** (Slack) on the operator side will tell
  you when an upgrade actually rolls out — but you only see your own,
  not the fleet. If you want fleet-wide visibility, ask operators to
  scrape `/metrics` and ship to a central Prometheus, then alert on
  `up{job="connito-validator"}` going to 0.

## When to bump versions

- **Patch** (`v1.2.3` → `v1.2.4`): bug fix, no schema/protocol change.
  Free to push to master and let `:latest` roll out immediately.
- **Minor** (`v1.2.x` → `v1.3.0`): new feature, no breaking config or
  protocol change. Push to master, but post in operators channel so
  they know what changed.
- **Major** (`v1.x` → `v2.0`): breaking config / protocol / chain
  interaction. Cut a tag, give operators a deadline to upgrade,
  consider gating via a `:stable` tag temporarily.

We're not currently using SemVer rigorously. If you start, document the
contract here.
