# AGENTS.md — Repo hygiene

Scope: this file covers *repo hygiene* — branching, remotes, CI — plus a
short map of the binaries. It is not user-facing project documentation.

This repo follows the branch/release workflow documented in `CONTRIBUTING.md`
— read and follow it for any git, branch, or release work here (the
single-trunk model, `feature/x`/`fix/x` branch naming, how RC tags work,
etc). Don't improvise a different workflow. The short version: there is one
long-lived branch, `main` — no `dev` or `beta` branch exists. `main`
auto-publishes a dev-track build on every push. "Beta" and "stable" are both
just tags, not branches: push a `vX.Y.Z-rc.N` tag to publish a beta-track
build, push a plain `vX.Y.Z` tag to cut the signed stable release.
"Freezing" for stabilization means pausing pushes to `main`, not moving a
branch. This replaced an earlier three-branch (`dev`/`beta`/`main`) model
after `main` was found to have silently rotted out of sync with `dev`/`beta`
across most repos in this ecosystem.

When starting work on a new feature, create branch `feature/<feature-name>`.
When working on a bug or issue, create branch `fix/<issue you are fixing>`.

## Remotes
- `origin` — Forgejo (`git.breadway.dev` via Hestia, SSH) — authoritative.
- `github` — GitHub mirror. Push both when publishing.

## CI
- `dev-release.yml` triggers on `push: branches: ['main']`.
- `rc-release.yml` triggers on `vX.Y.Z-rc.N` tag pushes (beta track).
- `release.yml` triggers on any other `v*` tag push (stable).
  None of these run on plain commits or PRs beyond what's listed.

## Architecture

Two binaries + a shared lib:

| Crate | Role |
|---|---|
| `breadsearch-shared` | XDG paths, config, IPC types, Unix-socket client |
| `breadmill` | Daemon: walk → extract → chunk → embed → index; serves queries |
| `breadsearch` | GTK4 layer-shell overlay: ranked document hits from breadmill |

breadsearch is document/meaning search. breadbox is the app launcher. They
are not the same overlay.

`--screenshot` (`breadsearch/src/screenshot.rs`) captures the search panel
through `bread-screenshots`; do not rewrite it just to retarget the crate pin.

## Don't
- Don't embed credentials in remote URLs — SSH or a credential helper only.
- Don't rewrite `screenshot.rs` as part of pin/docs work.
