# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this is

sars fits **species-area relationship curves** in Python, a conceptual mirror
of the R `sars` package (Matthews et al. 2019) on the Python scientific stack.

## Workflow

- **The user handles all commits.** Make changes in the working tree and stop
  there. Do not run `git commit`, and do not offer to, unless asked. Branching
  and pushing likewise only on request. Work happens on `dev`; `main` releases.

## Commands

The project uses **`uv`**; the `Makefile` wraps the common tasks:

```sh
make install   # uv sync --extra dev
make test      # uv run pytest
make lint      # ruff check src tests
make fmt       # ruff format + ruff check --fix
make check     # lint + format check + test
```

## Keeping dev in step with main

A merge to `main` that passes CI is released automatically (`release.yml`):
the bot commits "Update changelog for vX.Y.Z" to `main` and tags it. `dev`
never receives that commit, so after every release `dev` is behind `main`.
Before starting work on a dev branch, check:

```sh
git fetch origin
git rev-list --count HEAD..origin/main   # 0 means dev is current
```

If the count is non-zero, fast-forward first (`git merge --ff-only origin/main`)
so the changelog's `[Unreleased]` section starts empty and the next PR doesn't
carry a stale release roll. If `--ff-only` refuses, `dev` and `main` have
diverged: stop and tell the user rather than merging or rebasing on your own.
Say so too if you notice a release land mid-session.
