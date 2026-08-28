# Contributing

This repository contains the code, tests, configurations, and organizer starter kit needed to
develop the KuaiRand agent collaboratively. Keep local credentials, downloaded benchmark data,
and experiment outputs out of commits.

## Local setup

1. Use CPython 3.12.13 and install the locked environment:

   ```bash
   UV_CACHE_DIR=.uv-cache uv sync --locked
   ```

2. For deterministic local work, use `configs/default.toml` or `configs/smoke.toml`; they do not
   require an API credential.
3. Only contributors who are intentionally running the live research provider should copy
   `.env.example` to `.env.local` and set their own `OPENAI_API_KEY`. Do not send this file or key
   to other contributors.
4. Obtain the official KuaiRand data separately through the approved organizer source, then keep
   it under `.data/`. Do not commit or redistribute the data through this repository.

## What belongs in Git

Commit code, tests, configuration templates, scripts, documentation, and the organizer starter
kit. Do not commit or attach to pull requests:

- `.env`, `.env.local`, or any credential-bearing configuration;
- `.data/` or downloaded archives;
- `runs/`, SQLite databases, checkpoints, predictions, model artifacts, or logs;
- virtual environments, caches, `.DS_Store`, or other machine-specific files.

Before opening a pull request, run the smallest relevant test set and check `git status` to make
sure only intended source files are staged. Use a focused branch and pull request for each change;
avoid committing generated outputs alongside source changes.
