# Static Context Contract

This folder selects the minimum accurate context before each task. It has one
job: reduce repeat reading without replacing canonical sources.

## Inputs

- `STATIC.md` is the compact derived cache for stable project facts.
- `static-sources.json` lists its canonical sources and byte hashes.
- `check-static-context.ps1` checks those hashes without showing source text.
- The nearest `AGENTS.md`, the task request, and files changed for the task
  remain required inputs.

## Process

1. Reuse `STATIC.md` in the current thread and later turns.
2. On a new thread, read this folder and run the checker. Do not load the full
   sources when it reports `FRESH`.
3. If it reports `STALE`, read only the listed canonical source paths. Then
   update `STATIC.md` and `static-sources.json` only if the stable contract
   changed.
4. Read task-specific files and current dynamic evidence as needed.
5. Before execution, report: static context reused; dynamic information read
   now; and content not reread.

## Dynamic Information

New requirements, modified code or files, error logs, test results, and runtime
output are dynamic. Read them for the active task. Do not add them to the
static cache by default.

## Default Non-Reads

Do not load `memory-bank/`, `outputs/`, `database/`, `.venv/`, caches, active
logs, `outline.md`, refactor logs, runtime output, or old test results unless
the task needs them. Do not load capability matrices or task lists by default;
read the workflow catalog, registry, or API contract when the task needs them.

## Authority

`STATIC.md` is a derived cache only. The source documents and code remain
authoritative. Keep one home per fact: routing is in root `AGENTS.md`, this
file defines selection, and canonical sources define project behavior.
