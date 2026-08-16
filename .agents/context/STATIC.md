# SynthaEngine Static Context

Verified: 2026-08-16. This is a compact derived cache. Its listed canonical
sources remain authoritative and must resolve any conflict.

## Project Background

SynthaEngine is a local image-generation service. It provides a FastAPI backend
and a lightweight static web UI. It uses Hugging Face Diffusers and Transformers
to provide several model families through one workflow-job API. The system
supports ordered task chains, uploaded artifacts, persistent job status, and
SSE job updates. Do not treat this cache as a current capability list. Read the
workflow catalog, registry, and API contract for current task identifiers,
fields, defaults, outputs, or feature combinations.

## Technical Architecture

`backend/main.py` is the only FastAPI application composition root and keeps
the stable `backend.main:app` entry target. API behavior is owned by
`backend/api/`. `backend/settings.py` owns typed process settings. It resolves
repository-relative paths and controls CORS, upload limits, embedded-worker
policy, path-picker policy, logs, worker capacity, and pipeline-cache budgets.

Jobs use SQLite storage. The job subsystem owns persistence, leases, status
transitions, cancellation, polling, execution, and cleanup. A renderer can run
inside FastAPI or as the separate worker that `run_app.bat` starts. The normal
startup uses three processes: API, renderer, and a static frontend server.
Use one renderer process for normal local operation.

The workflow subsystem owns input and output contracts, task definitions,
assembly, validation, reference resolution, dispatch, progress, and the
derived schema and catalog. The assembled task definitions are the source for
task support. Workflow tasks run in stable topological order. The system
rejects invalid references and cycles before model execution. Image and video
inputs use uploaded artifacts. Artifacts are ephemeral and cleanup runs after
the workflow reaches a terminal state.

Dependency direction is strict: application -> settings -> API -> jobs and
workflow views; renderer -> jobs -> workflow -> family runtimes -> adapters,
registries, LoRA support, and utilities. API modules must not import concrete
family runtimes. Jobs must not import API modules or family runtimes. Workflow
must not import API or jobs. A family runtime must not import another family,
workflow, jobs, or API. Supporting layers must not depend on orchestration or
family runtime layers. The two documented support exceptions are preset access
to assembled workflow input models and pipeline utility access to the model
registry contract.

## Runtime and Business Constraints

The only generation job kind is `workflow`. Jobs have persistent status and
best-effort cancellation. Clients can poll job data or use SSE. Use the
workflow API contract for endpoint shapes, validation rules, references, and
error behavior. The catalog endpoint is the machine-readable capability source.

Pipelines are task-scoped by default. Generation code loads a pipeline, applies
task options, generates output, saves it, and releases state in `finally`.
Adapters, hooks, pipeline references, and large tensors must not leak to a
later task. Feature-specific cleanup occurs before
`backend.utilities.pipeline.release_pipeline`; that helper releases hooks and
memory. Log cleanup failure but do not hide the original generation error.

The shared subprocess transport owns typed parent-child request and result
envelopes, temporary artifacts, launch, validation, and child cleanup. Many
model renders use one-shot subprocesses. A persistent pipeline cache is opt-in,
has explicit ownership and budgets, and must preserve adapter isolation and
release on failure, eviction, and shutdown. Read
`docs/PIPELINE_LIFECYCLE.md` before runtime-generation changes.

Validate artifacts, paths, task references, model identifiers, and other user
inputs at trust boundaries. Do not commit secrets, tokens, model credentials,
private prompts, generated artifacts, or database contents. Do not make network
downloads, destructive file changes, database migrations, dependency upgrades,
or Git-history changes unless the active task requires and authorizes them.

## Coding and Maintenance Rules

Use ASD-STE100 Simplified Technical English. Inspect relevant code, tests,
documentation, and local `AGENTS.md` files before changing behavior. Preserve
user changes. Make the smallest coherent change. Do not silently change a
documented contract when code and documentation disagree; report the mismatch.

Keep changes to public task inputs, outputs, capabilities, routes, schemas,
catalog exposure, tests, and documentation aligned. Respect model-family
boundaries. Keep errors actionable and preserve job state transitions. Prefer
existing dependencies and document platform-specific requirements. Do not edit
generated or third-party code when a maintained source or wrapper is available.
Runtime dependencies are in `requirements.txt`; development dependencies are in
`requirements-dev.txt`; tested direct pins are in `constraints.txt`. Install
the platform-appropriate PyTorch or CUDA build separately.

Frontend styles use the documented stable entrypoint and layer ownership. Read
`frontend/styles/README.md` before style work. Do not change the style import
contract by accidental local imports or duplicated shared rules.

## Testing Rules

Use the smallest focused check that proves the change. Ordinary unit and
contract tests should not need model downloads or a GPU unless the test is an
explicit integration or hardware test. Run the repository virtual-environment
Python when it is available.

Architecture contracts protect imports, the one composition root, route and
router ownership, the workflow envelope, public task identifiers, and catalog
derivation. Run architecture checks after architecture changes. Run focused
workflow, lifecycle, subprocess, frontend, or API tests according to the
changed boundary. Run `git diff --check` before completion. Report whether a
test was mocked, unit-level, or hardware backed; do not claim GPU generation
unless it ran.

## Source Routing and Default Non-Reads

- `skills.md`: working rules, validation, scope, and security guardrails.
- `README.md`: project overview, install, startup, and process configuration.
- `docs/ARCHITECTURE.md`: owners, layers, boundaries, and route structure.
- `docs/PIPELINE_LIFECYCLE.md`: generation cleanup, subprocess, and cache rules.
- `docs/WORKFLOW_API.md`: public workflow API and current task contract.
- `pytest.ini` and `testing/test_architecture_contracts.py`: test discovery and
  static contract guards.
- `requirements*.txt` and `constraints.txt`: dependency policy and pins.
- `frontend/styles/README.md`: style entrypoint and ownership rules.

Do not load `memory-bank/`, `outputs/`, `database/`, `.venv/`, caches, active
logs, `outline.md`, refactor logs, runtime output, or prior test results by
default. They are dynamic or historical. New requirements, modified files,
errors, test results, and runtime output are dynamic: read only the items that
the active task needs.
