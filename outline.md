# SynthaEngine Architecture Simplification Plan

This is the living plan for the current body of work. Update it as scope,
decisions, evidence, and risks change. Operational rules are defined in
`skills.md`.

## Objective

Simplify SynthaEngine's internal architecture so that ownership, dependencies,
and model-family differences are easy to understand and change. Preserve the
workflow API, catalog/schema, job behavior, generated output shapes, and
pipeline cleanup guarantees while reducing duplicated orchestration code.

The optimization target is maintainability and predictable control flow. GPU
generation performance is not part of this initiative unless a measured
architectural change affects it.

## Acceptance criteria

- [x] The current automated test suite has a green, documented non-GPU baseline
      before structural changes begin.
- [ ] `backend/main.py` is an application composition root; HTTP domains are
      owned by focused modules under `backend/api/`.
- [x] Workflow contracts, registration, runtime dependency binding, and
      execution have explicit owners without package-level module proxy magic.
- [x] All subprocess-backed model families use one tested transport/protocol
      implementation while retaining family-specific runners.
- [ ] Job persistence/leases and workflow execution are separated behind clear,
      typed interfaces.
- [ ] SD1.5 and SDXL generation code is divided by responsibility without
      hiding their model-specific behavior behind deep inheritance.
- [ ] Shared frontend orchestration is not reimplemented by individual pages;
      SD1.5/SDXL special behavior remains visible in named feature modules.
- [ ] Global styles and configuration are divided into discoverable,
      single-purpose modules with stable compatibility entrypoints.
- [ ] Public route paths, task identifiers, payload fields, output models,
      catalog capabilities, reference syntax, and error contracts are unchanged
      unless separately approved.
- [ ] Focused checks, the complete automated suite, Python compilation,
      JavaScript syntax checks, code metrics, and `git diff --check` pass at the
      completion audit.

## Scope

### In scope

- FastAPI application assembly and the remaining route domains in
  `backend/main.py`.
- Workflow package exports, task definitions, runtime binding, validation,
  reference resolution, catalog generation, and orchestration.
- Job queue responsibilities: persistence, leases, worker coordination,
  execution, cancellation, progress, and cleanup.
- The repeated subprocess JSON protocol used by SD1.5, SDXL, Flux, Qwen-Image,
  Z-Image, WAN, ERNIE-Image, and Anima.
- Oversized SD1.5/SDXL workflow and pipeline modules, shared pipeline lifecycle
  helpers, and configuration ownership.
- Repeated frontend generation-page orchestration, large SD1.5/SDXL scripts,
  and the global stylesheet.
- Characterization tests, architecture-boundary tests, code metrics, and
  architecture/lifecycle documentation.

### Out of scope

- Breaking workflow/API changes, task renames, schema redesign, or database
  migrations.
- A frontend framework, bundler, transpiler, or build step.
- A generic model-family inheritance hierarchy or generic CRUD framework.
- Model downloads, GPU-backed quality evaluation, scheduler changes, default
  changes, or output-image comparisons requiring unavailable weights.
- Broad dependency upgrades. The existing uncommitted `constraints.txt` update
  is user work and must remain untouched by this initiative.
- Deleting custom pipelines without a separate reachability and compatibility
  audit.
- Resolving the `SynthaEngine`/`SynthiaEngine` spelling inconsistency without a
  separate naming decision.

## Constraints and assumptions

- Follow `skills.md` and the nearest applicable `AGENTS.md`.
- Preserve existing uncommitted work and keep each refactor slice reviewable.
- Treat `docs/WORKFLOW_API.md`, `GET /api/workflow/catalog`, and workflow schema
  output as contract surfaces.
- Follow `docs/PIPELINE_LIFECYCLE.md`; adapter, hook, temporary-artifact, and
  accelerator-memory cleanup must remain failure-safe.
- Prefer shallow composition, typed data structures, and explicit imports over
  reflection, module mutation, or deep inheritance.
- Share repeated mechanics only after tests demonstrate that behavior is truly
  common. Keep family policy close to the family module.
- Run focused tests after every task. Do not start the next structural task if
  the current slice adds an unexplained regression.
- Current measured baseline (2026-08-01): 210 maintained files and 38,593 likely
  code lines; the largest maintained files include `backend/sd15/pipeline.py`,
  `backend/sdxl/pipeline.py`, `backend/workflow/sd15.py`, the SD1.5/SDXL page
  scripts, and `frontend/style.css`.

## Sequenced task list

| ID | Task | Depends on | Primary result |
| --- | --- | --- | --- |
| ARC-00 | Restore a trustworthy test baseline | None | All current tests pass or are explicitly classified and isolated |
| ARC-01 | Lock contracts and dependency boundaries | ARC-00 | Characterization and architecture tests protect public behavior |
| ARC-02 | Centralize settings and finish API decomposition | ARC-01 | Thin application composition root and explicit configuration |
| ARC-03 | Separate workflow contracts, assembly, and execution | ARC-01 | Explicit workflow imports and single-purpose modules |
| ARC-04 | Separate job storage/leases from execution | ARC-03 | Smaller queue services with stable state transitions |
| ARC-05 | Consolidate the subprocess transport | ARC-01 | One typed and failure-safe parent/child protocol |
| ARC-06 | Decompose SD1.5 and SDXL runtime hotspots | ARC-05 | Task-specific pipeline modules with shared lifecycle mechanics |
| ARC-07 | Simplify SD1.5/SDXL frontend composition | ARC-03 | Shared page mechanics and explicit feature controllers |
| ARC-08 | Modularize frontend styles | ARC-07 | Layered styles with a stable `style.css` entrypoint |
| ARC-09 | Enforce boundaries, update docs, and audit completion | ARC-02 through ARC-08 | Verified architecture and handoff documentation |

## Task definitions

### ARC-00 - Restore a trustworthy test baseline

**Outcome:** Structural work starts from a suite that can reliably identify new
regressions.

- [x] Resolve the LoRA create/list response mismatch by confirming whether
      `prompt_presets: []` is the public contract, then align implementation,
      tests, and docs.
- [x] Repair the 19 modular SD1.5 failures against the pinned Diffusers version;
      keep true integration tests clearly marked and separable.
- [x] Repair the 6 SD1.5 IP-Adapter failures by updating production fallback
      behavior or the incomplete test double, based on the intended prompt
      contract.
- [x] Isolate Diffusers' shared local dynamic-module namespace so SD1.5 and
      PixelDiT tests remain order-independent.
- [x] Register test markers and remove avoidable pytest cache warnings.
- [x] Record the exact passing suite and any hardware-only exclusions.

**Done when:** `.venv\Scripts\python.exe -m pytest testing -q` has no
unclassified failures, and no production behavior was changed merely to satisfy
an inaccurate test.

### ARC-01 - Lock contracts and dependency boundaries

**Outcome:** Refactors can move code without silently changing behavior or
creating new dependency cycles.

- [x] Add characterization coverage for workflow schema/catalog output, route
      registration, job state transitions, subprocess failure propagation, and
      cleanup on failure.
- [x] Define allowed dependency directions for `api`, `jobs`, `workflow`,
      family runtimes, adapters, registries, and utilities.
- [x] Add a lightweight architecture-boundary test for forbidden imports and
      composition-root rules.
- [x] Capture current metrics by maintained code, tests, and vendored custom
      pipelines; use metrics as evidence, not as a deletion target.

**Done when:** The public contract and intended dependency graph are executable
checks, and the refactor can distinguish movement from behavior change.

### ARC-02 - Centralize settings and finish API decomposition

**Outcome:** `backend/main.py` creates the app and owns lifecycle assembly, not
unrelated endpoint behavior or scattered environment parsing.

- [x] Introduce a typed settings boundary for paths, upload limits, CORS,
      worker startup, logging role, cache budgets, and related environment
      values; keep model-specific tuning next to its runtime.
- [x] Resolve repository-relative paths in one place and avoid unnecessary
      import-time filesystem side effects.
- [x] Move artifacts, local-path selection, ControlNet preprocessing, model
      analysis, and mask utilities into focused `backend/api/` routers/services.
- [x] Provide an application factory or equally testable assembly function while
      preserving the current `backend.main:app` startup contract.

**Done when:** `backend/main.py` contains application assembly, middleware,
mounts, router inclusion, health, and lifespan wiring; focused API tests prove
all existing paths, statuses, bodies, upload limits, and security checks.

### ARC-03 - Separate workflow contracts, assembly, and execution

**Outcome:** A reader can find a task's contract, registration, runtime binding,
and execution path without following dynamic package mutation.

- [x] Replace the lazy `backend.workflow` module proxy and `ModuleType`
      `__setattr__` forwarding with explicit compatibility exports.
- [x] Keep schemas in contract modules, task definitions in family modules,
      runtime dependency binding in an assembly module, and DAG execution in
      the engine.
- [x] Remove the large central handler map where family registration can be
      explicit and validated for duplicates.
- [x] Update internal callers and tests to import from owning modules; retain
      only intentional public compatibility imports.
- [x] Keep catalog/schema generation derived from the authoritative task
      definitions rather than adding a second metadata registry.

**Done when:** The engine performs validation, ordering, reference resolution,
dispatch, progress, and result aggregation only; workflow tests and exact
catalog/schema comparisons pass.

### ARC-04 - Separate job storage/leases from execution

**Outcome:** SQLite coordination and workflow execution can be understood and
tested independently.

- [x] Split job/task persistence and lease operations from worker polling and
      workflow execution orchestration.
- [x] Define typed boundaries for claim, heartbeat, progress, terminal state,
      cancellation, crash recovery, and artifact cleanup.
- [x] Keep the single-render invariant and existing transaction semantics
      explicit; do not introduce parallel rendering in this initiative.
- [x] Make cleanup and failure precedence deterministic when both rendering and
      cleanup fail.

**Done when:** Queue/lease tests exercise persistence without loading workflow
runtimes, execution tests use a fake store, and existing job API response/state
behavior remains unchanged.

### ARC-05 - Consolidate the subprocess transport

**Outcome:** Parent/child process mechanics have one implementation, while each
model family retains a small, explicit runner.

- [x] Define typed request/result envelopes and a shared JSON serializer for
      PIL images, paths, primitives, runtime profiles, and errors.
- [x] Centralize temporary-directory creation, command construction, repository
      working directory, exit-code checks, malformed/missing-result handling,
      logging, and cleanup.
- [x] Migrate SD1.5, SDXL, Flux, Qwen-Image, Z-Image, WAN, ERNIE-Image, and Anima
      one family at a time.
- [x] Keep family runners responsible only for operation dispatch, family input
      conversion, generation calls, and final pipeline cleanup.
- [x] Preserve one-shot process isolation and the opt-in Flux cache behavior.

**Done when:** Family-specific subprocess I/O copies are removed, every family
passes focused subprocess tests, and crash/invalid-result tests prove actionable
errors plus temporary-artifact cleanup.

### ARC-06 - Decompose SD1.5 and SDXL runtime hotspots

**Outcome:** Complex model-family behavior stays explicit but no single file
owns loading, adapters, prompt preparation, every operation, saving, and cleanup.

- [x] Inventory the stable public generation entrypoints and characterize their
      pipeline-call arguments before moving code.
- [x] Split loaders/factories, prompt preparation, adapter policy, ControlNet
      preparation, operation-specific generation, and result saving into named
      modules with a thin compatibility facade.
- [x] Split the oversized SD1.5/SDXL workflow adapters by operation or coherent
      feature group while keeping public task identifiers unchanged.
- [x] Reuse existing scheduler, seed/output, LoRA, IP-Adapter, and pipeline
      release helpers; remove local copies only when semantics match.
- [x] Verify all adapter unloading, hook release, cache invalidation, and memory
      cleanup in `finally` paths.

**Done when:** Text2img, img2img, inpaint, ControlNet, IP-Adapter, Hi-Res Fix,
and AnimateDiff responsibilities are independently navigable and testable; all
SD1.5/SDXL workflow, pipeline, lifecycle, and subprocess tests pass.

### ARC-07 - Simplify SD1.5/SDXL frontend composition

**Outcome:** Pages declare task-specific behavior and reuse shared request,
catalog, preset, artifact, job, SSE, and gallery mechanics.

- [x] Characterize current payloads and script load order for every SD1.5/SDXL
      page before extraction.
- [x] Extend shared generation-page hooks only for mechanics that are genuinely
      common; avoid a catch-all configuration object.
- [x] Move ControlNet, IP-Adapter, mask editing, and AnimateDiff/video behavior
      into named feature controllers that compose with the shared page runtime.
- [x] Remove repeated model loading, validation, submission, SSE, error, and
      result-rendering code from family pages.
- [x] Keep family task names, defaults, field mapping, and unsupported feature
      combinations visible near each page entrypoint.

**Done when:** Page files read as small compositions of explicit features,
payload/output behavior is unchanged, JavaScript syntax checks pass, and all
frontend contract tests pass.

### ARC-08 - Modularize frontend styles

**Outcome:** Styles are discoverable by layer and feature without introducing a
frontend build system.

- [ ] Classify `frontend/style.css` into tokens/base, layout, shared components,
      generation pages, registry/tools pages, and responsive rules.
- [ ] Split those layers into named CSS files while retaining `style.css` as the
      stable compatibility entrypoint.
- [ ] Remove dead selectors only with repository-wide usage evidence.
- [ ] Visually verify representative desktop and narrow layouts for text2img,
      inpaint, workflow builder, registry, history, and profiler pages.

**Done when:** Existing HTML entrypoints continue to load styles without a
build step, representative screenshots show no unexplained regressions, and
style ownership is documented.

### ARC-09 - Enforce boundaries, update docs, and audit completion

**Outcome:** The simplified structure is documented, measurable, and protected
against drifting back into the previous shape.

- [ ] Update `docs/ARCHITECTURE.md`, `docs/PIPELINE_LIFECYCLE.md`, README startup
      details, and directory `AGENTS.md` files for the final ownership model.
- [ ] Add or update module-level documentation only where it helps navigation,
      lifecycle, or contract understanding.
- [ ] Run focused suites, the full automated suite, backend compilation,
      JavaScript syntax checks, architecture checks, metrics, and
      `git diff --check`.
- [ ] Compare final metrics with the 2026-08-01 baseline and explain increases
      or decreases; do not reduce tests to improve the count.
- [ ] Review the final diff for public-contract drift, secrets, generated
      outputs, local databases, and accidental edits to user-owned changes.

**Done when:** Every acceptance criterion has recorded evidence, documentation
matches implementation, the diff contains only intended architecture work, and
deferred ideas are explicit.

## Decisions

| Date | Decision | Reason | Consequence |
| --- | --- | --- | --- |
| 2026-08-01 | Restore the test baseline before refactoring. | The current 26 failures would hide regressions introduced by structural work. | ARC-00 gates every architecture task. |
| 2026-08-01 | Optimize for clarity and duplication reduction, not minimum LOC. | Previous work already removed simple duplication; remaining hotspots contain real family-specific behavior. | Shared abstractions require characterization evidence. |
| 2026-08-01 | Preserve static HTML/JS and public workflow contracts. | These are explicit repository constraints and stable user-facing surfaces. | No framework migration or API redesign is included. |
| 2026-08-01 | Refactor in vertical, independently verified slices. | API, job, workflow, runtime, and frontend contracts are coupled end to end. | Each task must finish green before dependent work begins. |
| 2026-08-01 | Guard forbidden dependency directions instead of snapshotting every current edge. | Some current support-layer edges are temporary and should be removable without weakening the architecture policy. | New upward/cross-family coupling fails tests; three narrow exceptions remain explicit in `docs/ARCHITECTURE.md`. |
| 2026-08-01 | Carry typed settings on each FastAPI application instance and keep `backend.main:app` as a factory-created compatibility object. | Upload/security behavior and lifecycle policy must be testable without reloading modules or reparsing environment values inside handlers. | `backend/main.py` is a small composition root; focused routers read application-owned settings through a typed dependency. |
| 2026-08-02 | Make `backend.workflow.assembly` the runtime composition boundary and keep `engine.py` orchestration-only. | Runtime imports, dependency binding, registry construction, and DAG execution previously shared one module and were exposed through package mutation. | Package exports are explicit, family registrations merge with duplicate detection, internal imports target owners, and engine responsibility is enforced by architecture tests. |

## Validation record

| Check | Result | Notes |
| --- | --- | --- |
| Repository instructions and architecture review | Passed | Read `skills.md`, relevant `AGENTS.md`, README, architecture/lifecycle docs, current modules, tests, and prior refactor log. |
| Code metrics | Baseline recorded | 210 maintained files; 38,593 likely code lines (backend 19,422, frontend 18,328, tools 843). |
| Full automated suite (initial) | Baseline failed | 590 passed, 26 failed, 27 warnings in 23.34s: 1 LoRA contract, 19 modular SD1.5, 6 SD1.5 IP-Adapter. |
| ARC-00 focused tests | Passed | LoRA API/docs: 7 passed; modular SD1.5: 27 passed; SD1.5 IP-Adapter plus prompt utilities: 27 passed; SD1.5-to-PixelDiT load order: 2 passed. |
| ARC-00 full automated suite | Passed | 616 passed, 0 failed, 36 third-party deprecation warnings in 83.54s; integration cases included and no hardware-only exclusions. |
| ARC-01 focused contract suite | Passed | 41 route, workflow, job, lease, subprocess, cleanup, and lifecycle tests; 2 code-metrics tests and 2 docs-contract tests also passed. |
| ARC-01 full automated suite | Passed | 622 passed, 0 failed, 36 third-party deprecation warnings in 86.54s. |
| ARC-01 metrics | Recorded | Maintained code unchanged at 210 files / 38,593 likely code lines; tests now 72 files / 15,174 likely code lines; vendored pipelines unchanged at 22 files / 9,143 likely code lines. |
| ARC-02 focused API/settings/subprocess suite | Passed | 93 settings, factory, route ownership, API behavior, queue lifecycle, cache, and family subprocess tests; upload bytes/pixels, CORS, loopback path security, response bodies, and repository working directories covered. |
| ARC-02 full automated suite | Passed | 635 passed, 0 failed, 36 third-party warnings in 86.79s. |
| ARC-02 composition-root metric | Recorded | `backend/main.py` reduced from 461 to 120 lines; endpoint behavior moved to focused routers plus shared artifact persistence. |
| ARC-03 focused workflow/architecture suite | Passed | 245 workflow, catalog/schema, DAG, family adapter, job integration, and architecture-boundary tests passed; 395 unrelated tests were deselected. |
| ARC-03 full automated suite | Passed | 640 passed, 0 failed, 36 third-party warnings in 84.91s. |
| ARC-03 workflow ownership metric | Recorded | `backend/workflow/engine.py` reduced from 587 to 123 physical lines; runtime binding moved to the 415-line assembly module. Maintained code is 219 files / 38,958 likely code lines; tests are 73 files / 15,484 likely code lines. |
| ARC-05 focused subprocess suite | Passed | 64 shared-transport and family subprocess tests passed across SD1.5, SDXL, Flux, Qwen-Image, Z-Image, WAN, ERNIE-Image, and Anima; malformed, missing, crashed, typed-error, cleanup, serializer, cwd, and Flux-cache boundaries covered. |
| ARC-05 full automated suite | Passed | 651 passed, 0 failed, 36 third-party warnings in 83.97s. |
| Test compilation | Passed | `.venv\Scripts\python.exe -m compileall -q backend testing`. |
| Runtime/GPU generation | Not run | ARC-02, ARC-03, and ARC-05 change application/workflow/process structure, not model inference behavior; no model downloads were performed. |
| `git diff --check` | Passed | No whitespace errors; Git reported line-ending conversion advisories for edited files only. |

## Risks and blockers

- **Resolved baseline risk:** ARC-00 repaired the 26 original failures and one
  previously masked order-dependent PixelDiT failure. The full suite is green.
- **Dependency drift:** `constraints.txt` has user-owned uncommitted version
  changes. Preserve it and verify whether those pins explain any ARC-00 failures
  without reverting or modifying the file.
- **Contract coupling:** Workflow changes affect registry, schema/catalog, job
  persistence, frontend payloads, tests, and docs. Exact contract comparisons
  are required; intentional additions must update the ARC-01 route/task sets.
- **Runtime cleanup:** Subprocess and pipeline consolidation could leak adapters,
  hooks, temporary files, or GPU memory on failure. Failure-path tests are a
  release gate.
- **Over-abstraction:** SD1.5, SDXL, video, ControlNet, and adapter paths contain
  legitimate differences. Keep policy local and share only repeated mechanics.
- **Visual regression risk:** Static pages lack a component build system; CSS
  and page refactors require representative browser verification.
- **Naming ambiguity:** The repository uses both `SynthaEngine` and
  `SynthiaEngine`. A rename could affect imports, startup commands, docs, and UI
  labels, so it remains deferred pending an explicit canonical-name decision.

## Deferred work

- GPU-backed image-quality and performance benchmarks.
- New task types, model families, schedulers, or adapter combinations.
- Database schema changes or multi-renderer concurrency.
- Frontend framework/build-tool adoption.
- Canonical product/package spelling migration.
- Further custom-pipeline deletion or upstream replacement audits.

## Progress log

- 2026-08-01: Created the initial `skills.md` operational guide and reusable
  `outline.md` plan structure.
- 2026-08-01: Inspected the current architecture, code metrics, instructions,
  contract surfaces, prior refactor results, and full test baseline.
- 2026-08-01: Replaced the starter outline with the sequenced ARC-00 through
  ARC-09 architecture-simplification backlog. No runtime code was changed.
- 2026-08-01: Completed ARC-00. Corrected the documented LoRA response
  assertion, isolated prompt behavior in IP-Adapter tests, moved Hugging Face
  dynamic-module cache setup ahead of test collection, isolated local custom
  module namespaces, and registered pytest integration/cache settings. The full
  suite passes with 616 tests and no hardware-only exclusions.
- 2026-08-01: Completed ARC-01. Added exact public route/task characterization,
  schema/catalog derivation checks, static dependency rules, and a single
  FastAPI composition-root guard. Added subprocess error and failed-job artifact
  cleanup coverage, documented the allowed dependency direction and current
  exceptions, made the architecture source-of-truth document trackable, and
  finished with 622 passing tests.
- 2026-08-02: Completed ARC-03. Replaced package proxy mutation with explicit
  compatibility exports, separated runtime assembly from DAG execution, made
  family registration explicit and duplicate-safe, moved WAN normalization to
  its family module, updated internal import/mock owners, and added executable
  workflow-boundary guards. The full suite passes with 640 tests.
- 2026-08-02: Completed ARC-05. Added one typed, failure-safe subprocess
  transport and serializer; migrated all eight families; reduced runners to
  dispatch and cleanup; removed six family protocol copies; preserved one-shot
  isolation and Flux cache behavior; and added crash, malformed-result, typed
  error, serializer, cwd, and temporary-cleanup tests. The full suite passes
  with 651 tests.
