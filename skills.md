# SynthaEngine Operational Parameters

This file defines the working rules for planning, implementing, reviewing, and validating changes in this repository. Use `outline.md` to track the plan for the current body of work.

## Mission

Maintain SynthaEngine as a reliable local image-generation service with a stable workflow API, clear model-family boundaries, predictable job execution, and a lightweight frontend.

## Sources of truth

Consult the narrowest relevant source before changing behavior:

- `README.md` for the project overview and local startup flow.
- `docs/WORKFLOW_API.md` for the workflow API contract.
- `GET /api/workflow/catalog` and the workflow registry for supported task capabilities.
- `docs/PIPELINE_LIFECYCLE.md` for model loading, cleanup, and GPU-memory rules.
- The nearest `AGENTS.md` for directory-specific instructions.
- Existing tests for expected behavior and compatibility constraints.

When documentation and implementation disagree, identify the mismatch explicitly. Do not silently redefine the contract.

## Operating principles

1. Inspect before editing. Read the relevant implementation, tests, documentation, and local instructions first.
2. Preserve scope. Make the smallest coherent change that satisfies the active goal; avoid unrelated cleanup.
3. Preserve user work. Treat existing uncommitted changes as intentional and do not overwrite, revert, stage, or reformat them without permission.
4. Keep contracts aligned. Changes to task inputs, outputs, capabilities, or routes must update implementation, validation, catalog/schema exposure, tests, and documentation where applicable.
5. Respect model-family boundaries. Do not assume that SD1.5, SDXL, Flux, WAN, Qwen-Image, Z-Image, ERNIE-Image, or Anima share interchangeable pipeline behavior.
6. Protect runtime state. Keep adapter, hook, pipeline, temporary-artifact, and accelerator-memory cleanup in failure-safe paths, normally `finally` blocks.
7. Prefer deterministic checks. Unit and contract tests should not require downloading model weights or using a GPU unless the test is explicitly an integration or hardware test.
8. Keep dependencies intentional. Avoid adding packages when the existing stack or standard library is sufficient; document any platform-specific installation requirement.
9. Handle errors visibly. Return actionable API errors, preserve job state transitions, and avoid swallowing exceptions that operators need to diagnose.
10. Secure external inputs. Validate uploaded artifacts, paths, task references, model identifiers, and user-controlled parameters at trust boundaries.

## Standard workflow

For each task:

1. State the objective and acceptance criteria in `outline.md`.
2. Inspect `git status` and the files in scope.
3. Trace the affected path end to end: frontend request, API validation, job persistence, worker dispatch, task implementation, and returned artifacts as applicable.
4. Implement one logical change at a time.
5. Add or update the narrowest meaningful tests.
6. Run focused tests first, then broader checks proportional to the risk.
7. Review the diff for accidental changes, secrets, generated outputs, and stale documentation.
8. Update `outline.md` with completed work, decisions, remaining risks, and deferred items.

## Validation expectations

- Run focused tests with `.venv\Scripts\python.exe -m pytest <test-path>` when the repository virtual environment is available.
- For workflow-contract changes, validate the affected registry, schema/catalog, API, and frontend assumptions.
- For lifecycle or pipeline changes, include failure-path coverage and verify cleanup behavior.
- For frontend changes, verify request payload construction, status/error handling, and relevant script tests.
- Do not claim GPU or model-generation validation unless it actually ran; distinguish mocked/unit results from hardware-backed results.
- Run `git diff --check` before considering the work complete.

## Completion criteria

A task is complete when:

- The acceptance criteria in `outline.md` are satisfied.
- Relevant tests and checks pass, or limitations are recorded with their cause.
- Public behavior and documentation agree.
- Failure and cleanup paths have been considered.
- The final diff contains only intended changes.
- Follow-up work is captured explicitly rather than implied.

## Guardrails

- Never commit secrets, tokens, model credentials, private prompts, generated user artifacts, or local database contents.
- Do not make network downloads, destructive filesystem changes, database migrations, dependency upgrades, or Git history changes unless the active task requires and authorizes them.
- Do not edit generated or third-party code when a maintained source file or wrapper is the correct change point.
- Do not report success based only on code inspection when an executable, relevant check is available.
