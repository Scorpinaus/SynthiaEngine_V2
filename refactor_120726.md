# SynthaEngine Refactor Log — 12 July 2026

## Purpose

Reduce the amount of code that the team must maintain while making the project
easier to read. Public workflow behavior and model-specific behavior must remain
clear and stable.

## Working rules

- Remove repeated mechanics, not useful explanations or explicit model behavior.
- Keep model-family differences easy to find.
- Use small, typed building blocks instead of deep inheritance or hidden magic.
- Change simple model families first. SD1.5 and SDXL come later because they have
  more specialized behavior.
- Run focused tests after every migration.
- Measure application, test, and vendored code separately.

## Changes made

### Phase 0 — Baseline measurement

- Added `tools/code_metrics.py`.
- The script counts Python, JavaScript, HTML, and CSS files and lines.
- It reports backend, frontend, tests, tools, and custom pipelines separately.
- "Maintained total" includes backend, frontend, and tools. It excludes tests
  and custom pipelines so they do not distort the refactor result.
- Added focused tests in `testing/test_code_metrics.py`.

### Baseline

The first verified report is:

| Category | Files | Physical lines | Likely code lines |
|---|---:|---:|---:|
| Backend | 109 | 23,178 | 19,654 |
| Frontend | 90 | 23,577 | 20,843 |
| Tools | 4 | 989 | 843 |
| Maintained total | 203 | 47,744 | 41,340 |
| Tests | 69 | 17,434 | 14,882 |
| Custom pipelines | 42 | 26,635 | 21,729 |

Verification: `testing/test_code_metrics.py` passed with 2 tests.

### Phase 1 — Shared frontend generation controller (first slice)

- Added `frontend/generation_page.js` for the repeated single-task image page
  behavior: model loading, catalog defaults, presets, LoRA payloads, job
  submission, SSE updates, and gallery results.
- Migrated Flux text-to-image and image-to-image to the shared controller.
- Kept each task name, field mapping, defaults, and artifact upload visible in
  its small family page file.
- Did not move the Flux mask editor. It is special inpainting behavior and is
  easier to understand in the inpainting page.
- Added `testing/test_frontend_generation_page.py` to check script ordering and
  the visible Flux task contracts.
- Maintained likely-code lines changed from 41,340 to 41,108: a net reduction
  of 232 lines in this first two-page slice.
- Updated older frontend contract tests so they verify the shared controller
  and the small Flux configuration files instead of requiring duplicated setup
  code inside every page.
- Verification passed: 131 frontend tests and JavaScript syntax checks for the
  shared controller and both migrated Flux pages.

### Phase 1 — Qwen-Image and Z-Image migration

- Migrated the text-to-image and image-to-image pages for Qwen-Image and
  Z-Image to `generation_page.js`.
- Qwen's `true_cfg_scale` remains visible in both Qwen page configurations. It
  was not added to the shared generic behavior.
- Image upload remains visible in each image-to-image page, so readers can see
  where `initial_image` comes from.
- The inpainting mask editors remain local because they contain real page-specific
  drawing and blur behavior.
- Updated frontend tests to check the new shared ownership and the visible
  family task contracts.
- Verification passed: all 131 frontend tests and JavaScript syntax checks.
- Maintained likely-code lines are now 40,334, down 1,006 from the 41,340
  baseline.

### Phase 2 — Shared simple image workflow mechanics

- Added `backend/workflow/image_tasks.py`.
- It owns repeated image opening, RGB/mask conversion, resizing, strength
  validation, common runtime payload creation, and result validation.
- Flux, Qwen-Image, and Z-Image still have their own workflow modules and named
  task functions. This keeps task navigation and stack traces clear.
- Family defaults are short typed `ImageTaskDefaults` values in each family
  module.
- Qwen's LoRA contract normalization remains in `qwen_image.py`, where the
  family-specific rule is easy to find.
- Qwen and Z-Image strength remapping is explicit at their named img2img call
  sites. Flux does not enable it.
- Verification passed: backend compile and all 227 workflow tests.
- Maintained likely-code lines are now 40,244, down 1,096 from baseline.

### Phase 3 — Simple-family schema composition

- Added one shallow private input base for each of Flux, Qwen-Image, and
  Z-Image.
- Each base contains only fields repeated by all three tasks in that family.
- Text-to-image, image-to-image, and inpaint classes still show their required
  images, masks, sizes, strengths, and prompt requirements explicitly.
- Did not create one base shared by every model family. Family defaults remain
  local and readable.
- Audited the task registry and catalog. `TaskDefinition` already provides one
  source for input model, output model, and handler; catalog capabilities and UI
  hints are derived from those models. Adding another metadata layer now would
  add code rather than remove it, so no redundant registry abstraction was
  introduced.
- Verification passed: workflow compilation plus 30 focused registry, catalog,
  Flux, Qwen-Image, and Z-Image tests.
- Maintained likely-code lines are now 40,197, down 1,143 from baseline.

### Phase 4 — Shared pipeline seed and output operations

- Added `resolve_base_seed` and `save_generated_image` to the existing pipeline
  utility module.
- Flux, Qwen-Image, and Z-Image now use the same seed rule, PNG filename rule,
  metadata fields, output URL construction, and removal of image objects from
  metadata.
- The actual Diffusers arguments and generation loops remain in each family
  pipeline. This keeps model behavior visible.
- Flux still measures output-save time around the shared save call, so runtime
  profiling behavior is preserved.
- Added focused tests for explicit seeds and saved PNG metadata.
- Verification passed: compilation, 9 pipeline lifecycle tests, and 20 focused
  Flux/Qwen/Z workflow tests.
- Maintained likely-code lines are now 40,140, down 1,200 from baseline.

### Phase 5 — Shared inpainting editor

- Compared the Flux, Qwen-Image, and Z-Image inpainting scripts. Their mask
  editors differed by only about 33–39 lines out of roughly 490 lines.
- Added `frontend/components/inpaint_editor.js` for image loading, canvas sizing,
  painting and erasing, zoom, mask preview, saving, and server-side mask blur.
- All three family pages now reuse the editor and `generation_page.js`.
- Each family page remains a small readable configuration that shows its task
  name, defaults, fields, artifact uploads, and generation function.
- Qwen's true CFG field remains visible only in the Qwen inpaint page.
- Resetting the source image also resets old mask previews and blurred masks, as
  the original pages did.
- Verification passed: JavaScript syntax checks and all 132 frontend tests.
- Maintained likely-code lines are now 39,021, down 2,319 from baseline.

### Phase 6 — API domain split (history and presets)

- Moved `/history` and all of its PNG/video metadata helpers into
  `backend/api/history.py`.
- Moved `/api/presets` request models and CRUD routes into
  `backend/api/presets.py`.
- Kept literal route strings and direct typed registry calls. No generic CRUD
  router or hidden route discovery was added.
- `backend/main.py` now focuses more clearly on application assembly and the
  API domains that have not moved yet. Its measured physical size dropped from
  about 746 lines at the original review to 567 lines now.
- The total LOC change is intentionally small because this step improves file
  ownership rather than pretending moved code was deleted.
- Verification passed: backend compilation plus 5 focused history and preset
  API tests.
- Maintained likely-code lines are now 39,002, down 2,338 from baseline.

### Phase 6 continued — Model and LoRA APIs

- Moved model registry routes and request models to `backend/api/models.py`.
- Moved LoRA registry routes, validation, and request models to
  `backend/api/loras.py`.
- Preserved model family aliases and the existing 400/404/409 error behavior.
- `backend/main.py` is now 383 measured lines. It mainly assembles the app and
  owns artifacts, local path selection, ControlNet/tool endpoints, and mask blur.
- Compilation and 16 focused model, LoRA, and docs contract tests passed. The
  known LoRA `prompt_presets: []` response mismatch remains one failing test.
- Maintained likely-code lines are now 38,982, down 2,358 from baseline.

### Phase 7 — Custom pipeline audit and cleanup

- Compared all 42 original custom pipeline files with same-named files in the
  installed Diffusers package. None were byte-identical.
- Searched backend, frontend, tests, tools, and docs for actual module imports.
- Removed 20 unreachable pipeline files: 6 old Stable Diffusion experiments,
  10 unused Flux variants, and 4 unused local Z-Image files.
- Kept production Flux and Anima code and the documented/tested FluxModular
  low-memory implementation.
- Added `docs/CUSTOM_PIPELINES_AUDIT.md` with the classification and evidence.
- Custom pipeline physical lines decreased from 26,635 to 11,193: 15,442 fewer
  repository lines outside maintained application code.
- Verification passed: remaining custom pipeline compilation and 34 focused
  Flux, Z-Image, and FluxModular tests.

## Errors encountered

- A repository inspection command returned exit code 1 because PowerShell did
  not expand `frontend\\flux\\*.js` for `rg`. No files were changed by the failed
  command. The inspection was rerun by reading the files directly.
- The documented `pytest testing/test_frontend_*.py` command also did not expand
  the wildcard in PowerShell. JavaScript syntax checks still passed. The tests
  were rerun using an explicit list of matching files.
- The first explicit frontend test run found 6 failures. The tests expected the
  old cache version and expected LoRA/preset setup to be copied into each Flux
  page. The application behavior was intentionally moved to the shared
  controller, so these assertions were updated to check the new ownership. The
  complete frontend suite then passed.
- The first test run after migrating Qwen-Image and Z-Image found 12 failures
  for the same reason: old tests required duplicated setup code and old script
  cache versions. The tests were updated to verify shared behavior plus each
  family's explicit task configuration. The full suite then passed.
- Backend workflow consolidation produced only existing dependency deprecation
  warnings from SWIG, TorchScript, and Python 3.14. There were no test failures.
- A combined Flux/Qwen/Z subprocess test run failed because the test files
  replace the global `diffusers` module with incomplete lightweight stubs. The
  Qwen and Z tests also fail alone because their stubs omit scheduler classes
  imported by `backend.utilities.schedulers`. This is a test harness limitation,
  not a compile or workflow failure. The shared utility and family workflow
  tests passed; the subprocess isolation issue remains recorded for later test
  cleanup.
- The complete `testing/` suite ran 614 tests: 588 passed and 26 failed. The
  failures are outside this refactor slice: one existing LoRA API response
  mismatch, nineteen SD1.5 modular pipeline failures, and six SD1.5 IP-Adapter
  failures. Focused frontend, workflow, schema, lifecycle, compile, and output
  metadata checks for the changed code pass. These full-suite failures must be
  resolved or formally baselined before the final completion audit.
- The first frontend test run after extracting the inpaint editor found 9 stale
  assertions. They expected editor, LoRA, and preset implementation code inside
  each family script. Tests now verify the shared editor plus the explicit
  family task fields. The complete frontend suite then passed.
- Two custom-pipeline audit commands had PowerShell construction errors: one
  tried to hash a missing same-name candidate, and one used an invalid empty
  pipeline element. Both were read-only commands. Corrected commands completed
  and reported zero exact upstream matches.
- Another audit display command passed file objects to `Get-Content` in a way
  that resolved names from the repository root. It printed errors but changed
  nothing; the corrected `-LiteralPath` measurement was used.
- A route-search command used a backslash inside an `rg` regular expression and
  failed to parse. It was rerun with a simple literal search.
- A read-only PowerShell inspection command had mismatched quote characters and
  was rerun with a simpler single-quoted search pattern.
- A frontend verification command used `pytest`, which is not on this shell's
  PATH. JavaScript syntax checks completed before that error; the test command
  was rerun through the project's virtual-environment Python.

## Learnings

- Raw repository line counts are misleading because `custom_pipelines/` contains
  a large amount of copied or customized pipeline code.
- Tests should be measured but should not be reduced just to improve a number.
- Physical line counts provide a simple stable baseline across all languages.
  The script also separates blank, comment, and likely code lines for context.

## Next work

- Reuse the shared controller in remaining simple pages where it improves
  readability; keep specialized mask/video behavior local.
- Compose the repeated simple-family input schemas without changing their
  generated contracts.

## Frontend generation follow-up

- Anima text-to-image now declares only its fields, defaults, and task name; it
  uses the shared controller for models, presets, jobs, and gallery updates.
- ERNIE-Image now uses the same shared plumbing. Its adapter modal stays in the
  family file because that behavior is model-specific and easier to understand
  there.
- The controller now supports explicitly disabled LoRA behavior, payloads that
  do not use a `Lora` envelope, checkbox fields, and preset-only model names.
  These options keep the family payload contracts visible.
- Focused frontend verification passes: 81 tests, plus JavaScript syntax checks
  for the controller and both migrated scripts.
- The first full-suite run after these two migrations found four stale source-
  shape tests for Anima and ERNIE. They checked that shared model, preset, and
  job code remained copied into each family file. The tests now check the
  family configuration and the shared controller contract instead. The other
  26 failures exactly match the earlier broad-suite baseline.

## Completion audit

- Final maintained-code measurement: 210 files, 44,672 physical lines and
  38,587 likely-code lines. The baseline was 41,340 likely-code lines, so the
  maintained code is 2,753 likely-code lines smaller (6.7%).
- The audited custom-pipeline area is 11,193 physical lines and 9,143 likely-
  code lines after removing 20 unreachable files. The overall Git diff removes
  more than 20,000 physical lines because these unused pipeline copies were the
  largest source of repository bulk.
- Python compilation succeeds. JavaScript syntax checks succeed for the shared
  controller and the newly migrated family scripts.
- The final full suite reports 590 passing and 26 failing tests. Those are the
  same 26 failures recorded before the last frontend work: one LoRA response-
  shape mismatch, nineteen modular SD1.5 failures, and six SD1.5 IP-Adapter
  test-double failures. The refactor added no new broad-suite failures.
- An exact-content scan found no duplicate maintained JavaScript, CSS, or Python
  files. The remaining large SD1.5, SDXL, video, mask, and adapter flows have
  specialized behavior. They were deliberately left explicit because forcing
  them through the simple controller would reduce line count at the cost of
  human readability.
- Tests were not shortened to improve the line metric. New focused tests cover
  the metrics tool, shared frontend controller, pipeline lifecycle helpers, and
  preserved family task contracts.

## Readability result

- Ordinary image pages now read as short declarations of family, task, model,
  and input fields. Upload and mask behavior remains in small family wrappers.
- Workflow files now show task-specific policy while shared validation and
  execution live in one named helper module.
- `backend/main.py` is a composition root again; history, preset, model, and
  LoRA endpoints live in named API modules.
- Repeated seed and output-file lifecycle behavior has one implementation.
- Custom pipeline ownership and deletion decisions are documented in
  `docs/CUSTOM_PIPELINES_AUDIT.md`, so future upgrades do not require another
  archaeology pass.
