# Frontend Style Ownership

All HTML pages load `frontend/style.css`. This file is the stable entrypoint.
It imports the style files in cascade order. Do not load a layer file directly
from an HTML page.

| File | Owner |
| --- | --- |
| `tokens.css` | Global fonts, colors, and other reusable values. |
| `base.css` | Element defaults and document behavior. |
| `layout.css` | The application shell, header navigation, and main columns. |
| `components.css` | Shared forms, job controls, adapters, presets, and modals. |
| `generation.css` | Generation galleries, viewers, masks, and adapter workspaces. |
| `registry-tools.css` | Workflow builder, history, registry, analysis, and profiler pages. |
| `responsive.css` | All viewport-specific overrides. This import must stay last. |

Put a new rule in the file that owns its UI. Put a rule in `components.css` if
more than one page uses the same component. Put all `@media` rules in
`responsive.css`.

Keep `style.css` as an import-only file. This rule lets existing HTML paths stay
stable and does not add a build step.

Do not remove a selector only because its name looks old. Search all HTML,
JavaScript, and generated markup first. Record the evidence in the change that
removes it.

Run this check after a style change:

```powershell
$frontendTests = Get-ChildItem testing -File -Filter "test_frontend_*.py"
.venv\Scripts\python.exe -m pytest $frontendTests.FullName -q
```
