# Checks static source byte hashes. This script does not write repository files.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path
$manifestPath = Join-Path $PSScriptRoot 'static-sources.json'
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json

$staleItems = @()
foreach ($source in $manifest.sources) {
    $sourcePath = Join-Path $repoRoot $source.path
    if (-not (Test-Path -LiteralPath $sourcePath -PathType Leaf)) {
        $staleItems += "MISSING $($source.path)"
        continue
    }

    $actualHash = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash
    if ($actualHash -ne $source.sha256) {
        $staleItems += "CHANGED $($source.path)"
    }
}

if ($staleItems.Count -eq 0) {
    'FRESH'
    exit 0
}

'STALE'
$staleItems
exit 1
