param([Parameter(Mandatory=$true)][string]$Tag)
$ErrorActionPreference = 'Stop'
if ($Tag -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,90}$') { throw 'Invalid tag' }
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskLedger = Join-Path $taskRoot "archives\$Tag"
$taskPlan = Get-Content -LiteralPath (Join-Path $taskLedger 'cleanup-plan.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskReceipt = Get-Content -LiteralPath (Join-Path $taskLedger 'verified.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$taskManifestHash = (Get-FileHash -LiteralPath (Join-Path $taskLedger 'manifest.json') -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskPlan.root -ne $taskRoot -or $taskPlan.manifest_sha256 -ne $taskManifestHash -or $taskReceipt.manifest_sha256 -ne $taskManifestHash) {
    throw 'Verification receipt or workspace mismatch'
}
$taskRemoved = 0
$taskFreed = [long]0
$taskChanged = @()
foreach ($taskEntry in $taskPlan.entries) {
    $taskPath = [IO.Path]::GetFullPath((Join-Path $taskRoot $taskEntry.path))
    if (-not $taskPath.StartsWith($taskRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Cleanup target escaped workspace'
    }
    if (-not (Test-Path -LiteralPath $taskPath -PathType Leaf)) { continue }
    $taskCursor = Get-Item -LiteralPath $taskPath -Force
    while ($taskCursor.FullName -ne $taskRoot) {
        if ($taskCursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked cleanup target refused' }
        $taskCursor = Get-Item -LiteralPath ([IO.Path]::GetDirectoryName($taskCursor.FullName)) -Force
    }
    $taskHash = (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($taskHash -ne $taskEntry.sha256) { $taskChanged += $taskEntry.path; continue }
    $taskSize = (Get-Item -LiteralPath $taskPath).Length
    Remove-Item -LiteralPath $taskPath
    $taskRemoved += 1
    $taskFreed += $taskSize
}
$taskResult = [ordered]@{tag=$Tag; removed_files=$taskRemoved; freed_bytes=$taskFreed; changed_files_preserved=$taskChanged}
$taskResult | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLedger 'cleanup.json') -Encoding UTF8
$taskResult | ConvertTo-Json -Depth 5
