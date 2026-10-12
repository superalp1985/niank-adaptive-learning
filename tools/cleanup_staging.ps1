param([Parameter(Mandatory=$true)][string]$Tag)
$ErrorActionPreference = 'Stop'
if ($Tag -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,90}$') { throw 'Invalid tag' }
$taskRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$taskLedger = Join-Path $taskRoot "archives\$Tag"
$taskManifestPath = Join-Path $taskLedger 'manifest.json'
$taskManifest = Get-Content -LiteralPath $taskManifestPath -Raw | ConvertFrom-Json
$taskReceipt = Get-Content -LiteralPath (Join-Path $taskLedger 'verified.json') -Raw | ConvertFrom-Json
$taskHash = (Get-FileHash -LiteralPath $taskManifestPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($taskReceipt.manifest_sha256 -ne $taskHash -or
    ($taskReceipt.parts | ConvertTo-Json -Depth 5 -Compress) -ne
    ($taskManifest.parts | ConvertTo-Json -Depth 5 -Compress)) { throw 'Complete verification required' }
$taskStage = [IO.Path]::GetFullPath((Join-Path $taskRoot ".archive-staging\$Tag"))
if (-not $taskStage.StartsWith($taskRoot + '\.archive-staging\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Unexpected staging path'
}
$taskUpload = Join-Path $taskStage 'upload-repo'
if ((Get-Content -LiteralPath (Join-Path $taskUpload '.git\HEAD') -Raw).Trim() -ne
    "ref: refs/heads/codex/archive-$Tag") { throw 'Unexpected staging repository' }
$taskRemote = & git -C $taskUpload remote get-url origin
if ($LASTEXITCODE -ne 0 -or $taskRemote -ne 'https://github.com/superalp1985/niank-adaptive-learning.git') {
    throw 'Unexpected staging remote'
}
# Only the immutable ZIP and its exact uploaded parts/blobs are disposable here.
$taskTargets = @([pscustomobject]@{path=(Join-Path $taskStage 'snapshot.zip'); sha256=$taskManifest.zip_sha256})
foreach ($taskPart in $taskManifest.parts) {
    if ([IO.Path]::GetFileName($taskPart.name) -ne $taskPart.name) { throw 'Unsafe part name' }
    $taskPartPath = Join-Path $taskUpload ('upload\' + $taskPart.name)
    $taskTargets += [pscustomobject]@{path=$taskPartPath; sha256=$taskPart.sha256}
    $taskObject = & git -C $taskUpload rev-parse "HEAD:upload/$($taskPart.name)"
    if ($LASTEXITCODE -ne 0 -or $taskObject -notmatch '^[a-f0-9]{40}$') { throw 'Missing staged object' }
    $taskObjectPath = Join-Path $taskUpload ('.git\objects\' + $taskObject.Substring(0,2) + '\' + $taskObject.Substring(2))
    if (Test-Path -LiteralPath $taskObjectPath -PathType Leaf) {
        $taskTargets += [pscustomobject]@{path=$taskObjectPath; sha256=$null}
    }
}
foreach ($taskEntry in $taskTargets) {
    $taskPath = [IO.Path]::GetFullPath($taskEntry.path)
    if (-not $taskPath.StartsWith($taskStage + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'Escaped staging' }
    $taskCursor = Get-Item -LiteralPath $taskPath -Force
    while ($taskCursor.FullName -ne $taskRoot) {
        if ($taskCursor.Attributes -band [IO.FileAttributes]::ReparsePoint) { throw 'Linked target refused' }
        $taskCursor = Get-Item -LiteralPath ([IO.Path]::GetDirectoryName($taskCursor.FullName)) -Force
    }
    if ($taskEntry.sha256 -and
        (Get-FileHash -LiteralPath $taskPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskEntry.sha256) {
        throw 'Staging file changed'
    }
}
$taskFreed = [long]0
foreach ($taskEntry in $taskTargets) {
    $taskFreed += (Get-Item -LiteralPath $taskEntry.path -Force).Length
    Remove-Item -LiteralPath $taskEntry.path -Force
}
$taskResult = [ordered]@{tag=$Tag;removed_files=$taskTargets.Count;freed_bytes=$taskFreed;
    scope='Verified archive ZIP, exact parts, and corresponding disposable staging blobs; metadata retained'}
$taskResult | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $taskLedger 'staging-cleanup.json') -Encoding UTF8
$taskResult | ConvertTo-Json -Depth 5
