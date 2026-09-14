$ErrorActionPreference = "Stop"

$sourceRoot = $PSScriptRoot
$stagingRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("cove-dissertation-" + [guid]::NewGuid().ToString("N"))
$outputPath = Join-Path $sourceRoot "Cove-dissertation-overleaf.zip"

New-Item -ItemType Directory -Path $stagingRoot | Out-Null

$entries = @(
    "main.tex",
    "references.bib",
    "frontmatter",
    "chapters",
    "appendices",
    "assets"
)
foreach ($entry in $entries) {
    Copy-Item -LiteralPath (Join-Path $sourceRoot $entry) -Destination $stagingRoot -Recurse
}

# ZIP timestamps cannot represent dates before 1980. Some imported assets can
# preserve an older filesystem timestamp, so normalise only the temporary copy.
$zipEpoch = [datetime]::SpecifyKind([datetime]'1980-01-01T00:00:00', [DateTimeKind]::Local)
Get-ChildItem -LiteralPath $stagingRoot -Recurse -File | ForEach-Object {
    if ($_.LastWriteTime -lt $zipEpoch) {
        $_.LastWriteTime = $zipEpoch
    }
}

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $outputPath -Force
Remove-Item -LiteralPath $stagingRoot -Recurse -Force
Write-Host "Created $outputPath"
