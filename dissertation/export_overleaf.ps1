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

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $outputPath -Force
Remove-Item -LiteralPath $stagingRoot -Recurse -Force
Write-Host "Created $outputPath"
