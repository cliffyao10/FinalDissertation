$ErrorActionPreference = "Stop"

$dissertationDir = $PSScriptRoot
$texBin = Join-Path $dissertationDir "..\.tools\miktex\texmfs\install\miktex\bin\x64"
$xelatex = Join-Path $texBin "xelatex.exe"
$biber = Join-Path $texBin "biber.exe"

if (-not (Test-Path -LiteralPath $xelatex) -or -not (Test-Path -LiteralPath $biber)) {
    throw "The project-local MiKTeX installation is missing. Expected it under $texBin"
}

Push-Location $dissertationDir
try {
    & $xelatex --enable-installer -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "The first XeLaTeX pass failed." }

    & $biber main
    if ($LASTEXITCODE -ne 0) { throw "Biber failed while building the references." }

    & $xelatex --enable-installer -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "The second XeLaTeX pass failed." }

    & $xelatex --enable-installer -interaction=nonstopmode -halt-on-error main.tex
    if ($LASTEXITCODE -ne 0) { throw "The final XeLaTeX pass failed." }

    Write-Host "Built successfully: $(Join-Path $dissertationDir 'main.pdf')"
}
finally {
    Pop-Location
}
