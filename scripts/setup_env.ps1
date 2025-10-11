Param(
    [string]$PythonBin = "python",
    [string]$VenvDir = ""
)

$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($VenvDir)) {
    $VenvDir = Join-Path $RootDir ".venv"
} else {
    try {
        $Resolved = Resolve-Path -LiteralPath $VenvDir -ErrorAction Stop
        $VenvDir = $Resolved.Path
    } catch {
        $VenvDir = [System.IO.Path]::GetFullPath($VenvDir)
    }
}

function Assert-Command {
    param([string]$Name)
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Python interpreter '$Name' not found. Override with -PythonBin or ensure it is on PATH."
    }
}

Assert-Command -Name $PythonBin

Write-Host "Using virtual environment directory: $VenvDir"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment..."
    & $PythonBin -m venv $VenvDir
} else {
    Write-Host "Reusing existing virtual environment."
}

$VenvPython = Join-Path $VenvDir "Scripts/python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Python executable not found in $VenvDir\Scripts."
}

Write-Host "Upgrading pip..."
& $VenvPython -m pip install --upgrade pip

$RequirementsPath = Join-Path $RootDir "requirements.txt"
if (Test-Path $RequirementsPath) {
    Write-Host "Installing requirements..."
    & $VenvPython -m pip install -r $RequirementsPath
}

$SetupPy = Join-Path $RootDir "setup.py"
$PyProject = Join-Path $RootDir "pyproject.toml"
if ((Test-Path $SetupPy) -or (Test-Path $PyProject)) {
    Write-Host "Installing project in editable mode..."
    & $VenvPython -m pip install -e $RootDir
}

Write-Host ""
Write-Host "Virtual environment ready at $VenvDir"
Write-Host "Activate it with:"
Write-Host "  $VenvDir\Scripts\Activate.ps1"
