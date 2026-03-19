# DeFeedback Pro - Windows Installer Script
# Run as Administrator: Right-click → Run with PowerShell
# Or: powershell -ExecutionPolicy Bypass -File install_windows.ps1

param(
    [string]$SourceDir = $PSScriptRoot
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "  DeFeedback Pro - VST3 Plugin Installer" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Determine source paths
$vst3Source = Join-Path $SourceDir "vst3\DeFeedback Pro.vst3"
$onnxDll = Join-Path $SourceDir "vst3\onnxruntime.dll"
$modelSource = Join-Path $SourceDir "models\nsnet2.onnx"

# Fallback: if run from project root
if (-not (Test-Path $vst3Source)) {
    $vst3Source = Join-Path $SourceDir "build\DeFeedbackPro_artefacts\Release\VST3\DeFeedback Pro.vst3"
    $onnxDll = Join-Path $SourceDir "onnxruntime-win-x64-1.17.1\lib\onnxruntime.dll"
    $modelSource = Join-Path $SourceDir "build\models\nsnet2.onnx"
}

if (-not (Test-Path $vst3Source)) {
    Write-Host "ERROR: VST3 plugin not found!" -ForegroundColor Red
    Write-Host "Make sure you run this from the installer directory or project root." -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Installation paths
$vst3Dest = "C:\Program Files\Common Files\VST3\DeFeedback Pro.vst3"
$onnxDest = "$vst3Dest\Contents\x86_64-win"
$modelDest = "$vst3Dest\Contents\Resources\models"

Write-Host "Source:  $vst3Source" -ForegroundColor Gray
Write-Host "Install: $vst3Dest" -ForegroundColor Gray
Write-Host ""

# Check admin rights
$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
    Start-Process powershell.exe -ArgumentList "-ExecutionPolicy Bypass -File `"$PSCommandPath`" -SourceDir `"$SourceDir`"" -Verb RunAs
    exit
}

try {
    # Step 1: Copy VST3 bundle
    Write-Host "[1/3] Installing VST3 plugin..." -ForegroundColor Green
    if (Test-Path $vst3Dest) {
        Remove-Item -Recurse -Force $vst3Dest
    }
    Copy-Item -Recurse -Force $vst3Source "C:\Program Files\Common Files\VST3\"
    Write-Host "  OK: VST3 installed" -ForegroundColor DarkGreen

    # Step 2: Copy ONNX Runtime
    if (Test-Path $onnxDll) {
        Write-Host "[2/3] Installing ONNX Runtime..." -ForegroundColor Green
        Copy-Item -Force $onnxDll $onnxDest
        Write-Host "  OK: onnxruntime.dll installed" -ForegroundColor DarkGreen
    } else {
        Write-Host "[2/3] ONNX Runtime DLL not found, skipping" -ForegroundColor Yellow
    }

    # Step 3: Copy ML model
    if (Test-Path $modelSource) {
        Write-Host "[3/3] Installing ML model (NSNet2)..." -ForegroundColor Green
        New-Item -ItemType Directory -Force -Path $modelDest | Out-Null
        Copy-Item -Force $modelSource $modelDest
        Write-Host "  OK: nsnet2.onnx installed (24MB)" -ForegroundColor DarkGreen
    } else {
        Write-Host "[3/3] ML model not found, skipping" -ForegroundColor Yellow
    }

    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "  Installation complete!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Restart your DAW and scan for new plugins." -ForegroundColor White
    Write-Host "DeFeedback Pro will appear as a VST3 effect." -ForegroundColor White
    Write-Host ""
}
catch {
    Write-Host ""
    Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Read-Host "Press Enter to exit"
