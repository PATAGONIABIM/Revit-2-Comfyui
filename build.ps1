<!-- ===== build.ps1 (PowerShell Build Script) ===== -->
<#
.SYNOPSIS
    Build script for WabiSabi Bridge Revit Plugin
.DESCRIPTION
    Builds the plugin for multiple Revit versions and creates installer
.PARAMETER Configuration
    Build configuration (Debug/Release)
.PARAMETER RevitVersion
    Target Revit version (2024/2025)
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("Debug", "Release")]
    [string]$Configuration = "Release",
    
    [Parameter(Mandatory=$false)]
    [ValidateSet("2024", "2025", "All")]
    [string]$RevitVersion = "All"
)

$ErrorActionPreference = "Stop"

Write-Host "WabiSabi Bridge Build Script" -ForegroundColor Cyan
Write-Host "============================" -ForegroundColor Cyan

# Clean previous builds
Write-Host "Cleaning previous builds..." -ForegroundColor Yellow
if (Test-Path ".\bin") { Remove-Item -Path ".\bin" -Recurse -Force }
if (Test-Path ".\obj") { Remove-Item -Path ".\obj" -Recurse -Force }

# Build function
function Build-RevitVersion {
    param($Version)
    
    Write-Host "`nBuilding for Revit $Version..." -ForegroundColor Green
    
    $config = "${Configuration}_${Version}"
    dotnet build -c $config
    
    if ($LASTEXITCODE -ne 0) {
        throw "Build failed for Revit $Version"
    }
    
    # Create output directory
    $outputDir = ".\dist\Revit$Version"
    New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
    
    # Copy files
    Copy-Item ".\bin\$Configuration\WabiSabiRevitBridge.dll" -Destination $outputDir
    Copy-Item ".\WabiSabiRevitBridge.addin" -Destination $outputDir
    
    # Update addin file for version
    $addinContent = Get-Content "$outputDir\WabiSabiRevitBridge.addin"
    $addinContent = $addinContent -replace "<MinRevitVersion>\d+</MinRevitVersion>", "<MinRevitVersion>$Version</MinRevitVersion>"
    $addinContent = $addinContent -replace "<MaxRevitVersion>\d+</MaxRevitVersion>", "<MaxRevitVersion>$Version</MaxRevitVersion>"
    Set-Content -Path "$outputDir\WabiSabiRevitBridge.addin" -Value $addinContent
    
    Write-Host "Build completed for Revit $Version" -ForegroundColor Green
}

# Build based on parameter
if ($RevitVersion -eq "All") {
    Build-RevitVersion "2024"
    Build-RevitVersion "2025"
} else {
    Build-RevitVersion $RevitVersion
}

# Create installer
Write-Host "`nCreating installer..." -ForegroundColor Yellow

$installerScript = @"
@echo off
echo WabiSabi Bridge Installer
echo ========================
echo.

set REVIT_VERSION=%1
if "%REVIT_VERSION%"=="" (
    echo Please specify Revit version: install.bat 2024 or install.bat 2025
    exit /b 1
)

set ADDIN_DIR=%APPDATA%\Autodesk\Revit\Addins\%REVIT_VERSION%
echo Installing for Revit %REVIT_VERSION%...

if not exist "%ADDIN_DIR%" mkdir "%ADDIN_DIR%"

copy /Y "Revit%REVIT_VERSION%\WabiSabiRevitBridge.dll" "%ADDIN_DIR%\"
copy /Y "Revit%REVIT_VERSION%\WabiSabiRevitBridge.addin" "%ADDIN_DIR%\"

echo.
echo Installation completed!
echo Please restart Revit to load the plugin.
pause
"@

Set-Content -Path ".\dist\install.bat" -Value $installerScript

Write-Host "`nBuild completed successfully!" -ForegroundColor Green
Write-Host "Output files in: .\dist\" -ForegroundColor Cyan