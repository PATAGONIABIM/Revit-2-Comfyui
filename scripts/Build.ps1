# Build.ps1 - Script de compilación e instalación para WabiSabi Bridge
# Ejecutar desde PowerShell como administrador

param(
    [string]$Configuration = "Release",
    [string]$RevitVersion = "2026"
)

Write-Host "=== WabiSabi Bridge - Compilación e Instalación ===" -ForegroundColor Cyan
Write-Host ""

# Verificar que estamos en el directorio correcto
if (-not (Test-Path "WabiSabiBridge.csproj")) {
    Write-Host "ERROR: No se encuentra WabiSabiBridge.csproj en el directorio actual" -ForegroundColor Red
    Write-Host "Por favor, ejecuta este script desde la carpeta del proyecto" -ForegroundColor Yellow
    exit 1
}

# Buscar MSBuild
$msbuildPath = ""
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"

if (Test-Path $vsWhere) {
    $vsPath = & $vsWhere -latest -requires Microsoft.Component.MSBuild -property installationPath
    $msbuildPath = Join-Path $vsPath "MSBuild\Current\Bin\MSBuild.exe"
}

if (-not (Test-Path $msbuildPath)) {
    # Intentar con .NET Framework MSBuild
    $msbuildPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\MSBuild\Current\Bin\MSBuild.exe"
    $msbuildPath = (Get-Item $msbuildPath -ErrorAction SilentlyContinue | Select-Object -First 1).FullName
}

if (-not $msbuildPath -or -not (Test-Path $msbuildPath)) {
    Write-Host "ERROR: No se pudo encontrar MSBuild" -ForegroundColor Red
    Write-Host "Asegúrate de tener Visual Studio 2022 instalado" -ForegroundColor Yellow
    exit 1
}

Write-Host "MSBuild encontrado en: $msbuildPath" -ForegroundColor Green
Write-Host ""

# Limpiar compilaciones anteriores
Write-Host "Limpiando compilaciones anteriores..." -ForegroundColor Yellow
Remove-Item -Path "bin" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path "obj" -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -Path ".vs" -Recurse -Force -ErrorAction SilentlyContinue

# Restaurar paquetes NuGet
Write-Host "Restaurando paquetes NuGet..." -ForegroundColor Yellow
& $msbuildPath /t:Restore /p:Configuration=$Configuration

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo al restaurar paquetes NuGet" -ForegroundColor Red
    exit 1
}

# Compilar el proyecto
Write-Host ""
Write-Host "Compilando WabiSabi Bridge..." -ForegroundColor Yellow
& $msbuildPath /p:Configuration=$Configuration /p:Platform=x64

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Fallo al compilar el proyecto" -ForegroundColor Red
    exit 1
}

Write-Host "Compilación exitosa!" -ForegroundColor Green
Write-Host ""

# Instalar en Revit
$addinPath = "$env:APPDATA\Autodesk\Revit\Addins\$RevitVersion"
Write-Host "Instalando en: $addinPath" -ForegroundColor Yellow

# Crear directorio si no existe
if (-not (Test-Path $addinPath)) {
    New-Item -ItemType Directory -Path $addinPath -Force | Out-Null
}

# Copiar archivos
$outputPath = "bin\$Configuration"
Copy-Item "$outputPath\WabiSabiBridge.dll" -Destination $addinPath -Force
Copy-Item "WabiSabiBridge.addin" -Destination $addinPath -Force

# Copiar dependencias
$dependencies = @(
    "Newtonsoft.Json.dll"
)

foreach ($dep in $dependencies) {
    $depPath = "$outputPath\$dep"
    if (Test-Path $depPath) {
        Copy-Item $depPath -Destination $addinPath -Force
        Write-Host "  - Copiado: $dep" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "=== Instalación completada! ===" -ForegroundColor Green
Write-Host ""
Write-Host "Próximos pasos:" -ForegroundColor Cyan
Write-Host "1. Abre Revit $RevitVersion"
Write-Host "2. Carga un proyecto con vista 3D"
Write-Host "3. Ve a la pestaña 'WabiSabi' en el ribbon"
Write-Host "4. Haz clic en 'WabiSabi Bridge'"
Write-Host ""
Write-Host "La carpeta de salida por defecto es:" -ForegroundColor Yellow
Write-Host "  $env:USERPROFILE\Documents\WabiSabiBridge" -ForegroundColor White
Write-Host ""

# Preguntar si abrir la carpeta de addins
$response = Read-Host "¿Deseas abrir la carpeta de Addins? (S/N)"
if ($response -eq 'S' -or $response -eq 's') {
    Start-Process explorer.exe $addinPath
}

Write-Host ""
Write-Host "Script completado." -ForegroundColor Green