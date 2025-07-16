@echo off
REM Build.bat - Script de compilación para WabiSabi Bridge
REM No requiere permisos especiales de PowerShell

echo === WabiSabi Bridge - Compilacion e Instalacion ===
echo.

REM Verificar que estamos en el directorio correcto
if not exist "WabiSabiBridge.csproj" (
    echo ERROR: No se encuentra WabiSabiBridge.csproj
    echo Ejecuta este script desde la carpeta src del proyecto
    pause
    exit /b 1
)

REM Buscar MSBuild
set MSBUILD_PATH=""
set VS_PATH=%ProgramFiles%\Microsoft Visual Studio\2022

REM Intentar Community Edition
if exist "%VS_PATH%\Community\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD_PATH="%VS_PATH%\Community\MSBuild\Current\Bin\MSBuild.exe"
    goto :found_msbuild
)

REM Intentar Professional Edition
if exist "%VS_PATH%\Professional\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD_PATH="%VS_PATH%\Professional\MSBuild\Current\Bin\MSBuild.exe"
    goto :found_msbuild
)

REM Intentar Enterprise Edition
if exist "%VS_PATH%\Enterprise\MSBuild\Current\Bin\MSBuild.exe" (
    set MSBUILD_PATH="%VS_PATH%\Enterprise\MSBuild\Current\Bin\MSBuild.exe"
    goto :found_msbuild
)

REM Si no se encuentra MSBuild
echo ERROR: No se pudo encontrar MSBuild
echo Asegurate de tener Visual Studio 2022 instalado
pause
exit /b 1

:found_msbuild
echo MSBuild encontrado!
echo.

REM Limpiar compilaciones anteriores
echo Limpiando compilaciones anteriores...
if exist "bin" rmdir /s /q "bin"
if exist "obj" rmdir /s /q "obj"

REM Restaurar paquetes NuGet
echo Restaurando paquetes NuGet...
%MSBUILD_PATH% /t:Restore /p:Configuration=Release

if %ERRORLEVEL% neq 0 (
    echo ERROR: Fallo al restaurar paquetes NuGet
    pause
    exit /b 1
)

REM Compilar el proyecto
echo.
echo Compilando WabiSabi Bridge...
%MSBUILD_PATH% /p:Configuration=Release /p:Platform=x64

if %ERRORLEVEL% neq 0 (
    echo ERROR: Fallo al compilar el proyecto
    pause
    exit /b 1
)

echo.
echo Compilacion exitosa!
echo.

REM Instalar en Revit
set ADDIN_PATH=%APPDATA%\Autodesk\Revit\Addins\2026
echo Instalando en: %ADDIN_PATH%

REM Crear directorio si no existe
if not exist "%ADDIN_PATH%" mkdir "%ADDIN_PATH%"

REM Copiar archivos
echo Copiando archivos...
copy /Y "bin\Release\WabiSabiBridge.dll" "%ADDIN_PATH%\" > nul
copy /Y "WabiSabiBridge.addin" "%ADDIN_PATH%\" > nul

REM Copiar dependencias
if exist "bin\Release\Newtonsoft.Json.dll" (
    copy /Y "bin\Release\Newtonsoft.Json.dll" "%ADDIN_PATH%\" > nul
    echo   - Copiado: Newtonsoft.Json.dll
)

echo.
echo === Instalacion completada! ===
echo.
echo Proximos pasos:
echo 1. Abre Revit 2026
echo 2. Carga un proyecto con vista 3D
echo 3. Ve a la pestana 'WabiSabi' en el ribbon
echo 4. Haz clic en 'WabiSabi Bridge'
echo.
echo La carpeta de salida por defecto es:
echo   %USERPROFILE%\Documents\WabiSabiBridge
echo.

REM Preguntar si abrir la carpeta de addins
set /p OPEN_FOLDER=Deseas abrir la carpeta de Addins? (S/N): 
if /i "%OPEN_FOLDER%"=="S" (
    start "" "%ADDIN_PATH%"
)

echo.
echo Script completado.
pause