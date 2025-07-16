@echo off
REM FixVersionBuild.bat - Compilación que resuelve conflictos de versiones

cd /d "%~dp0\..\src"

echo === WabiSabi Bridge - Compilacion con Fix de Versiones ===
echo.

echo [1] Limpiando todo completamente...
if exist bin rmdir /s /q bin 2>nul
if exist obj rmdir /s /q obj 2>nul
if exist .vs rmdir /s /q .vs 2>nul
del *.user 2>nul

REM Limpiar caché de NuGet local
if exist "%USERPROFILE%\.nuget\packages\wabisabibridge" rmdir /s /q "%USERPROFILE%\.nuget\packages\wabisabibridge" 2>nul

echo [2] Restaurando paquetes con limpieza...
dotnet restore --force --no-cache

echo.
echo [3] Compilando con configuracion especial...
dotnet build -c Release /p:Platform=x64 /p:CopyLocalLockFileAssemblies=false -v minimal

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: La compilacion fallo
    pause
    exit /b 1
)

echo.
echo [4] Preparando archivos para instalacion...
set OUTPUT_PATH=""
if exist "bin\Release\net48" set OUTPUT_PATH=bin\Release\net48
if exist "bin\Release" set OUTPUT_PATH=bin\Release
if exist "bin\x64\Release" set OUTPUT_PATH=bin\x64\Release

if "%OUTPUT_PATH%"=="" (
    echo ERROR: No se encontro la carpeta de salida
    pause
    exit /b 1
)

echo Carpeta de salida: %OUTPUT_PATH%

REM Verificar que NO se copiaron las DLLs del sistema
echo.
echo [5] Verificando que no hay DLLs conflictivas...
if exist "%OUTPUT_PATH%\System.Windows.Forms.dll" (
    echo  ADVERTENCIA: Encontrada System.Windows.Forms.dll - eliminando...
    del /f "%OUTPUT_PATH%\System.Windows.Forms.dll"
)
if exist "%OUTPUT_PATH%\System.Drawing.dll" (
    echo  ADVERTENCIA: Encontrada System.Drawing.dll - eliminando...
    del /f "%OUTPUT_PATH%\System.Drawing.dll"
)

echo.
echo [6] Instalando en Revit...
set ADDIN_PATH=%APPDATA%\Autodesk\Revit\Addins\2026
if not exist "%ADDIN_PATH%" mkdir "%ADDIN_PATH%"

REM Copiar solo los archivos necesarios
copy /Y "%OUTPUT_PATH%\WabiSabiBridge.dll" "%ADDIN_PATH%\" >nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: No se pudo copiar WabiSabiBridge.dll
    pause
    exit /b 1
)
echo  OK - WabiSabiBridge.dll

copy /Y "WabiSabiBridge.addin" "%ADDIN_PATH%\" >nul
echo  OK - WabiSabiBridge.addin

if exist "%OUTPUT_PATH%\Newtonsoft.Json.dll" (
    copy /Y "%OUTPUT_PATH%\Newtonsoft.Json.dll" "%ADDIN_PATH%\" >nul
    echo  OK - Newtonsoft.Json.dll
)

echo.
echo === Compilacion e instalacion exitosa! ===
echo.
echo IMPORTANTE: Las DLLs del sistema NO se copiaron para evitar conflictos
echo Revit usara sus propias versiones de System.Windows.Forms y System.Drawing
echo.
echo Plugin instalado en:
echo %ADDIN_PATH%
echo.
pause