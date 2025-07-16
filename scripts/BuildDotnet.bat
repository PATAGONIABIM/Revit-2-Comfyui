@echo off
REM BuildDotnet.bat - Script simplificado usando dotnet CLI
REM Alternativa más simple para compilar

cd /d "%~dp0\..\src"

echo === WabiSabi Bridge - Compilacion con dotnet CLI ===
echo.

REM Verificar que dotnet está instalado
where dotnet >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: dotnet CLI no está instalado
    echo Descarga .NET SDK desde: https://dotnet.microsoft.com/download
    pause
    exit /b 1
)

REM Verificar archivos
if not exist "WabiSabiBridge.csproj" (
    echo ERROR: No se encuentra WabiSabiBridge.csproj
    echo Verifica que estás en la carpeta correcta
    pause
    exit /b 1
)

REM Limpiar
echo Limpiando...
dotnet clean -c Release 2>nul

REM Restaurar paquetes
echo Restaurando paquetes NuGet...
dotnet restore

if %ERRORLEVEL% neq 0 (
    echo ERROR: Fallo al restaurar paquetes
    pause
    exit /b 1
)

REM Compilar
echo.
echo Compilando...
dotnet build -c Release

if %ERRORLEVEL% neq 0 (
    echo ERROR: Fallo la compilacion
    pause
    exit /b 1
)

echo.
echo === Compilacion exitosa! ===
echo.

REM Instalar
set ADDIN_PATH=%APPDATA%\Autodesk\Revit\Addins\2026
echo Instalando en: %ADDIN_PATH%

if not exist "%ADDIN_PATH%" mkdir "%ADDIN_PATH%"

REM Buscar DLL
set DLL_FOUND=0
for /r "bin" %%f in (WabiSabiBridge.dll) do (
    if exist "%%f" (
        echo Copiando desde: %%~dpf
        copy /Y "%%f" "%ADDIN_PATH%\" >nul
        copy /Y "WabiSabiBridge.addin" "%ADDIN_PATH%\" >nul
        
        REM Copiar Newtonsoft.Json.dll si existe
        if exist "%%~dpfNewtonsoft.Json.dll" (
            copy /Y "%%~dpfNewtonsoft.Json.dll" "%ADDIN_PATH%\" >nul
        )
        
        set DLL_FOUND=1
        goto :install_done
    )
)

:install_done
if %DLL_FOUND%==0 (
    echo ERROR: No se encontro WabiSabiBridge.dll
    pause
    exit /b 1
)

echo.
echo Instalacion completada!
echo.
echo Abre Revit 2026 y busca la pestana 'WabiSabi' en el ribbon
echo.
pause