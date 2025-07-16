@echo off
REM CleanBuild.bat - Compilación limpia desde cero

cd /d "%~dp0\..\src"

echo === WabiSabi Bridge - Compilacion Limpia ===
echo.

echo [1] Limpiando todo...
if exist bin rmdir /s /q bin 2>nul
if exist obj rmdir /s /q obj 2>nul
if exist .vs rmdir /s /q .vs 2>nul
del *.user 2>nul

echo [2] Restaurando paquetes...
dotnet restore --force

echo.
echo [3] Compilando...
dotnet build -c Release /p:Platform=x64 -v minimal

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: La compilacion fallo
    echo Intenta abrir el proyecto en Visual Studio para mas detalles
    pause
    exit /b 1
)

echo.
echo [4] Instalando...
set ADDIN_PATH=%APPDATA%\Autodesk\Revit\Addins\2026
if not exist "%ADDIN_PATH%" mkdir "%ADDIN_PATH%"

REM Buscar la DLL
set DLL_FOUND=0
for /r "bin" %%f in (WabiSabiBridge.dll) do (
    if exist "%%f" (
        echo Copiando archivos desde: %%~dpf
        copy /Y "%%f" "%ADDIN_PATH%\" >nul
        copy /Y "WabiSabiBridge.addin" "%ADDIN_PATH%\" >nul
        
        REM Copiar dependencias
        if exist "%%~dpfNewtonsoft.Json.dll" (
            copy /Y "%%~dpfNewtonsoft.Json.dll" "%ADDIN_PATH%\" >nul
        )
        
        set DLL_FOUND=1
        goto :done
    )
)

:done
if %DLL_FOUND%==0 (
    echo ERROR: No se encontro WabiSabiBridge.dll
    echo Revisa la carpeta bin\
    pause
    exit /b 1
)

echo.
echo === Compilacion e instalacion exitosa! ===
echo.
echo Plugin instalado en:
echo %ADDIN_PATH%
echo.
echo Archivos instalados:
dir /b "%ADDIN_PATH%\WabiSabi*.*" 2>nul
if exist "%ADDIN_PATH%\Newtonsoft.Json.dll" echo Newtonsoft.Json.dll
echo.
pause