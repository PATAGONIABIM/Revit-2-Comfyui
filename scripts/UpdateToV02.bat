@echo off
REM UpdateToV02.bat - Actualiza WabiSabi Bridge a la version 0.2 con mapa de profundidad

echo === Actualizacion a WabiSabi Bridge v0.2 ===
echo.
echo Esta actualizacion agrega:
echo - Generacion de mapas de profundidad
echo - Resoluciones configurables (256-2048)
echo - Interfaz mejorada
echo.

cd /d "%~dp0\..\src"

REM 1. Crear estructura de carpetas
echo [1] Creando estructura de carpetas...
if not exist "Extractors" (
    mkdir Extractors
    echo  OK - Carpeta Extractors creada
) else (
    echo  OK - Carpeta Extractors ya existe
)

REM 2. Verificar archivos
echo.
echo [2] Verificando archivos...
echo.
echo IMPORTANTE: Asegurate de haber guardado:
echo - WabiSabiBridge.cs (actualizado) en src\
echo - DepthExtractor.cs (nuevo) en src\Extractors\
echo.
pause

if not exist "Extractors\DepthExtractor.cs" (
    echo.
    echo ERROR: No se encuentra DepthExtractor.cs en src\Extractors\
    echo Por favor, guarda el archivo y ejecuta este script de nuevo
    pause
    exit /b 1
)

REM 3. Limpiar y compilar
echo.
echo [3] Compilando version 0.2...
echo.

REM Limpiar
if exist bin rmdir /s /q bin 2>nul
if exist obj rmdir /s /q obj 2>nul

REM Compilar
dotnet build -c Release -v minimal

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Fallo la compilacion
    echo Ejecuta CleanBuild.bat para mas detalles
    pause
    exit /b 1
)

REM 4. Instalar
echo.
echo [4] Instalando actualizacion...
set ADDIN_PATH=%APPDATA%\Autodesk\Revit\Addins\2026
if not exist "%ADDIN_PATH%" mkdir "%ADDIN_PATH%"

set FOUND=0
for /r "bin" %%f in (WabiSabiBridge.dll) do (
    if exist "%%f" (
        copy /Y "%%f" "%ADDIN_PATH%\" >nul
        copy /Y "WabiSabiBridge.addin" "%ADDIN_PATH%\" >nul
        if exist "%%~dpfNewtonsoft.Json.dll" (
            copy /Y "%%~dpfNewtonsoft.Json.dll" "%ADDIN_PATH%\" >nul
        )
        set FOUND=1
        echo  OK - Plugin actualizado
        goto :done
    )
)

:done
if %FOUND%==0 (
    echo ERROR: No se encontro WabiSabiBridge.dll
    pause
    exit /b 1
)

echo.
echo === Actualizacion completada! ===
echo.
echo Novedades en v0.2:
echo - Checkbox "Generar mapa de profundidad" en la interfaz
echo - Selector de resolucion (256, 512, 1024, 2048)
echo - Archivo de salida: current_depth.png
echo.
echo Para usar:
echo 1. Abre Revit 2026
echo 2. Ejecuta WabiSabi Bridge
echo 3. Activa "Generar mapa de profundidad"
echo 4. Selecciona resolucion y exporta
echo.
pause