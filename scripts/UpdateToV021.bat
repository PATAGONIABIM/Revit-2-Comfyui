@echo off
REM UpdateToV021.bat - Actualiza WabiSabi Bridge a v0.2.1 con correcciones de profundidad

echo === Actualizacion a WabiSabi Bridge v0.2.1 ===
echo.
echo Esta actualizacion corrige:
echo - Mapas de profundidad ahora con escalas de grises correctas
echo - Rendimiento mejorado hasta 16x con modo rapido
echo - Selector de calidad: Rapida, Normal, Alta
echo - Tiempo estimado en la interfaz
echo.

cd /d "%~dp0\..\src"

REM 1. Verificar estructura
echo [1] Verificando estructura...
if not exist "Extractors" (
    mkdir Extractors
    echo  - Carpeta Extractors creada
)

REM 2. Verificar archivos
echo.
echo [2] Archivos necesarios:
echo - WabiSabiBridge.cs (actualizado)
echo - Extractors\DepthExtractor.cs (actualizado)
echo - Extractors\DepthExtractorFast.cs (NUEVO)
echo.
pause

REM 3. Verificar que el nuevo archivo existe
if not exist "Extractors\DepthExtractorFast.cs" (
    echo.
    echo ERROR: No se encuentra DepthExtractorFast.cs
    echo Por favor, guarda el archivo en src\Extractors\
    pause
    exit /b 1
)

REM 4. Limpiar completamente
echo.
echo [3] Limpiando proyecto...
if exist bin rmdir /s /q bin 2>nul
if exist obj rmdir /s /q obj 2>nul
if exist .vs rmdir /s /q .vs 2>nul

REM 5. Compilar
echo.
echo [4] Compilando v0.2.1...
dotnet restore --force
dotnet build -c Release -v minimal

if %ERRORLEVEL% neq 0 (
    echo.
    echo ERROR: Fallo la compilacion
    echo Revisa que todos los archivos esten actualizados
    pause
    exit /b 1
)

REM 6. Instalar
echo.
echo [5] Instalando...
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
        echo  OK - Plugin actualizado a v0.2.1
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
echo MEJORAS en v0.2.1:
echo.
echo 1. MAPAS DE PROFUNDIDAD CORREGIDOS:
echo    - Ahora con gradientes correctos (no solo blanco/negro)
echo    - Objetos cercanos = blanco, lejanos = negro
echo.
echo 2. RENDIMIENTO MEJORADO:
echo    - Modo Rapido: 512x512 en ~3 segundos
echo    - Modo Normal: Balance calidad/velocidad
echo    - Modo Alta: Maxima calidad
echo.
echo 3. NUEVA INTERFAZ:
echo    - Selector de calidad
echo    - Tiempo estimado visible
echo.
echo Recomendacion: Prueba primero con 512x512 en modo Rapido
echo.
pause