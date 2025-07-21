@echo off
setlocal enabledelayedexpansion
REM BuildClean.bat - v1.1
REM Compila e instala WabiSabi Bridge para todas las versiones de Revit compatibles.
REM Mejorado para mayor robustez y claridad.

echo ======================================================
echo  WabiSabi Bridge v0.3.0 - Script de Compilacion e Instalacion
echo ======================================================
echo.

REM --- 1. NAVEGAR AL DIRECTORIO DE FUENTES ---
cd /d "%~dp0\..\src"
if not exist "WabiSabiBridge.csproj" (
    echo ! ERROR: No se encuentra el archivo 'WabiSabiBridge.csproj'.
    echo   Asegurate de que el script esta en la carpeta 'scripts' junto a 'src'.
    goto :error
)

REM --- 2. LIMPIEZA ---
echo [1] Limpiando artefactos de compilaciones anteriores...
if exist "bin" rmdir /s /q "bin"
if exist "obj" rmdir /s /q "obj"
echo   OK - Directorios 'bin' y 'obj' limpios.
echo.

REM --- 3. COMPILACION ---
echo [2] Iniciando proceso de compilacion...
echo.
echo   [2.1] Intentando compilacion con ACELERACION GPU (usando ComputeSharp)...
dotnet build -c Release -p:EnableGPU=true -v minimal > build_gpu.log 2>&1

if %ERRORLEVEL% equ 0 (
    echo   OK - Compilacion con GPU exitosa.
    del build_gpu.log 2>nul
    goto :find_artifacts
)

echo   ADVERTENCIA: La compilacion con GPU fallo. Reintentando en modo CPU.
echo   (Revisa 'build_gpu.log' para mas detalles si el error persiste)
echo.
echo   [2.2] Intentando compilacion con CPU PARALELA OPTIMIZADA...
dotnet build -c Release -p:EnableGPU=false -v minimal > build_cpu.log 2>&1

if %ERRORLEVEL% neq 0 (
    echo.
    echo  ! ERROR CRITICO: No se pudo compilar el proyecto en ningun modo.
    echo  Revisa 'build_cpu.log' para entender la causa del fallo.
    goto :error
)

echo   OK - Compilacion de respaldo (CPU Paralela) exitosa.
del build_cpu.log 2>nul
echo.

REM --- 4. BUSQUEDA DE ARTEFACTOS ---
:find_artifacts
echo [3] Buscando artefactos de compilacion...
set "DLL_SOURCE_PATH="

REM Usar un bucle recursivo para encontrar el directorio que contiene la DLL.
for /r "bin\Release" %%f in (WabiSabiBridge.dll) do (
    if not defined DLL_SOURCE_PATH (
        set "DLL_SOURCE_PATH=%%~dpf"
    )
)

if not defined DLL_SOURCE_PATH (
    echo  ! ERROR: No se encontro WabiSabiBridge.dll en la carpeta 'bin\Release'.
    echo    La compilacion pudo haber fallado o la ruta es inesperada.
    goto :error
)
echo   - Artefactos encontrados en: !DLL_SOURCE_PATH!
echo.

REM --- 5. INSTALACION ---
:install
echo [4] Instalando el plugin en las versiones de Revit encontradas...

REM Comprobar que el archivo .addin existe ANTES de empezar a copiar
if not exist "WabiSabiBridge.addin" (
    echo ! ERROR: El archivo de manifiesto 'WabiSabiBridge.addin' no se encuentra.
    echo   Debe estar en la misma carpeta que los archivos fuente C#.
    goto :error
)

set "INSTALLED_COUNT=0"
REM Iterar sobre las versiones comunes de Revit para instalar en todas
for %%V in (2024, 2025, 2026, 2027, 2028) do (
    set "REVIT_INSTALL_DIR=%ProgramFiles%\Autodesk\Revit %%V"
    set "ADDIN_DEST_PATH=%APPDATA%\Autodesk\Revit\Addins\%%V"
    
    if exist "!REVIT_INSTALL_DIR!\" (
        echo   - Detectada instalacion de Revit %%V.
        
        if not exist "!ADDIN_DEST_PATH!\" (
            echo     Creando directorio de Addins que faltaba...
            mkdir "!ADDIN_DEST_PATH!"
        )
        
        echo     Copiando archivos a: !ADDIN_DEST_PATH!
        
        REM MEJORA: Usar xcopy es mas robusto para copiar todos los artefactos de compilacion.
        xcopy "!DLL_SOURCE_PATH!" "!ADDIN_DEST_PATH!\" /Y /I /E /Q /R >nul
        
        REM Copiar el manifiesto .addin por separado
        copy /Y "WabiSabiBridge.addin" "!ADDIN_DEST_PATH!\" >nul
        
        set /a INSTALLED_COUNT+=1
    )
)

if %INSTALLED_COUNT% equ 0 (
    echo.
    echo  ! ADVERTENCIA: No se encontro ninguna instalacion de Revit compatible (2024-2028).
    echo  El plugin fue compilado pero no instalado.
    echo  Puedes copiar los archivos manualmente desde '!DLL_SOURCE_PATH!'
) else (
    echo.
    echo   OK - Plugin instalado en %INSTALLED_COUNT% version(es) de Revit.
)
echo.

REM --- 6. RESUMEN FINAL ---
:summary
echo [5] Limpiando archivos de registro temporales...
del build_gpu.log 2>nul
del build_cpu.log 2>nul
echo.

echo ======================================================
echo      OPERACION COMPLETADA EXITOSAMENTE!
echo ======================================================
echo.

if exist "!DLL_SOURCE_PATH!ComputeSharp.dll" (
    echo   Modo Compilado: ACELERACION GPU (Rendimiento optimo)
) else (
    echo   Modo Compilado: CPU PARALELO (Rendimiento mejorado)
)
echo.
echo   Por favor, reinicia Revit para que los cambios surtan efecto.
echo.
goto :end

:error
echo.
echo  ! La operacion ha fallado. Revisa los mensajes de error.
echo.
pause
exit /b 1

:end
endlocal
pause
exit /b 0