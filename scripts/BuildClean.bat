@echo off
setlocal enabledelayedexpansion
REM BuildClean_Fixed.bat - v1.2
REM Compila e instala WabiSabi Bridge con soporte DirectContext3D mejorado
REM Corrige problemas de exportación automática

echo ======================================================
echo  WabiSabi Bridge v0.3.3 Fixed - Script de Compilacion e Instalacion
echo ======================================================
echo.

REM --- 1. NAVEGAR AL DIRECTORIO DE FUENTES ---
cd /d "%~dp0"
if exist "src" (
    cd src
) else if exist "..\src" (
    cd ..\src
) else (
    echo ! ERROR: No se encuentra el directorio 'src'.
    echo   Asegurate de ejecutar este script desde la carpeta correcta.
    goto :error
)

if not exist "WabiSabiBridge.csproj" (
    echo ! ERROR: No se encuentra el archivo 'WabiSabiBridge.csproj'.
    goto :error
)

REM --- 2. LIMPIEZA ---
echo [1] Limpiando artefactos de compilaciones anteriores...
if exist "bin" rmdir /s /q "bin"
if exist "obj" rmdir /s /q "obj"
echo   OK - Directorios 'bin' y 'obj' limpios.
echo.

REM --- 3. RESTAURAR PAQUETES NUGET ---
echo [2] Restaurando paquetes NuGet...
dotnet restore > restore.log 2>&1
if %ERRORLEVEL% neq 0 (
    echo   ERROR: Fallo al restaurar paquetes NuGet.
    echo   Revisa 'restore.log' para mas detalles.
    goto :error
)
del restore.log 2>nul
echo   OK - Paquetes restaurados.
echo.

REM --- 4. COMPILACION ---
echo [3] Iniciando proceso de compilacion...
echo.
echo   [3.1] Intentando compilacion con ACELERACION GPU (usando ComputeSharp)...
dotnet build -c Release -p:EnableGPU=true -v minimal > build_gpu.log 2>&1

if %ERRORLEVEL% equ 0 (
    echo   OK - Compilacion con GPU exitosa.
    del build_gpu.log 2>nul
    goto :find_artifacts
)

echo   ADVERTENCIA: La compilacion con GPU fallo. Reintentando en modo CPU.
echo   (Revisa 'build_gpu.log' para mas detalles si el error persiste)
echo.
echo   [3.2] Intentando compilacion con CPU PARALELA OPTIMIZADA...
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

REM --- 5. BUSQUEDA DE ARTEFACTOS ---
:find_artifacts
echo [4] Buscando artefactos de compilacion...
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

REM --- 6. CREAR ARCHIVO .ADDIN SI NO EXISTE ---
echo [5] Verificando archivo de manifiesto .addin...
if not exist "WabiSabiBridge.addin" (
    echo   - Creando archivo WabiSabiBridge.addin...
    (
        echo ^<?xml version="1.0" encoding="utf-8"?^>
        echo ^<RevitAddIns^>
        echo   ^<AddIn Type="Application"^>
        echo     ^<Name^>WabiSabi Bridge^</Name^>
        echo     ^<Assembly^>WabiSabiBridge.dll^</Assembly^>
        echo     ^<FullClassName^>WabiSabiBridge.WabiSabiBridgeApp^</FullClassName^>
        echo     ^<ClientId^>8b8f8e8a-4e4e-4e4e-8e8e-8e8e8e8e8e8e^</ClientId^>
        echo     ^<VendorId^>WabiSabi^</VendorId^>
        echo     ^<VendorDescription^>WabiSabi Bridge - High Performance Export Plugin^</VendorDescription^>
        echo   ^</AddIn^>
        echo ^</RevitAddIns^>
    ) > WabiSabiBridge.addin
    echo   OK - Archivo .addin creado.
) else (
    echo   OK - Archivo .addin existente encontrado.
)
echo.

REM --- 7. INSTALACION ---
:install
echo [6] Instalando el plugin en las versiones de Revit encontradas...

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
        
        REM Copiar todos los archivos de compilacion
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

REM --- 8. RESUMEN FINAL ---
:summary
echo [7] Limpiando archivos de registro temporales...
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
echo   Version: v0.3.3 Fixed (Corrige exportacion automatica)
echo.
echo   Por favor, reinicia Revit para que los cambios surtan efecto.
echo.
echo   NOTAS DE LA VERSION:
echo   - Corregido: El servidor DirectContext3D ahora usa OnDrawContext
echo   - Mejorado: Sistema de captura continua mas agresivo
echo   - Agregado: Mas logging para diagnostico
echo   - Corregido: Deteccion de movimiento de camara
echo.
goto :end

:error
echo.
echo  ! La operacion ha fallado. Revisa los mensajes de error.
echo.
pause
exit /b 1

:end
echo Presiona cualquier tecla para cerrar...
pause > nul
exit /b 0