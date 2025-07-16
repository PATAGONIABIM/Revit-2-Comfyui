@echo off
REM Uninstall.bat - Desinstala WabiSabi Bridge de Revit

echo === Desinstalador de WabiSabi Bridge ===
echo.

set ADDIN_PATH=%APPDATA%\Autodesk\Revit\Addins\2026

echo Este script eliminara WabiSabi Bridge de:
echo %ADDIN_PATH%
echo.

set /p CONFIRM=Estas seguro? (S/N): 
if /i not "%CONFIRM%"=="S" (
    echo Cancelado.
    pause
    exit /b 0
)

echo.
echo Eliminando archivos...

if exist "%ADDIN_PATH%\WabiSabiBridge.dll" (
    del /f "%ADDIN_PATH%\WabiSabiBridge.dll"
    echo  - Eliminado: WabiSabiBridge.dll
)

if exist "%ADDIN_PATH%\WabiSabiBridge.addin" (
    del /f "%ADDIN_PATH%\WabiSabiBridge.addin"
    echo  - Eliminado: WabiSabiBridge.addin
)

if exist "%ADDIN_PATH%\Newtonsoft.Json.dll" (
    echo.
    echo Nota: Newtonsoft.Json.dll puede ser usado por otros plugins
    set /p DEL_JSON=Eliminar Newtonsoft.Json.dll tambien? (S/N): 
    if /i "%DEL_JSON%"=="S" (
        del /f "%ADDIN_PATH%\Newtonsoft.Json.dll"
        echo  - Eliminado: Newtonsoft.Json.dll
    )
)

echo.
echo === Desinstalacion completada ===
echo.
echo El plugin ha sido removido de Revit 2026
echo.
pause