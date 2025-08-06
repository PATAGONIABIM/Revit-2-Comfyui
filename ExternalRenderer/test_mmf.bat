@echo off
echo ========================================
echo Verificando Memory Mapped Files
echo ========================================
echo.

REM Usar PowerShell para listar MMFs
powershell -Command "Get-CimInstance -ClassName Win32_PerfRawData_PerfProc_Process | Where-Object {$_.Name -like '*WabiSabi*'} | Select-Object Name, HandleCount"

echo.
echo Buscando archivos de estado...
if exist "%LOCALAPPDATA%\WabiSabiBridge\GeometryCache\wabisabi_state.json" (
    echo Archivo de estado encontrado:
    type "%LOCALAPPDATA%\WabiSabiBridge\GeometryCache\wabisabi_state.json"
) else (
    echo No se encontro archivo de estado
)

echo.
echo Buscando archivos de cache persistente...
dir "%LOCALAPPDATA%\WabiSabiBridge\GeometryCache\*.wabi_geom" 2>nul

echo.
pause