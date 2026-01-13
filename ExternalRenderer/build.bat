@echo off
setlocal

echo ============================================
echo WabiSabi External Renderer - Smart Build
echo ============================================

:: --- RUTA AL TOOLCHAIN DE VCPKG ---
:: Asegúrate de que esta ruta a tu instalación de vcpkg sea correcta.
set VCPKG_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

if not exist "%VCPKG_TOOLCHAIN_FILE%" (
    echo.
    echo X ERROR: No se encontro el archivo toolchain de vcpkg.
    echo   Ruta buscada: %VCPKG_TOOLCHAIN_FILE%
    echo   Por favor, instala vcpkg o corrige la ruta en build.bat.
    pause
    exit /b 1
)

:: Limpiar build anterior
if exist build rmdir /s /q build
mkdir build
cd build

:: Configurar con CMake, pasándole el toolchain
echo.
echo ============================================
echo Configurando con CMake...
echo ============================================
cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_TOOLCHAIN_FILE="%VCPKG_TOOLCHAIN_FILE%" -T cuda="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

if %errorlevel% neq 0 (
    echo.
    echo X Error en configuracion CMake. Revisa los mensajes.
    cd ..
    pause
    exit /b 1
)

:: Compilar
echo.
echo ============================================
echo Compilando...
echo ============================================
cmake --build . --config Release --parallel

if %errorlevel% neq 0 (
    echo.
    echo X Error en compilacion. Revisa los mensajes.
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo ============================================
echo V Compilacion exitosa!
echo ============================================
echo Ejecutable: build\Release\WabiSabiRenderer.exe
echo.
pause