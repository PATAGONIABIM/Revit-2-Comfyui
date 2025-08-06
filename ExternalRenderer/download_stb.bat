@echo off
echo Descargando stb_image_write.h...

cd C:\WabiSabiRevitBridge\ExternalRenderer\src

if not exist external mkdir external
cd external

echo Descargando archivo...
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h' -OutFile 'stb_image_write.h'"

if exist stb_image_write.h (
    echo Descarga exitosa!
) else (
    echo Error en la descarga!
)

cd ..\..\..
pause