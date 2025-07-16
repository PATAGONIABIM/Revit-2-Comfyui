# WabiSabi Bridge v0.2 - Actualización con Mapa de Profundidad

## 🎉 Novedades en v0.2

### ✨ Nueva característica: Generación de Mapa de Profundidad

- **Mapa de profundidad en escala de grises** - Los objetos cercanos aparecen blancos, los lejanos negros
- **Resoluciones configurables**: 256, 512, 1024, 2048 píxeles
- **Procesamiento optimizado** con raytracing usando ReferenceIntersector
- **Archivo de salida**: `current_depth.png`

## 📁 Nueva estructura del proyecto

```
C:\WabiSabiRevitBridge\
├── 📁 src/
│   ├── 📄 WabiSabiBridge.cs        # Archivo principal actualizado
│   ├── 📁 Extractors/              # NUEVA carpeta
│   │   └── 📄 DepthExtractor.cs    # NUEVO extractor de profundidad
│   ├── 📄 WabiSabiBridge.csproj
│   └── 📄 WabiSabiBridge.addin
└── 📁 scripts/
    └── ...todos los scripts
```

## 🛠️ Para compilar e instalar v0.2:

### 1. Crear la estructura de carpetas:
```batch
cd C:\WabiSabiRevitBridge\src
mkdir Extractors
```

### 2. Colocar archivos:
- Guardar `WabiSabiBridge.cs` actualizado en `src\`
- Guardar `DepthExtractor.cs` en `src\Extractors\`

### 3. Compilar:
```batch
cd C:\WabiSabiRevitBridge\scripts
CleanBuild.bat
```

## 🎮 Uso de la nueva característica

1. **Abrir Revit** y cargar un modelo
2. **Activar vista 3D**
3. **Ejecutar WabiSabi Bridge**
4. **Activar "Generar mapa de profundidad"**
5. **Seleccionar resolución** (512 por defecto)
6. **Exportar**

### Archivos generados:

```
WabiSabiBridge/
├── current_render.png      # Vista con líneas ocultas
├── current_depth.png       # NUEVO - Mapa de profundidad
├── current_metadata.json   # Metadatos
└── last_update.txt        # Timestamp
```

## 🖼️ Ejemplo de mapa de profundidad

- **Blanco (255)**: Objetos muy cercanos a la cámara
- **Gris (128)**: Objetos a distancia media
- **Negro (0)**: Objetos lejanos o fondo

## ⚡ Rendimiento

| Resolución | Tiempo aproximado* |
|------------|-------------------|
| 256x256    | 2-5 segundos     |
| 512x512    | 8-15 segundos    |
| 1024x1024  | 30-60 segundos   |
| 2048x2048  | 2-5 minutos      |

*Depende de la complejidad del modelo

## 🔧 Solución de problemas

### El mapa de profundidad es todo negro o todo blanco
- Verifica que la vista 3D tenga geometría visible
- Intenta con una resolución menor primero
- Asegúrate de que no hay section box muy restrictivo

### Error al generar profundidad
- El plugin continuará exportando la imagen normal
- Aparecerá una advertencia en naranja
- Revisa que la vista sea 3D (no 2D)

## 🚀 Integración con ComfyUI

En ComfyUI puedes usar:
- `current_render.png` - Como imagen base
- `current_depth.png` - Como mapa de profundidad para ControlNet Depth

Ejemplo de workflow:
1. Load Image → `current_render.png`
2. Load Image → `current_depth.png`
3. ControlNet (depth) → Usar el mapa de profundidad
4. KSampler → Generar imagen con profundidad consistente

## 📝 Configuración guardada

La configuración se guarda automáticamente en:
```
%APPDATA%\WabiSabiBridge\config.json
```

Incluye:
- Ruta de salida
- Estado de exportación automática
- **NUEVO**: Estado de exportación de profundidad
- **NUEVO**: Resolución de profundidad

## 🔮 Próximas características planeadas

- [ ] Mapa de segmentación por categorías
- [ ] Detección inteligente de cambios
- [ ] Exportación de normales
- [ ] Barra de progreso durante exportación
- [ ] Posición real de cámara en metadatos

---

**WabiSabi Bridge v0.2** - Con soporte de mapa de profundidad