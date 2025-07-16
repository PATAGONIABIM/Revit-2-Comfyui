# WabiSabi Bridge para Revit 2026 - MVP

## 🎯 Funcionalidad Actual (MVP)

Esta es la versión mínima funcional del plugin WabiSabi Bridge. Actualmente incluye:

- ✅ Exportación manual de la vista 3D activa
- ✅ Exportación de imagen con líneas ocultas
- ✅ Exportación de metadatos básicos (JSON)
- ✅ Modo automático experimental (exporta cada 2 segundos)
- ✅ Interfaz de usuario simple
- ✅ Configuración persistente

## 📋 Requisitos

- Autodesk Revit 2026
- .NET Framework 4.8
- Visual Studio 2022 (para compilar)
- Windows 10/11

## 🔧 Instalación

### Opción 1: Compilar desde código fuente

1. **Clonar o descargar los archivos**:
   - `WabiSabiBridge.cs`
   - `WabiSabiBridge.csproj`
   - `WabiSabiBridge.addin`

2. **Abrir el proyecto en Visual Studio 2022**:
   ```bash
   # Abrir Visual Studio y cargar WabiSabiBridge.csproj
   ```

3. **Ajustar la ruta de Revit** (si es necesario):
   - Editar `WabiSabiBridge.csproj`
   - Modificar `<RevitInstallPath>` si Revit está en otra ubicación

4. **Compilar el proyecto**:
   - Build → Build Solution (Ctrl+Shift+B)
   - El proyecto se compilará e instalará automáticamente

### Opción 2: Instalación manual

1. **Copiar archivos al directorio de Addins**:
   ```
   C:\Users\[TuUsuario]\AppData\Roaming\Autodesk\Revit\Addins\2026\
   ```
   - `WabiSabiBridge.dll`
   - `WabiSabiBridge.addin`

2. **Reiniciar Revit**

## 🚀 Uso

### Primera vez:

1. **Abrir Revit 2026** y cargar un proyecto
2. **Activar una vista 3D**
3. **Ir a la pestaña "WabiSabi"** en el ribbon
4. **Hacer clic en "WabiSabi Bridge"**

### Ventana del plugin:

1. **Configurar carpeta de salida**:
   - Por defecto: `Documentos\WabiSabiBridge`
   - Usar botón "..." para cambiar

2. **Exportar manualmente**:
   - Clic en "Exportar Vista Actual"
   - Se generarán archivos:
     - `current_render.png` - Imagen de líneas ocultas
     - `current_metadata.json` - Información de la vista
     - `last_update.txt` - Timestamp de actualización

3. **Modo automático** (experimental):
   - Activar checkbox "Exportación automática"
   - Exportará cada 2 segundos si detecta cambios

## 📁 Estructura de archivos exportados

```
WabiSabiBridge/
├── current_render.png      # Imagen actual (se sobrescribe)
├── current_metadata.json   # Metadatos actuales
├── last_update.txt        # Timestamp de última actualización
├── render_20240115_143022.png  # Archivos con timestamp (histórico)
└── ...
```

## 🔄 Integración con ComfyUI

En ComfyUI, usar el nodo `Watch Image` (del otro plan) apuntando a:
- Imagen: `[carpeta]\current_render.png`
- Metadatos: `[carpeta]\current_metadata.json`

## 🐛 Problemas conocidos (MVP)

- El modo automático exporta siempre (no detecta cambios reales)
- Solo exporta líneas ocultas (sin profundidad, normales, etc.)
- La posición de cámara en metadatos está hardcodeada
- No hay indicador de progreso para modelos grandes

## 🔧 Solución de problemas comunes

### Error: "Starting a transaction from an external application"
**Causa**: Conflicto de versiones + contexto de transacción inválido
**Solución**: Usar el script `FixVersionBuild.bat` que corrige ambos problemas

### Error: Referencias ambiguas (Form, TextBox, Color)
**Solución**: Ya corregido en el código usando alias para namespaces

### Error: Elementos "Compile" duplicados
**Solución**: Ya corregido removiendo la sección problemática del .csproj

### Error: "IsVisible" no existe
**Solución**: Ya corregido usando `IsDisposed` en lugar de `IsVisible`

### Error: Conflicto de versiones de ensamblados
**Solución**: Usar `FixVersionBuild.bat` que evita copiar DLLs del sistema

### Error: No se encuentra Revit API
**Solución**: Verificar que Revit 2026 esté instalado o actualizar la ruta en el .csproj

Para más detalles, ver:
- `COMPILATION_FIX.md` - Errores de compilación
- `VERSION_CONFLICT_FIX.md` - Conflictos de versiones y transacciones

## 🚧 Próximas iteraciones

**v0.2 - Mejoras básicas**:
- [ ] Detección real de cambios en la vista
- [ ] Barra de progreso durante exportación
- [ ] Configuración de resolución

**v0.3 - Más tipos de datos**:
- [ ] Exportar mapa de profundidad
- [ ] Exportar segmentación por categorías
- [ ] Posición real de cámara

**v0.4 - Optimización**:
- [ ] Named pipes para comunicación directa
- [ ] Caché para evitar re-exportar
- [ ] Procesamiento en background thread

**v1.0 - Versión completa**:
- [ ] Todos los extractores del plan original
- [ ] UI avanzada con tabs
- [ ] Modo batch
- [ ] Documentación completa

## 📝 Notas de desarrollo

- El plugin usa `ImageExportOptions` de Revit API para exportar
- La configuración se guarda en `%APPDATA%\WabiSabiBridge\config.json`
- Los logs de error aparecen en los diálogos de Revit

## 🤝 Contribuir

Para reportar bugs o sugerir mejoras, por favor documentar:
1. Versión de Revit
2. Descripción del problema
3. Pasos para reproducir
4. Mensajes de error (si los hay)

---

**WabiSabi Bridge MVP v0.1** - Enero 2024