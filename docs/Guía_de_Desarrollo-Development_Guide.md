# Guía de Desarrollo - WabiSabi Bridge MVP

## 🏗️ Arquitectura del Código

### Estructura de clases principales:

```
WabiSabiBridgeCommand (IExternalCommand)
    ├── ExportEventHandler (IExternalEventHandler)
    │   └── Execute() - Ejecuta en contexto válido de Revit
    └── WabiSabiBridgeWindow (Form)
         ├── ExternalEvent para ejecutar comandos
         └── Timer para auto-export

WabiSabiBridgeApp (IExternalApplication)
    └── Crea el botón en el Ribbon

WabiSabiConfig
    └── Gestión de configuración persistente
```

### ⚠️ IMPORTANTE: Patrón ExternalEvent

En Revit, las transacciones **SOLO** pueden ejecutarse desde el thread principal de Revit. Si intentas ejecutar una transacción desde un Windows Form, obtendrás el error:
```
Starting a transaction from an external application
```

**Solución**: Usar el patrón `IExternalEventHandler`:

```csharp
// 1. Crear el handler
public class MyEventHandler : IExternalEventHandler
{
    public void Execute(UIApplication app)
    {
        // Aquí SÍ puedes usar transacciones
        using (Transaction t = new Transaction(doc, "Mi operación"))
        {
            t.Start();
            // ... hacer cambios
            t.Commit();
        }
    }
}

// 2. Crear el evento
var handler = new MyEventHandler();
var externalEvent = ExternalEvent.Create(handler);

// 3. Ejecutar desde el Form
private void Button_Click(object sender, EventArgs e)
{
    externalEvent.Raise(); // Ejecuta en el contexto correcto
}
```

## 🔧 Puntos de extensión para siguientes iteraciones

### 1. **Agregar nuevos tipos de exportación**

#### Patrón para nuevos extractores:

1. **Crear clase en `src/Extractors/`**:
```csharp
namespace WabiSabiBridge.Extractors
{
    public class MyNewExtractor
    {
        private readonly UIApplication _uiApp;
        
        public MyNewExtractor(UIApplication uiApp)
        {
            _uiApp = uiApp;
        }
        
        public void Extract(View3D view3D, string outputPath, string timestamp)
        {
            // Implementación
        }
    }
}
```

2. **Agregar en ExportEventHandler.Execute()**:
```csharp
if (ExportMyFeature)
{
    var extractor = new MyNewExtractor(app);
    extractor.Extract(view3D, OutputPath, timestamp);
}
```

3. **Agregar controles en la UI**:
- CheckBox para habilitar/deshabilitar
- ComboBox o TextBox para opciones
- Actualizar WabiSabiConfig

#### Ejemplo: Agregar extractor de segmentación

```csharp
// SegmentationExtractor.cs
public class SegmentationExtractor
{
    public void ExtractSegmentation(View3D view3D, string outputPath, string timestamp)
    {
        // 1. Obtener categorías únicas
        var categories = GetUniqueCategories(doc, view3D);
        
        // 2. Asignar color a cada categoría
        var colorMap = AssignColors(categories);
        
        // 3. Renderizar imagen segmentada
        var segmentationMap = RenderSegmentation(view3D, colorMap);
        
        // 4. Guardar imagen y leyenda
        SaveSegmentation(segmentationMap, colorMap, outputPath, timestamp);
    }
}
```

### 2. **Mejorar la detección de cambios**

Reemplazar el método `HasViewChanged()`:

```csharp
private bool HasViewChanged()
{
    // TODO: Implementar detección real
    // Ideas:
    // - Comparar hash de ViewOrientation3D
    // - Detectar cambios en elementos visibles
    // - Monitorear eventos de Revit API
    
    var currentViewState = GetViewState();
    bool changed = !currentViewState.Equals(_lastViewState);
    _lastViewState = currentViewState;
    return changed;
}
```

### 3. **Implementar mapa de profundidad**

Ejemplo básico:

```csharp
private void ExportDepthMap(Document doc, View3D view3D, string outputPath, string timestamp)
{
    int width = 512;  // Resolución inicial baja para MVP
    int height = 512;
    
    Bitmap depthMap = new Bitmap(width, height);
    
    // Obtener bounding box de la vista
    BoundingBoxXYZ viewBounds = view3D.GetSectionBox();
    
    // Para cada píxel, lanzar un rayo
    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            // Calcular rayo desde la cámara
            XYZ rayOrigin = CalculateRayOrigin(x, y, width, height, view3D);
            XYZ rayDirection = CalculateRayDirection(x, y, width, height, view3D);
            
            // Usar ReferenceIntersector
            ReferenceIntersector intersector = new ReferenceIntersector(
                view3D, 
                ElementClassFilters.Solid, 
                FindReferenceTarget.Element);
                
            ReferenceWithContext refContext = intersector.FindNearest(
                rayOrigin, 
                rayDirection);
                
            // Calcular profundidad y normalizar
            double depth = refContext?.GetReference()?.GlobalPoint.DistanceTo(rayOrigin) ?? double.MaxValue;
            byte depthValue = NormalizeDepth(depth, viewBounds);
            
            depthMap.SetPixel(x, y, Color.FromArgb(depthValue, depthValue, depthValue));
        }
    }
    
    depthMap.Save(Path.Combine(outputPath, "current_depth.png"));
}
```

### 4. **Agregar comunicación por Named Pipes**

Reemplazar la escritura de archivos:

```csharp
public class NamedPipeChannel : ICommunicationChannel
{
    private NamedPipeServerStream _pipeServer;
    
    public void Initialize()
    {
        _pipeServer = new NamedPipeServerStream(
            "WabiSabiBridge", 
            PipeDirection.Out, 
            1, 
            PipeTransmissionMode.Byte);
    }
    
    public async Task SendDataAsync(byte[] data)
    {
        await _pipeServer.WriteAsync(data, 0, data.Length);
    }
}
```

### 5. **Optimizar el renderizado de imágenes**

Usar múltiples resoluciones:

```csharp
private void ExportMultiResolution(Document doc, View3D view3D, string outputPath)
{
    int[] resolutions = { 512, 1024, 2048 };
    
    Parallel.ForEach(resolutions, res =>
    {
        var options = new ImageExportOptions
        {
            PixelSize = res,
            FilePath = Path.Combine(outputPath, $"render_{res}")
            // ... otras opciones
        };
        
        // Exportar en paralelo
        using (var subTransaction = new SubTransaction(doc))
        {
            subTransaction.Start();
            doc.ExportImage(options);
            subTransaction.Commit();
        }
    });
}
```

## 📊 Métricas de rendimiento a implementar

```csharp
public class PerformanceMetrics
{
    public TimeSpan ExportDuration { get; set; }
    public long MemoryUsed { get; set; }
    public int ElementsProcessed { get; set; }
    
    public void LogToFile()
    {
        // Guardar métricas para análisis
    }
}
```

## 🧪 Testing

### Unit Tests básicos a implementar:

```csharp
[TestClass]
public class WabiSabiConfigTests
{
    [TestMethod]
    public void TestConfigSaveAndLoad()
    {
        var config = new WabiSabiConfig
        {
            OutputPath = @"C:\Test",
            AutoExport = true
        };
        
        config.Save();
        var loaded = WabiSabiConfig.Load();
        
        Assert.AreEqual(config.OutputPath, loaded.OutputPath);
        Assert.AreEqual(config.AutoExport, loaded.AutoExport);
    }
}
```

## 🔌 Integración con ComfyUI

### Formato de datos esperado:

```json
{
  "version": "1.0",
  "timestamp": "20240115_143022",
  "data_available": {
    "render": true,
    "depth": false,
    "segmentation": false,
    "normals": false
  },
  "files": {
    "render": "current_render.png",
    "metadata": "current_metadata.json"
  }
}
```

### Protocolo de comunicación futuro:

```
1. Revit → ComfyUI: "NEW_DATA_AVAILABLE"
2. ComfyUI → Revit: "ACK"
3. Revit → ComfyUI: [Binary Data Stream]
4. ComfyUI → Revit: "DATA_RECEIVED"
```

## 🛠️ Herramientas útiles para desarrollo

1. **RevitLookup** - Para inspeccionar elementos de Revit
2. **Visual Studio Diagnostic Tools** - Para profiling
3. **ILSpy** - Para entender Revit API internals
4. **Process Monitor** - Para debugging de I/O

## 📝 Checklist para nuevas características

- [ ] Implementar la lógica core
- [ ] Agregar UI controls si es necesario
- [ ] Actualizar la configuración
- [ ] Agregar manejo de errores
- [ ] Escribir tests
- [ ] Actualizar documentación
- [ ] Probar con modelos grandes
- [ ] Verificar compatibilidad con ComfyUI

## 🚀 Roadmap técnico detallado

### v0.2 - Fundación sólida
- Refactorizar a patrón MVVM
- Agregar logging estructurado
- Implementar detección real de cambios
- Tests unitarios básicos

### v0.3 - Extractores básicos
- Depth map con ReferenceIntersector
- Segmentación por categorías
- Exportación de materiales

### v0.4 - Optimización
- Threading apropiado
- Caché en memoria
- Named pipes básico

### v0.5 - Características avanzadas
- Normal maps
- Múltiples vistas
- Batch processing

### v1.0 - Producción
- Todos los extractores
- UI completa
- Documentación exhaustiva
- Instalador MSI