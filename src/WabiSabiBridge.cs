// WabiSabiBridge.cs - Implementación con aceleración GPU v0.3.0
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Drawing.Imaging;
using System.Diagnostics;
using Newtonsoft.Json;

// Revit API
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Autodesk.Revit.DB.Events;

// Windows Forms con alias para evitar conflictos
using WinForms = System.Windows.Forms;
using Drawing = System.Drawing;

// Extractores
using WabiSabiBridge.Extractors;
using WabiSabiBridge.Extractors.Gpu; // <<-- AÑADIDO para resolver ambigüedades
using WabiSabiBridge.Extractors.Cache;
using ComputeSharp;

namespace WabiSabiBridge
{
    /// <summary>
    /// Comando principal del plugin WabiSabi Bridge
    /// </summary>
    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    public class WabiSabiBridgeCommand : IExternalCommand
    {
        private static WabiSabiBridgeWindow? _window;
        private static ExternalEvent? _externalEvent;
        private static ExportEventHandler? _eventHandler;
        
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            try
            {
                UIApplication uiApp = commandData.Application;
                UIDocument uiDoc = uiApp.ActiveUIDocument;
                Document doc = uiDoc.Document;
                
                View3D? view3D = doc.ActiveView as View3D;
                if (view3D == null)
                {
                    Autodesk.Revit.UI.TaskDialog.Show("WabiSabi Bridge", "Por favor, activa una vista 3D antes de ejecutar el comando.");
                    return Result.Failed;
                }
                
                if (_eventHandler == null)
                {
                    _eventHandler = new ExportEventHandler();
                    _externalEvent = ExternalEvent.Create(_eventHandler);
                }
                
                if (_window == null || _window.IsDisposed)
                {
                    // CORREGIDO: Usar el operador "null-forgiving" (!) porque sabemos que no son nulos aquí.
                    _window = new WabiSabiBridgeWindow(uiApp, _externalEvent!, _eventHandler!);
                    _window.Show();
                }
                else
                {
                    _window.Show();
                    _window.Focus();
                }
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                message = ex.Message;
                return Result.Failed;
            }
        }
    }
    
    /// <summary>
    /// Handler para ejecutar exportaciones en el contexto válido de Revit
    /// </summary>
    public class ExportEventHandler : IExternalEventHandler
    {
        public UIApplication? UiApp { get; set; }
        public string OutputPath { get; set; } = string.Empty;
        public bool ExportDepth { get; set; }
        public int DepthResolution { get; set; } = 512;
        public int DepthQuality { get; set; } = 1; // 0=Rápida, 1=Normal, 2=Alta
        public bool AutoDepthRange { get; set; } = true;
        public double DepthRangeDistance { get; set; } = 50.0;
        public bool UseGpuAcceleration { get; set; } = true;
        public bool UseGeometryExtraction { get; set; } = false;
        public Action<string, Drawing.Color>? UpdateStatusCallback { get; set; }
        
        private DepthExtractor? _depthExtractor;
        private DepthExtractorFast? _depthExtractorFast;
        private IGpuAccelerationManager? _gpuManager;
        
        public void Execute(UIApplication app)
        {
            try
            {
                UiApp = app;
                UIDocument uiDoc = app.ActiveUIDocument;
                Document doc = uiDoc.Document;
                View3D? view3D = doc.ActiveView as View3D;

                if (view3D == null)
                {
                    UpdateStatusCallback?.Invoke("Error: No hay vista 3D activa", Drawing.Color.Red);
                    return;
                }

                if (!Directory.Exists(OutputPath))
                {
                    Directory.CreateDirectory(OutputPath);
                }

                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");

                // --- INICIO DE LA LÓGICA CORREGIDA ---

                // 1. Obtener la vista UI para calcular el aspect ratio real, que es la fuente de verdad.
                UIView? uiView = uiDoc.GetOpenUIViews().FirstOrDefault(v => v.ViewId == view3D.Id);
                if (uiView == null)
                {
                    UpdateStatusCallback?.Invoke("Error: No se pudo obtener la ventana de la vista para el encuadre.", Drawing.Color.Red);
                    return;
                }
                var viewRect = uiView.GetWindowRectangle();
                double trueAspectRatio = (double)(viewRect.Right - viewRect.Left) / (viewRect.Bottom - viewRect.Top);

                // 2. Calcular las dimensiones del mapa de profundidad basadas en este aspect ratio.
                int depthWidth = this.DepthResolution; // El ancho lo define el usuario.
                double rawDepthHeight = depthWidth / trueAspectRatio; // Calculamos el alto para que coincida.
                int depthHeight = (int)(Math.Round(rawDepthHeight / 2.0) * 2.0); // Asegurar que la altura sea par.

                // 3. Exportar la imagen de renderizado, forzándola a tener las MISMAS dimensiones que el mapa de profundidad.
                UpdateStatusCallback?.Invoke("Exportando y sincronizando vista...", Drawing.Color.Blue);
                ExportHiddenLineImage(doc, view3D, OutputPath, timestamp, depthWidth, depthHeight);

                // 4. Obtener los viewCorners para los extractores de profundidad.
                IList<XYZ> viewCorners = uiView.GetZoomCorners();

                // --- FIN DE LA LÓGICA CORREGIDA ---

                if (ExportDepth)
                {
                    _gpuManager ??= new GpuAccelerationManager(null);
                    // NUEVO: Usar flujo optimizado con caché para modo experimental
                    // Bloque a reemplazar dentro de ExportEventHandler.Execute(...)

                    if (UseGeometryExtraction && UseGpuAcceleration && _gpuManager?.IsGpuAvailable == true)
                    {
                        UpdateStatusCallback?.Invoke("Modo experimental con caché inteligente...", Drawing.Color.Blue);

                        try
                        {
                            var cacheManager = WabiSabiBridge.Extractors.Cache.GeometryCacheManager.Instance;
                            var cachedData = cacheManager.EnsureCacheIsValid(doc, view3D, UpdateStatusCallback);

                            var eyePosition = view3D.GetOrientation().EyePosition;
                            // --- LÍNEA AÑADIDA PARA CORREGIR EL ERROR DE COMPILACIÓN ---
                            var forwardDirection = view3D.GetOrientation().ForwardDirection.Normalize();
                            // --- FIN DE LA LÍNEA AÑADIDA ---

                            var corners = uiView.GetZoomCorners();
                            var cornerBottomLeft = corners[0];
                            var cornerTopRight = corners[1];

                            // Reconstruir las 4 esquinas del viewport para máxima precisión
                            var up = view3D.GetOrientation().UpDirection.Normalize();
                            var right = forwardDirection.CrossProduct(up).Normalize();
                            var viewDiagonal = cornerTopRight - cornerBottomLeft;
                            var rightComponent = viewDiagonal.DotProduct(right) * right;
                            var upComponent = viewDiagonal.DotProduct(up) * up;
                            var cornerBottomRight = cornerBottomLeft + rightComponent;
                            var cornerTopLeft = cornerBottomLeft + upComponent;

                            var baseVector = cornerBottomLeft - eyePosition;
                            var rightSpanVector = cornerBottomRight - cornerBottomLeft;
                            var upSpanVector = cornerTopLeft - cornerBottomLeft;
                            
                            // Ahora esta línea compilará correctamente porque 'forwardDirection' existe
                            double minDepth = 0.1;
                            double maxDepth = CalculateMaxDepth(view3D, eyePosition, forwardDirection, (cornerBottomLeft + cornerTopRight) / 2.0);

                            // Configuración para GPU usando los vectores de interpolación
                            var config = new WabiSabiBridge.Extractors.Gpu.RayTracingConfig
                            {
                                EyePosition = new ComputeSharp.Float3((float)eyePosition.X, (float)eyePosition.Y, (float)eyePosition.Z),
                                ViewDirection = new ComputeSharp.Float3((float)baseVector.X, (float)baseVector.Y, (float)baseVector.Z),
                                RightDirection = new ComputeSharp.Float3((float)rightSpanVector.X, (float)rightSpanVector.Y, (float)rightSpanVector.Z),
                                UpDirection = new ComputeSharp.Float3((float)upSpanVector.X, (float)upSpanVector.Y, (float)upSpanVector.Z),
                                Width = depthWidth,
                                Height = depthHeight,
                                MinDepth = (float)minDepth,
                                MaxDepth = (float)maxDepth
                            };

                            UpdateStatusCallback?.Invoke("Procesando en GPU con caché...", Drawing.Color.Blue);

                            // El resto del código no cambia...
                            var depthTask = Task.Run(async () =>
                            {
                                return await _gpuManager.ExecuteDepthRayTracingFromCacheAsync(
                                    cachedData.GeometryMmf,
                                    cachedData.VertexCount,
                                    cachedData.TriangleCount,
                                    config);
                            });

                            float[] gpuDepthBuffer = depthTask.Result;
                            double[,] depthData = new double[depthHeight, depthWidth];
                            for (int y = 0; y < depthHeight; y++)
                            {
                                for (int x = 0; x < depthWidth; x++)
                                {
                                    depthData[y, x] = gpuDepthBuffer[y * depthWidth + x];
                                }
                            }

                            GenerateDepthImage(depthData, depthWidth, depthHeight, OutputPath, timestamp);

                            UpdateStatusCallback?.Invoke(
                                $"Completado con caché - {cacheManager.GetPerformanceStats()}",
                                Drawing.Color.Green);
                        }
                        catch (Exception ex)
                        {
                            UpdateStatusCallback?.Invoke($"Error en modo experimental: {ex.Message}", Drawing.Color.Red);
                            UseGeometryExtraction = false;
                            Execute(app);
                            return;
                        }
                    }
                    else
                    {
                        // Flujo normal sin caché (DepthExtractor y DepthExtractorFast)
                        string gpuStatus = UseGpuAcceleration ? " (GPU)" : "";
                        string depthStatus = DepthQuality == 0 ? $"Generando mapa de profundidad (modo rápido{gpuStatus})..." :
                                           DepthQuality == 2 ? $"Generando mapa de profundidad (alta calidad{gpuStatus})..." :
                                           $"Generando mapa de profundidad{gpuStatus}...";
                        UpdateStatusCallback?.Invoke(depthStatus, Drawing.Color.Blue);

                        try
                        {
                            if (DepthQuality == 0) // Rápida
                            {
                                _depthExtractorFast ??= new DepthExtractorFast(app, DepthResolution, 4);
                                _depthExtractorFast.AutoDepthRange = this.AutoDepthRange;
                                _depthExtractorFast.ManualDepthDistance = this.DepthRangeDistance;
                                _depthExtractorFast.UseGpuAcceleration = this.UseGpuAcceleration;
                                _depthExtractorFast.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                            }
                            else if (DepthQuality == 2) // Alta
                            {
                                _depthExtractor ??= new DepthExtractor(app, DepthResolution);
                                _depthExtractor.AutoDepthRange = this.AutoDepthRange;
                                _depthExtractor.ManualDepthDistance = this.DepthRangeDistance;
                                _depthExtractor.UseGpuAcceleration = this.UseGpuAcceleration;
                                _depthExtractor.UseGeometryExtraction = false; // Ya no usar el modo antiguo
                                _depthExtractor.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                            }
                            else // Normal
                            {
                                _depthExtractorFast ??= new DepthExtractorFast(app, DepthResolution, 2);
                                _depthExtractorFast.AutoDepthRange = this.AutoDepthRange;
                                _depthExtractorFast.ManualDepthDistance = this.DepthRangeDistance;
                                _depthExtractorFast.UseGpuAcceleration = this.UseGpuAcceleration;
                                _depthExtractorFast.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                            }
                        }
                        catch (Exception ex)
                        {
                            UpdateStatusCallback?.Invoke($"Advertencia: Error en profundidad - {ex.Message}", Drawing.Color.Orange);
                            if (UseGpuAcceleration)
                            {
                                UpdateStatusCallback?.Invoke("Reintentando sin GPU...", Drawing.Color.Orange);
                                UseGpuAcceleration = false;
                                Execute(app); // Reintentar
                                return;
                            }
                        }
                    }
                }

                ExportMetadata(doc, view3D, OutputPath, timestamp);
                CreateNotificationFile(OutputPath, timestamp);

                string gpuInfo = UseGpuAcceleration ? " [GPU]" : "";
                UpdateStatusCallback?.Invoke($"Exportado: {timestamp}{gpuInfo}", Drawing.Color.Green);
            }
            catch (Exception ex)
            {
                UpdateStatusCallback?.Invoke($"Error: {ex.Message}", Drawing.Color.Red);
            }
        }
        
        // MÉTODO REEMPLAZADO CON TU ACTUALIZACIÓN
        private void ExportHiddenLineImage(Document doc, View3D view3D, string outputPath, string timestamp, 
    int targetWidth, int targetHeight)
        {
            ImageExportOptions options = new ImageExportOptions
            {
                FilePath = Path.Combine(outputPath, $"render_{timestamp}"),
                HLRandWFViewsFileType = ImageFileType.PNG,
                ImageResolution = ImageResolution.DPI_150,

                // --- INICIO DE LA CORRECCIÓN CRÍTICA ---
                // Forzar a Revit a usar el zoom y encuadre actual de la vista, sin hacer ajustes automáticos.
                // Esto garantiza que la perspectiva de la imagen renderizada coincida con la del mapa de profundidad.
                ZoomType = ZoomFitType.Zoom,
                ExportRange = ExportRange.VisibleRegionOfCurrentView,
                // --- FIN DE LA CORRECCIÓN CRÍTICA ---

                // Establecemos el tamaño de píxel deseado. Revit se aproximará lo mejor posible.
                PixelSize = targetWidth,
                FitDirection = FitDirectionType.Horizontal 
            };
            
            using (Transaction trans = new Transaction(doc, "Export Image"))
            {
                trans.Start();
                doc.ExportImage(options);
                trans.Commit();
            }

            // El resto de la lógica para redimensionar y nombrar el archivo sigue siendo válida y necesaria
            // para garantizar una coincidencia de píxeles perfecta.
            string generatedFile = Path.Combine(outputPath, $"render_{timestamp}.png");
            string targetFile = Path.Combine(outputPath, "current_render.png");

            if (File.Exists(generatedFile))
            {
                using (var originalImage = Drawing.Image.FromFile(generatedFile))
                {
                    // Si las dimensiones finales no coinciden, se redimensiona con letterboxing.
                    if (originalImage.Width != targetWidth || originalImage.Height != targetHeight)
                    {
                        using (var resizedImage = new Drawing.Bitmap(targetWidth, targetHeight))
                        {
                            resizedImage.SetResolution(originalImage.HorizontalResolution, originalImage.VerticalResolution);
                            using (var g = Drawing.Graphics.FromImage(resizedImage))
                            {
                                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                                g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                                
                                float srcAspect = (float)originalImage.Width / originalImage.Height;
                                float dstAspect = (float)targetWidth / targetHeight;
                                
                                int drawWidth, drawHeight, drawX, drawY;
                                
                                if (srcAspect > dstAspect)
                                {
                                    drawHeight = targetHeight;
                                    drawWidth = (int)(targetHeight * srcAspect);
                                    drawX = (targetWidth - drawWidth) / 2;
                                    drawY = 0;
                                }
                                else
                                {
                                    drawWidth = targetWidth;
                                    drawHeight = (int)(targetWidth / srcAspect);
                                    drawX = 0;
                                    drawY = (targetHeight - drawHeight) / 2;
                                }
                                
                                // Usar un color de fondo oscuro similar al de tu render.
                                g.Clear(Drawing.Color.FromArgb(255, 40, 43, 48));
                                g.DrawImage(originalImage, new Drawing.Rectangle(drawX, drawY, drawWidth, drawHeight));
                            }
                            
                            resizedImage.Save(targetFile, Drawing.Imaging.ImageFormat.Png);
                        }
                    }
                    else
                    {
                        // Las dimensiones ya coinciden, solo se guarda con el nombre correcto.
                        originalImage.Save(targetFile, Drawing.Imaging.ImageFormat.Png);
                    }
                }
                
                // Limpiar el archivo temporal.
                try { File.Delete(generatedFile); } catch (IOException) { /* Ignorar si está bloqueado */ }
            }
        }
        
        private void ExportMetadata(Document doc, View3D view3D, string outputPath, string timestamp)
        {
            var metadata = new
            {
                timestamp = timestamp,
                view_name = view3D.Name,
                view_type = "3D",
                scale = view3D.Scale,
                detail_level = view3D.DetailLevel.ToString(),
                display_style = view3D.DisplayStyle.ToString(),
                camera = new
                {
                    eye_position = new { x = 0, y = 0, z = 0 },
                    target_position = new { x = 0, y = 0, z = 0 },
                    up_vector = new { x = 0, y = 1, z = 0 }
                },
                project_info = new
                {
                    name = doc.Title,
                    path = doc.PathName
                },
                gpu_acceleration = UseGpuAcceleration
            };
            
            string metadataPath = Path.Combine(outputPath, "current_metadata.json");
            File.WriteAllText(metadataPath, JsonConvert.SerializeObject(metadata, Formatting.Indented));
        }
        
        private void CreateNotificationFile(string outputPath, string timestamp)
        {
            string notificationPath = Path.Combine(outputPath, "last_update.txt");
            File.WriteAllText(notificationPath, timestamp);
        }
        
        private double CalculateMaxDepth(View3D view3D, XYZ eyePosition, XYZ forwardDirection, XYZ viewCenter)
        {
            if (!AutoDepthRange)
            {
                return DepthRangeDistance;
            }

            double targetDistance = 10.0;

            if (view3D.IsPerspective)
            {
                double distanceToTarget = -1.0;

                BoundingBoxXYZ cropBox = view3D.CropBox;
                if (cropBox.Enabled)
                {
                    XYZ cropCenter = (cropBox.Min + cropBox.Max) * 0.5;
                    XYZ eyeToCenter = cropCenter.Subtract(eyePosition);
                    double distanceAlongView = eyeToCenter.DotProduct(forwardDirection);

                    if (distanceAlongView > 0)
                    {
                        distanceToTarget = distanceAlongView;
                    }
                }

                if (distanceToTarget < 0)
                {
                    XYZ eyeToCenter = viewCenter.Subtract(eyePosition);
                    double distanceToViewPlane = eyeToCenter.DotProduct(forwardDirection);
                    if (distanceToViewPlane > 0)
                    {
                        distanceToTarget = distanceToViewPlane;
                    }
                }

                if (distanceToTarget > 0)
                {
                    targetDistance = distanceToTarget;
                    return targetDistance * 1.2;
                }
                else
                {
                    BoundingBoxUV outline = view3D.Outline;
                    double viewWidth = outline.Max.U - outline.Min.U;
                    targetDistance = viewWidth / (2.0 * Math.Tan(Math.PI / 6.0));
                    return targetDistance * 1.5;
                }
            }
            else
            {
                return 50.0;
            }
        }
        
        private void GenerateDepthImage(double[,] depthData, int width, int height,
            string outputPath, string timestamp)
        {
            using (var depthMap = new Drawing.Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb))
            {
                BitmapData bmpData = depthMap.LockBits(
                    new Drawing.Rectangle(0, 0, width, height),
                    ImageLockMode.WriteOnly,
                    depthMap.PixelFormat);

                unsafe
                {
                    byte* ptr = (byte*)bmpData.Scan0;
                    int stride = bmpData.Stride;

                    Parallel.For(0, height, y =>
                    {
                        byte* row = ptr + (y * stride);
                        for (int x = 0; x < width; x++)
                        {
                            byte depthValue = (byte)(Math.Max(0.0, Math.Min(1.0, depthData[y, x])) * 255);
                            int pixelOffset = x * 3;
                            row[pixelOffset] = depthValue;
                            row[pixelOffset + 1] = depthValue;
                            row[pixelOffset + 2] = depthValue;
                        }
                    });
                }

                depthMap.UnlockBits(bmpData);

                var encoderParams = new EncoderParameters(1);
                encoderParams.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, 90L);

                var pngCodec = ImageCodecInfo.GetImageEncoders()
                    .First(codec => codec.FormatID == Drawing.Imaging.ImageFormat.Png.Guid);

                string depthPath = System.IO.Path.Combine(outputPath, $"depth_{timestamp}.png");
                depthMap.Save(depthPath, pngCodec, encoderParams);

                string currentDepthPath = System.IO.Path.Combine(outputPath, "current_depth.png");
                depthMap.Save(currentDepthPath, pngCodec, encoderParams);
            }
        }

        public string GetName()
        {
            return "WabiSabi Bridge Export Event";
        }
        
        public void Dispose()
        {
            _depthExtractor?.Dispose();
            _depthExtractorFast?.Dispose();
        }
    }
        
    /// <summary>
    /// Aplicación principal del plugin con soporte para invalidación de caché
    /// </summary>
    public class WabiSabiBridgeApp : IExternalApplication
    {
        public Result OnStartup(UIControlledApplication application)
        {
            try
            {
                // Crear ribbon panel
                string tabName = "WabiSabi";
                application.CreateRibbonTab(tabName);
                
                RibbonPanel panel = application.CreateRibbonPanel(tabName, "Bridge");
                
                // Crear botón
                string thisAssemblyPath = System.Reflection.Assembly.GetExecutingAssembly().Location;
                PushButtonData buttonData = new PushButtonData(
                    "WabiSabiBridge",
                    "WabiSabi\nBridge",
                    thisAssemblyPath,
                    "WabiSabiBridge.WabiSabiBridgeCommand");
                
                buttonData.ToolTip = "Exportar vista actual a ComfyUI";
                buttonData.LongDescription = "Extrae imágenes y datos de la vista 3D activa para usar en ComfyUI";
                
                PushButton? button = panel.AddItem(buttonData) as PushButton;
                
                // NUEVO: Suscribirse a eventos de cambios en el documento
                application.ControlledApplication.DocumentChanged += OnDocumentChanged;
                application.ControlledApplication.DocumentOpened += OnDocumentOpened;
                application.ControlledApplication.DocumentClosed += OnDocumentClosed;
                
                System.Diagnostics.Debug.WriteLine("WabiSabiBridge: Iniciado con soporte de caché inteligente");
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error en OnStartup: {ex.Message}");
                return Result.Failed;
            }
        }
        
        public Result OnShutdown(UIControlledApplication application)
        {
            try
            {
                // Desuscribirse de eventos
                application.ControlledApplication.DocumentChanged -= OnDocumentChanged;
                application.ControlledApplication.DocumentOpened -= OnDocumentOpened;
                application.ControlledApplication.DocumentClosed -= OnDocumentClosed;
                
                // Limpiar el caché al cerrar Revit
                WabiSabiBridge.Extractors.Cache.GeometryCacheManager.Instance.Dispose();
                
                System.Diagnostics.Debug.WriteLine("WabiSabiBridge: Apagado limpiamente");
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error en OnShutdown: {ex.Message}");
                return Result.Failed;
            }
        }
        
        /// <summary>
        /// Maneja cambios en el documento para invalidar el caché cuando sea necesario
        /// </summary>
        private void OnDocumentChanged(object? sender, Autodesk.Revit.DB.Events.DocumentChangedEventArgs e)
        {
            try
            {
                var addedIds = e.GetAddedElementIds();
                var deletedIds = e.GetDeletedElementIds();
                var modifiedIds = e.GetModifiedElementIds();
                
                // Verificar si algún cambio afecta a elementos 3D
                bool hasRelevantChanges = false;
                
                if (addedIds.Any() || deletedIds.Any())
                {
                    hasRelevantChanges = true;
                }
                else if (modifiedIds.Any())
                {
                    // Ser más selectivo con modificaciones para evitar invalidaciones innecesarias
                    var doc = e.GetDocument();
                    foreach (var id in modifiedIds.Take(10)) // Revisar solo los primeros 10 para rendimiento
                    {
                        var elem = doc.GetElement(id);
                        if (elem != null && elem.Category != null && 
                            elem.Category.CategoryType == CategoryType.Model)
                        {
                            hasRelevantChanges = true;
                            break;
                        }
                    }
                }
                
                if (hasRelevantChanges)
                {
                    WabiSabiBridge.Extractors.Cache.GeometryCacheManager.Instance.InvalidateCache();
                    System.Diagnostics.Debug.WriteLine($"Caché invalidado: {addedIds.Count} añadidos, {deletedIds.Count} eliminados, {modifiedIds.Count} modificados");
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error en OnDocumentChanged: {ex.Message}");
            }
        }
        
        /// <summary>
        /// Invalida el caché cuando se abre un nuevo documento
        /// </summary>
        private void OnDocumentOpened(object? sender, Autodesk.Revit.DB.Events.DocumentOpenedEventArgs e)
        {
            WabiSabiBridge.Extractors.Cache.GeometryCacheManager.Instance.InvalidateCache();
            System.Diagnostics.Debug.WriteLine("Caché invalidado: Documento abierto");
        }
        
        /// <summary>
        /// Invalida el caché cuando se cierra un documento
        /// </summary>
        private void OnDocumentClosed(object? sender, Autodesk.Revit.DB.Events.DocumentClosedEventArgs e)
        {
            WabiSabiBridge.Extractors.Cache.GeometryCacheManager.Instance.InvalidateCache();
            System.Diagnostics.Debug.WriteLine("Caché invalidado: Documento cerrado");
        }
    }
    
    /// <summary>
    /// Ventana principal del plugin (v0.3.0 con GPU)
    /// </summary>
    public class WabiSabiBridgeWindow : WinForms.Form
    {
        private readonly UIApplication _uiApp;
        private readonly ExternalEvent _externalEvent;
        private readonly ExportEventHandler _eventHandler;
        
        // CORREGIDO: Se elimina 'readonly' y se inicializa con 'null!' para suprimir la advertencia CS8618.
        // El compilador no puede saber que InitializeComponent() los inicializa.
        private WinForms.Button _exportButton = null!;
        private WinForms.TextBox _outputPathTextBox = null!;
        private WinForms.Button _browseButton = null!;
        private WinForms.Label _statusLabel = null!;
        private WinForms.CheckBox _autoExportCheckBox = null!;
        private WinForms.CheckBox _exportDepthCheckBox = null!;
        private WinForms.ComboBox _depthResolutionCombo = null!;
        private WinForms.ComboBox _depthQualityCombo = null!;
        private WinForms.Timer _autoExportTimer = null!;
        private WinForms.TrackBar _depthRangeTrackBar = null!;
        private WinForms.Label _depthRangeLabel = null!;
        private WinForms.Label _depthRangeValueLabel = null!;
        private WinForms.CheckBox _autoDepthCheckBox = null!;
        private WinForms.CheckBox _gpuAccelerationCheckBox = null!;
        private WinForms.CheckBox _geometryExtractionCheckBox = null!;
        private WinForms.Label _gpuStatusLabel = null!;

        // NUEVOS CAMPOS para control de caché
        private WinForms.Button _clearCacheButton = null!;
        private WinForms.Label _cacheStatusLabel = null!;
        private WinForms.Timer _cacheStatusTimer = null!;
        
        // Configuración
        private readonly WabiSabiConfig _config;
        
        public WabiSabiBridgeWindow(UIApplication uiApp, ExternalEvent externalEvent, ExportEventHandler eventHandler)
        {
            _uiApp = uiApp;
            _externalEvent = externalEvent;
            _eventHandler = eventHandler;
            _eventHandler.UpdateStatusCallback = UpdateStatus;
            _config = WabiSabiConfig.Load();
            InitializeComponent(); // Este método inicializa los campos de arriba
            CheckGpuStatus();
        }
        
        private void InitializeComponent()
        {
            // Configuración de la ventana
            Text = "WabiSabi Bridge - v0.3.0 (GPU Enhanced)";
            // MODIFICAR: Aumentar el tamaño de la ventana para los nuevos controles
            Size = new Drawing.Size(420, 620); // Era 550, ahora 620
            StartPosition = WinForms.FormStartPosition.CenterScreen;
            FormBorderStyle = WinForms.FormBorderStyle.FixedDialog;
            MaximizeBox = false;

            // Layout principal
            // MODIFICAR: Aumentar las filas del TableLayoutPanel
            WinForms.TableLayoutPanel mainLayout = new WinForms.TableLayoutPanel
            {
                Dock = WinForms.DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 12, // Era 10, ahora 12
                Padding = new WinForms.Padding(10)
            };

            // Fila 1: Ruta de salida
            WinForms.Panel pathPanel = new WinForms.Panel { Height = 30, Dock = WinForms.DockStyle.Fill };
            WinForms.Label pathLabel = new WinForms.Label { Text = "Carpeta de salida:", Width = 100, TextAlign = Drawing.ContentAlignment.MiddleLeft };
            _outputPathTextBox = new WinForms.TextBox
            {
                Text = _config.OutputPath,
                Left = 105,
                Width = 200,
                Top = 3
            };
            _browseButton = new WinForms.Button
            {
                Text = "...",
                Left = 310,
                Width = 30,
                Top = 2
            };
            _browseButton.Click += BrowseButton_Click;

            pathPanel.Controls.Add(pathLabel);
            pathPanel.Controls.Add(_outputPathTextBox);
            pathPanel.Controls.Add(_browseButton);

            // Fila 2: Botón de exportación
            _exportButton = new WinForms.Button
            {
                Text = "Exportar Vista Actual",
                Height = 40,
                Dock = WinForms.DockStyle.Fill,
                Font = new Drawing.Font(Font.FontFamily, 10, Drawing.FontStyle.Bold)
            };
            _exportButton.Click += ExportButton_Click;

            // Fila 3: Opciones de exportación
            WinForms.GroupBox exportOptionsGroup = new WinForms.GroupBox
            {
                Text = "Opciones de exportación",
                Height = 125,
                Dock = WinForms.DockStyle.Fill
            };

            // Checkbox para mapa de profundidad
            _exportDepthCheckBox = new WinForms.CheckBox
            {
                Text = "Generar mapa de profundidad",
                Location = new Drawing.Point(10, 20),
                Width = 200,
                Checked = _config.ExportDepth
            };
            _exportDepthCheckBox.CheckedChanged += ExportDepthCheckBox_CheckedChanged;

            // ComboBox para resolución de profundidad
            WinForms.Label depthResLabel = new WinForms.Label
            {
                Text = "Resolución:",
                Location = new Drawing.Point(30, 45),
                Width = 70
            };

            _depthResolutionCombo = new WinForms.ComboBox
            {
                Location = new Drawing.Point(100, 43),
                Width = 100,
                DropDownStyle = WinForms.ComboBoxStyle.DropDownList,
                Enabled = _config.ExportDepth
            };
            _depthResolutionCombo.Items.AddRange(new object[] { "256", "512", "1024", "2048" });
            _depthResolutionCombo.SelectedItem = _config.DepthResolution.ToString();
            _depthResolutionCombo.SelectedIndexChanged += DepthResolutionCombo_SelectedIndexChanged;

            // ComboBox para calidad
            WinForms.Label depthQualityLabel = new WinForms.Label
            {
                Text = "Calidad:",
                Location = new Drawing.Point(220, 45),
                Width = 50
            };

            _depthQualityCombo = new WinForms.ComboBox
            {
                Location = new Drawing.Point(270, 43),
                Width = 80,
                DropDownStyle = WinForms.ComboBoxStyle.DropDownList,
                Enabled = _config.ExportDepth
            };
            _depthQualityCombo.Items.AddRange(new object[] { "Rápida", "Normal", "Alta" });
            _depthQualityCombo.SelectedIndex = _config.DepthQuality;
            _depthQualityCombo.SelectedIndexChanged += DepthQualityCombo_SelectedIndexChanged;

            // Label de tiempo estimado
            WinForms.Label timeEstimateLabel = new WinForms.Label
            {
                Name = "timeEstimateLabel",
                Text = GetTimeEstimate(),
                Location = new Drawing.Point(30, 70),
                Width = 320,
                ForeColor = Drawing.Color.Gray,
                Font = new Drawing.Font(Font.FontFamily, 8)
            };

            // Checkbox para GPU
            _gpuAccelerationCheckBox = new WinForms.CheckBox
            {
                Text = "Aceleración GPU",
                Location = new Drawing.Point(30, 95),
                Width = 130,
                Checked = _config.UseGpuAcceleration,
                Enabled = _config.ExportDepth
            };
            _gpuAccelerationCheckBox.CheckedChanged += GpuAccelerationCheckBox_CheckedChanged;

            // Checkbox para extracción de geometría (experimental)
            _geometryExtractionCheckBox = new WinForms.CheckBox
            {
                Text = "Modo Experimental",
                Location = new Drawing.Point(170, 95),
                Width = 140,
                Checked = _config.UseGeometryExtraction,
                Enabled = _config.ExportDepth && _config.UseGpuAcceleration,
                ForeColor = Drawing.Color.DarkOrange
            };
            _geometryExtractionCheckBox.CheckedChanged += GeometryExtractionCheckBox_CheckedChanged;

            exportOptionsGroup.Controls.Add(_exportDepthCheckBox);
            exportOptionsGroup.Controls.Add(depthResLabel);
            exportOptionsGroup.Controls.Add(_depthResolutionCombo);
            exportOptionsGroup.Controls.Add(depthQualityLabel);
            exportOptionsGroup.Controls.Add(_depthQualityCombo);
            exportOptionsGroup.Controls.Add(timeEstimateLabel);
            exportOptionsGroup.Controls.Add(_gpuAccelerationCheckBox);
            exportOptionsGroup.Controls.Add(_geometryExtractionCheckBox);

            // NUEVO: Agregar después del grupo de opciones de exportación (después de la Fila 3)
            // Fila 3.5: Información del caché
            WinForms.Panel cacheInfoPanel = new WinForms.Panel { Height = 40, Dock = WinForms.DockStyle.Fill };

            _cacheStatusLabel = new WinForms.Label
            {
                Text = "Caché: No inicializado",
                Dock = WinForms.DockStyle.Fill,
                TextAlign = Drawing.ContentAlignment.MiddleLeft,
                Font = new Drawing.Font(Font.FontFamily, 8, Drawing.FontStyle.Regular),
                ForeColor = Drawing.Color.Gray
            };

            _clearCacheButton = new WinForms.Button
            {
                Text = "Limpiar Caché",
                Width = 100,
                Height = 25,
                Dock = WinForms.DockStyle.Right,
                BackColor = Drawing.Color.LightCoral,
                ForeColor = Drawing.Color.White,
                FlatStyle = WinForms.FlatStyle.Flat
            };
            _clearCacheButton.Click += ClearCacheButton_Click;

            cacheInfoPanel.Controls.Add(_cacheStatusLabel);
            cacheInfoPanel.Controls.Add(_clearCacheButton);

            // Fila 4: Control de rango de profundidad
            WinForms.Panel depthRangePanel = new WinForms.Panel { Height = 60, Dock = WinForms.DockStyle.Fill };

            _depthRangeLabel = new WinForms.Label
            {
                Text = "Distancia focal:",
                Location = new Drawing.Point(10, 5),
                Width = 100
            };

            _autoDepthCheckBox = new WinForms.CheckBox
            {
                Text = "Auto",
                Location = new Drawing.Point(280, 5),
                Width = 50,
                Checked = _config.AutoDepthRange
            };
            _autoDepthCheckBox.CheckedChanged += AutoDepthCheckBox_CheckedChanged;

            _depthRangeTrackBar = new WinForms.TrackBar
            {
                Location = new Drawing.Point(10, 25),
                Width = 260,
                Height = 30,
                Minimum = 10,
                Maximum = 500,
                Value = (int)_config.DepthRangeDistance,
                TickFrequency = 50,
                Enabled = !_config.AutoDepthRange
            };
            _depthRangeTrackBar.ValueChanged += DepthRangeTrackBar_ValueChanged;

            _depthRangeValueLabel = new WinForms.Label
            {
                Text = _config.AutoDepthRange ? "Auto" : $"{_config.DepthRangeDistance} ft",
                Location = new Drawing.Point(280, 30),
                Width = 80,
                TextAlign = Drawing.ContentAlignment.MiddleLeft
            };

            depthRangePanel.Controls.Add(_depthRangeLabel);
            depthRangePanel.Controls.Add(_autoDepthCheckBox);
            depthRangePanel.Controls.Add(_depthRangeTrackBar);
            depthRangePanel.Controls.Add(_depthRangeValueLabel);

            // Fila 5: Auto-exportación
            _autoExportCheckBox = new WinForms.CheckBox
            {
                Text = "Exportación automática (experimental)",
                Dock = WinForms.DockStyle.Fill,
                Checked = _config.AutoExport
            };
            _autoExportCheckBox.CheckedChanged += AutoExportCheckBox_CheckedChanged;

            // Fila 6: Estado GPU
            _gpuStatusLabel = new WinForms.Label
            {
                Text = "Verificando GPU...",
                Dock = WinForms.DockStyle.Fill,
                TextAlign = Drawing.ContentAlignment.MiddleCenter,
                ForeColor = Drawing.Color.Gray,
                Font = new Drawing.Font(Font.FontFamily, 8, Drawing.FontStyle.Italic)
            };

            // Fila 7: Estado
            _statusLabel = new WinForms.Label
            {
                Text = "Listo para exportar",
                Dock = WinForms.DockStyle.Fill,
                TextAlign = Drawing.ContentAlignment.MiddleCenter,
                ForeColor = Drawing.Color.Green,
                Font = new Drawing.Font(Font.FontFamily, 9, Drawing.FontStyle.Italic)
            };

            // MODIFICAR: Ajustar los índices de las filas al agregar controles al layout
            mainLayout.Controls.Add(pathPanel, 0, 0);
            mainLayout.Controls.Add(_exportButton, 0, 1);
            mainLayout.Controls.Add(exportOptionsGroup, 0, 2);
            mainLayout.Controls.Add(cacheInfoPanel, 0, 3); // NUEVO
            mainLayout.Controls.Add(depthRangePanel, 0, 4); // Era 3, ahora 4
            mainLayout.Controls.Add(_autoExportCheckBox, 0, 5); // Era 4, ahora 5
            mainLayout.Controls.Add(_gpuStatusLabel, 0, 6); // Era 5, ahora 6
            mainLayout.Controls.Add(_statusLabel, 0, 7); // Era 6, ahora 7

            Controls.Add(mainLayout);

            // Timer para auto-exportación
            _autoExportTimer = new WinForms.Timer();
            _autoExportTimer.Interval = 2000; // 2 segundos
            _autoExportTimer.Tick += AutoExportTimer_Tick;

            if (_config.AutoExport)
            {
                _autoExportTimer.Start();
            }

            // NUEVO: Timer para actualizar estado del caché
            _cacheStatusTimer = new WinForms.Timer();
            _cacheStatusTimer.Interval = 1000; // Actualizar cada segundo
            _cacheStatusTimer.Tick += CacheStatusTimer_Tick;
            _cacheStatusTimer.Start();
        }
        
        private void CheckGpuStatus()
        {
            try
            {
                // Pasa el `this` (la ventana) al manager para que pueda mostrar un diálogo
                IGpuAccelerationManager? gpuManager = new GpuAccelerationManager(this);
                
                if (gpuManager.IsGpuAvailable)
                {
                    _gpuStatusLabel.Text = "GPU: Disponible ✓";
                    _gpuStatusLabel.ForeColor = System.Drawing.Color.Green;
                }
                else
                {
                    // El mensaje de por qué no está disponible ya se habría mostrado
                    _gpuStatusLabel.Text = "GPU: No disponible (usando CPU paralela)";
                    _gpuStatusLabel.ForeColor = System.Drawing.Color.Orange;
                }
                gpuManager.Dispose();
            }
            catch (Exception ex) // Captura cualquier otra excepción inesperada
            {
                _gpuStatusLabel.Text = "GPU: Error al verificar";
                _gpuStatusLabel.ForeColor = System.Drawing.Color.Red;
                // Muestra el error detallado
                Autodesk.Revit.UI.TaskDialog.Show("Error de GPU", $"Error detallado al verificar el estado de la GPU:\n\n{ex.ToString()}");
            }
        }
        
        private void BrowseButton_Click(object? sender, EventArgs e)
        {
            using (WinForms.FolderBrowserDialog dialog = new WinForms.FolderBrowserDialog())
            {
                dialog.SelectedPath = _outputPathTextBox.Text;
                if (dialog.ShowDialog() == WinForms.DialogResult.OK)
                {
                    _outputPathTextBox.Text = dialog.SelectedPath;
                    _config.OutputPath = dialog.SelectedPath;
                    _config.Save();
                }
            }
        }
        
        private void ExportButton_Click(object? sender, EventArgs e)
        {
            ExportCurrentView();
        }
        
        private void AutoExportCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _config.AutoExport = _autoExportCheckBox.Checked;
            _config.Save();
            
            if (_autoExportCheckBox.Checked)
            {
                _autoExportTimer.Start();
            }
            else
            {
                _autoExportTimer.Stop();
            }
        }
        
        private void AutoExportTimer_Tick(object? sender, EventArgs e)
        {
            // Verificar si la vista ha cambiado antes de exportar
            if (HasViewChanged())
            {
                ExportCurrentView();
            }
        }
        
        private bool HasViewChanged()
        {
            // MVP: Siempre retorna true para exportar cada vez
            // TODO: Implementar detección real de cambios
            return true;
        }
        
        private void ExportDepthCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _config.ExportDepth = _exportDepthCheckBox.Checked;
            _depthResolutionCombo.Enabled = _exportDepthCheckBox.Checked;
            _depthQualityCombo.Enabled = _exportDepthCheckBox.Checked;
            _gpuAccelerationCheckBox.Enabled = _exportDepthCheckBox.Checked;
            _geometryExtractionCheckBox.Enabled = _exportDepthCheckBox.Checked && _gpuAccelerationCheckBox.Checked;
            _config.Save();
            UpdateTimeEstimate();
        }
        
        private void DepthResolutionCombo_SelectedIndexChanged(object? sender, EventArgs e)
        {
            if (int.TryParse(_depthResolutionCombo.SelectedItem?.ToString(), out int resolution))
            {
                _config.DepthResolution = resolution;
                _config.Save();
                UpdateTimeEstimate();
            }
        }
        
        private void DepthQualityCombo_SelectedIndexChanged(object? sender, EventArgs e)
        {
            _config.DepthQuality = _depthQualityCombo.SelectedIndex;
            _config.Save();
            UpdateTimeEstimate();
        }

        private void AutoDepthCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _depthRangeTrackBar.Enabled = !_autoDepthCheckBox.Checked;
            _depthRangeValueLabel.Text = _autoDepthCheckBox.Checked ? "Auto" : $"{_depthRangeTrackBar.Value} ft";
            _config.AutoDepthRange = _autoDepthCheckBox.Checked;
            _config.Save();
        }

        private void DepthRangeTrackBar_ValueChanged(object? sender, EventArgs e)
        {
            _depthRangeValueLabel.Text = $"{_depthRangeTrackBar.Value} ft";
            _config.DepthRangeDistance = _depthRangeTrackBar.Value;
            _config.Save();
        }
        
        private void GpuAccelerationCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _config.UseGpuAcceleration = _gpuAccelerationCheckBox.Checked;
            _geometryExtractionCheckBox.Enabled = _exportDepthCheckBox.Checked && _gpuAccelerationCheckBox.Checked;
            if (!_gpuAccelerationCheckBox.Checked)
            {
                _geometryExtractionCheckBox.Checked = false;
            }
            _config.Save();
            UpdateTimeEstimate();
        }
        
        private void GeometryExtractionCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _config.UseGeometryExtraction = _geometryExtractionCheckBox.Checked;
            _config.Save();
            UpdateTimeEstimate();
        }
        
        // NUEVO: Manejador para el botón de limpiar caché
        private void ClearCacheButton_Click(object? sender, EventArgs e)
        {
            var result = WinForms.MessageBox.Show(
                "¿Estás seguro de que deseas limpiar el caché de geometría?\n\n" +
                "La próxima exportación será más lenta mientras se reconstruye.",
                "Confirmar limpieza de caché",
                WinForms.MessageBoxButtons.YesNo,
                WinForms.MessageBoxIcon.Question);

            if (result == WinForms.DialogResult.Yes)
            {
                WabiSabiBridge.Extractors.Cache.GeometryCacheManager.Instance.InvalidateCache();
                UpdateStatus("Caché limpiado. Se reconstruirá en la próxima exportación.", Drawing.Color.Orange);
                UpdateCacheStatus();
            }
        }

        // NUEVO: Actualizar estado del caché periódicamente
        private void CacheStatusTimer_Tick(object? sender, EventArgs e)
        {
            UpdateCacheStatus();
        }

        // NUEVO: Actualizar la etiqueta de estado del caché
        private void UpdateCacheStatus()
        {
            try
            {
                var cacheManager = WabiSabiBridge.Extractors.Cache.GeometryCacheManager.Instance;

                if (cacheManager.IsCacheValid)
                {
                    string sizeInfo = cacheManager.CacheSizeBytes > 1048576 ?
                        $"{cacheManager.CacheSizeBytes / 1048576.0:F1}MB" :
                        $"{cacheManager.CacheSizeBytes / 1024.0:F1}KB";

                    string timeAgo = GetTimeAgo(cacheManager.LastCacheTime);

                    _cacheStatusLabel.Text = $"Caché válido: {cacheManager.VertexCount:N0} vértices, " +
                                           $"{cacheManager.TriangleCount:N0} triángulos ({sizeInfo}) - {timeAgo}";
                    _cacheStatusLabel.ForeColor = Drawing.Color.Green;
                    _clearCacheButton.Enabled = true;
                }
                else
                {
                    _cacheStatusLabel.Text = "Caché: No válido (se reconstruirá automáticamente)";
                    _cacheStatusLabel.ForeColor = Drawing.Color.Gray;
                    _clearCacheButton.Enabled = false;
                }

                // Mostrar estadísticas de rendimiento si hay datos
                double hitRate = cacheManager.GetHitRate();
                if (hitRate > 0)
                {
                    _cacheStatusLabel.Text += $" | Hit Rate: {hitRate:P0}";
                }
            }
            catch
            {
                _cacheStatusLabel.Text = "Caché: Estado desconocido";
                _cacheStatusLabel.ForeColor = Drawing.Color.Gray;
            }
        }
        
        // NUEVO: Helper para mostrar tiempo transcurrido
        private string GetTimeAgo(DateTime time)
        {
            var elapsed = DateTime.Now - time;
            if (elapsed.TotalSeconds < 60) return "hace un momento";
            if (elapsed.TotalMinutes < 60) return $"hace {(int)elapsed.TotalMinutes} min";
            if (elapsed.TotalHours < 24) return $"hace {(int)elapsed.TotalHours} h";
            return $"hace {(int)elapsed.TotalDays} días";
        }
        private void UpdateTimeEstimate()
        {
            var timeLabel = Controls.Find("timeEstimateLabel", true).FirstOrDefault() as WinForms.Label;
            if (timeLabel != null)
            {
                timeLabel.Text = GetTimeEstimate();
            }
        }
        
        private string GetTimeEstimate()
        {
            if (!_exportDepthCheckBox.Checked)
                return "";
                
            int resolution = _config.DepthResolution;
            int quality = _config.DepthQuality;
            bool useGpu = _config.UseGpuAcceleration;
            bool useGeometry = _config.UseGeometryExtraction;
            
            // Estimaciones basadas en pruebas
            int[,] timeMatrixCpu = new int[,] {
                // Rápida, Normal, Alta
                { 1, 3, 5 },      // 256
                { 3, 10, 20 },    // 512
                { 10, 40, 80 },   // 1024
                { 40, 160, 320 }  // 2048
            };
            
            int[,] timeMatrixGpu = new int[,] {
                // Rápida, Normal, Alta
                { 1, 1, 2 },      // 256
                { 1, 3, 5 },      // 512
                { 2, 8, 15 },     // 1024
                { 8, 30, 60 }     // 2048
            };
            
            int resIndex = resolution == 256 ? 0 : resolution == 512 ? 1 : resolution == 1024 ? 2 : 3;
            int seconds = useGpu ? timeMatrixGpu[resIndex, quality] : timeMatrixCpu[resIndex, quality];
            
            if (useGeometry)
            {
                seconds = (int)(seconds * 1.5); // Modo experimental es más lento
            }
            
            string gpuText = useGpu ? " (GPU)" : " (CPU)";
            
            if (seconds < 60)
                return $"Tiempo estimado: ~{seconds} segundos{gpuText}";
            else
                return $"Tiempo estimado: ~{seconds / 60} minutos{gpuText}";
        }
        
        private void ExportCurrentView()
        {
            try
            {
                UpdateStatus("Preparando exportación...", Drawing.Color.Blue);
                
                // Configurar el handler con la ruta actual
                _eventHandler.OutputPath = _outputPathTextBox.Text;
                _eventHandler.UiApp = _uiApp;
                _eventHandler.ExportDepth = _exportDepthCheckBox.Checked;
                _eventHandler.DepthResolution = _config.DepthResolution;
                _eventHandler.DepthQuality = _config.DepthQuality;
                _eventHandler.AutoDepthRange = _autoDepthCheckBox.Checked;
                _eventHandler.DepthRangeDistance = _depthRangeTrackBar.Value;
                _eventHandler.UseGpuAcceleration = _gpuAccelerationCheckBox.Checked;
                _eventHandler.UseGeometryExtraction = _geometryExtractionCheckBox.Checked;

                // Ejecutar la exportación a través del ExternalEvent
                _externalEvent.Raise();
            }
            catch (Exception ex)
            {
                UpdateStatus($"Error: {ex.Message}", Drawing.Color.Red);
            }
        }
        
        private void UpdateStatus(string message, Drawing.Color color)
        {
            if (InvokeRequired)
            {
                Invoke(new Action(() => UpdateStatus(message, color)));
                return;
            }
            
            _statusLabel.Text = message;
            _statusLabel.ForeColor = color;
        }
        
        // MODIFICAR: OnFormClosing para limpiar el nuevo timer
        protected override void OnFormClosing(WinForms.FormClosingEventArgs e)
        {
            _autoExportTimer.Stop();
            _autoExportTimer.Dispose();
            _cacheStatusTimer.Stop(); // NUEVO
            _cacheStatusTimer.Dispose(); // NUEVO
            _eventHandler.Dispose();
            base.OnFormClosing(e);
        }
    }
    
    /// <summary>
    /// Configuración del plugin
    /// </summary>
    public class WabiSabiConfig
    {
        public string OutputPath { get; set; } = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), 
            "WabiSabiBridge"
        );
        
        public bool AutoExport { get; set; } = false;
        public bool ExportDepth { get; set; } = false;
        public int DepthResolution { get; set; } = 512;
        public int DepthQuality { get; set; } = 1; // 0=Rápida, 1=Normal, 2=Alta
        public bool AutoDepthRange { get; set; } = true;
        public double DepthRangeDistance { get; set; } = 50.0;
        public bool UseGpuAcceleration { get; set; } = true;
        public bool UseGeometryExtraction { get; set; } = false;

        private static string ConfigPath => Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "WabiSabiBridge",
            "config.json"
        );
        
        public static WabiSabiConfig Load()
        {
            try
            {
                if (File.Exists(ConfigPath))
                {
                    string json = File.ReadAllText(ConfigPath);
                    return JsonConvert.DeserializeObject<WabiSabiConfig>(json) ?? new WabiSabiConfig();
                }
            }
            catch
            {
                // Ignorar errores al cargar la configuración y devolver una nueva
            }
            
            return new WabiSabiConfig();
        }
        
        public void Save()
        {
            try
            {
                string? dir = Path.GetDirectoryName(ConfigPath);
                if (dir != null && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                
                string json = JsonConvert.SerializeObject(this, Formatting.Indented);
                File.WriteAllText(ConfigPath, json);
            }
            catch 
            {
                // Ignorar errores al guardar la configuración
            }
        }
    }
}