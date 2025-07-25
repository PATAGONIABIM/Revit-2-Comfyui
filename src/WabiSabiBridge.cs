// WabiSabiBridge.cs - Implementación con soporte completo para Crop View v0.3.3 Fixed y arquitectura de alto rendimiento
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using System.Drawing.Imaging;
using System.Diagnostics;
using Newtonsoft.Json;
using System.Drawing;
using WabiSabiBridge.UI;

// Revit API
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Autodesk.Revit.DB.Events;
using Autodesk.Revit.UI.Events;


// Windows Forms con alias para evitar conflictos
using WinForms = System.Windows.Forms;
using Drawing = System.Drawing;

// Extractores
using WabiSabiBridge.Extractors;
using WabiSabiBridge.Extractors.Gpu;
using WabiSabiBridge.Extractors.Cache;
using ComputeSharp;

// --- INICIO DE NUEVOS USINGS ---
using WabiSabiBridge.DirectContext;
using Autodesk.Revit.DB.ExternalService;
using Autodesk.Revit.DB.DirectContext3D;
using System.ComponentModel;
using System.Threading;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using TaskDialog = Autodesk.Revit.UI.TaskDialog;
using System.Collections.Concurrent;
using System.Text.RegularExpressions;
// --- FIN DE NUEVOS USINGS ---

namespace WabiSabiBridge
{
    #region Clases de la Interfaz de Usuario y Comandos (Sin cambios)

    /// <summary>
    /// Comando principal del plugin WabiSabi Bridge
    /// </summary>
    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    public class WabiSabiDiagnosticCommand : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            try
            {
                WabiSabiLogger.Log("=== DIAGNÓSTICO WABISABI ===", LogLevel.Info);
                
                // Ejecutar diagnóstico completo
                var diagnostics = WabiSabiBridgeApp.RunDiagnostics();
                
                // Mostrar resultados
                string diagMessage = "Diagnóstico completado:\n\n" + diagnostics + 
                                "\n\nVer archivo de log para detalles completos.";
                
                TaskDialog td = new TaskDialog("WabiSabi Diagnóstico")
                {
                    MainContent = diagMessage,
                    MainIcon = Autodesk.Revit.UI.TaskDialogIcon.TaskDialogIconInformation
                };
                
                td.AddCommandLink(TaskDialogCommandLinkId.CommandLink1, "Abrir archivo de log");
                td.AddCommandLink(TaskDialogCommandLinkId.CommandLink2, "Cerrar");
                
                TaskDialogResult result = td.Show();
                
                if (result == TaskDialogResult.CommandLink1)
                {
                    WabiSabiLogger.ShowLogFile();
                }
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en comando de diagnóstico", ex);
                message = ex.Message ?? "Ocurrió un error inesperado en el diagnóstico."; // <-- LÍNEA CORREGIDA
                return Result.Failed;
            }
        }
    }
    
    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    public class WabiSabiBridgeCommand : IExternalCommand
    {
        private static WabiSabiBridgeWindow? _window;
        
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            try
            {
                UIApplication uiApp = commandData.Application;
                
                // CAMBIO: Uso de coincidencia de patrones para la comprobación de nulidad.
                if (uiApp.ActiveUIDocument.Document.ActiveView is not View3D)
                {
                    Autodesk.Revit.UI.TaskDialog.Show("WabiSabi Bridge", "Por favor, activa una vista 3D antes de ejecutar el comando.");
                    return Result.Failed;
                }
                
                // CAMBIO: Uso de coincidencia de patrones para la comprobación de nulidad.
                if (_window is null || _window.IsDisposed)
                {
                    // La ventana ahora usa los eventos estáticos de la App
                    _window = new WabiSabiBridgeWindow(
                        uiApp, 
                        WabiSabiBridgeApp.WabiSabiEvent!, 
                        WabiSabiBridgeApp.WabiSabiEventHandler!
                    );
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
        public int DepthQuality { get; set; } = 1;
        public bool AutoDepthRange { get; set; } = true;
        public double DepthRangeDistance { get; set; } = 50.0;
        public bool UseGpuAcceleration { get; set; } = true;
        public bool UseGeometryExtraction { get; set; } = false;
        public Action<string, Drawing.Color>? UpdateStatusCallback { get; set; }
        public bool SaveTimestampedRender { get; set; }
        public bool SaveTimestampedDepth { get; set; }

        private DepthExtractor? _depthExtractor = null;
        private DepthExtractorFast? _depthExtractorFast = null;
        private GpuAccelerationManager? _gpuManager;
        
        public async void Execute(UIApplication app)
        {
            try
            {
                UpdateStatusCallback?.Invoke("Capturando vista...", System.Drawing.Color.Blue);
                UIDocument uiDoc = app.ActiveUIDocument;
                Document doc = uiDoc.Document;

                if (doc.ActiveView is not View3D view3D) return;

                // --- TAREA 1: Recolectar datos de Revit ---
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmssfff");
                var orientation = view3D.GetOrientation();
                bool isCropActive = view3D.CropBoxActive;
                BoundingBoxUV outline = view3D.Outline;
                var uiView = uiDoc.GetOpenUIViews().FirstOrDefault(v => v.ViewId == view3D.Id);
                if (uiView == null) return;

                double trueAspectRatio = isCropActive
                    ? ((outline.Max.V - outline.Min.V) > 1e-9 ? (outline.Max.U - outline.Min.U) / (outline.Max.V - outline.Min.V) : 1.0)
                    : (double)(uiView.GetWindowRectangle().Right - uiView.GetWindowRectangle().Left) / (uiView.GetWindowRectangle().Bottom - uiView.GetWindowRectangle().Top);

                int targetWidth = this.DepthResolution;
                int targetHeight = (int)(Math.Round((targetWidth / trueAspectRatio) / 2.0) * 2.0);

                // --- TAREA 2: Exportar imagen cruda a un archivo temporal ---
                string tempRenderFileName = $"wabisabi_temp_{timestamp}.png";
                string tempRenderPath = Path.Combine(Path.GetTempPath(), tempRenderFileName);

                var options = new ImageExportOptions
                {
                    FilePath = tempRenderPath.Replace(".png", ""),
                    HLRandWFViewsFileType = ImageFileType.PNG,
                    ImageResolution = ImageResolution.DPI_150,
                    ZoomType = ZoomFitType.Zoom,
                    ExportRange = ExportRange.VisibleRegionOfCurrentView,
                };
                doc.ExportImage(options);

                // --- TAREA 3: CALCULAR DATOS DE PROFUNDIDAD Y/O LÍNEAS (en el hilo de Revit) ---
                double[,]? depthData = null;
                // NOTA PARA SOLUCIONAR EL ERROR: El error 'Object reference...' se debe a que el control _depthRangeTrackBar
                // no está inicializado en la clase WabiSabiBridgeWindow. Debe añadir su creación en el método InitializeComponent.
                if (this.ExportDepth)
                {
                    UpdateStatusCallback?.Invoke("Calculando profundidad/líneas...", System.Drawing.Color.Blue);
                    try
                    {
                        IList<XYZ> viewCorners = GetEffectiveViewCorners(uiView);
                        _gpuManager ??= new GpuAccelerationManager(null);

                        bool useFallbackMode = false;

                        // --- INICIO DE ACTUALIZACIÓN: Integración de ModernProgressDialog ---
                        if (DepthQuality == 2 && UseGpuAcceleration && _gpuManager?.IsGpuAvailable == true) // Calidad "Alta (Geometría/GPU)"
                        {
                            (double[,]? GpuDepthData, bool RenderedOnGpu) result = (null, false);
                            using (var progressDialog = new ModernProgressDialog("Procesando Geometría (GPU)"))
                            {
                                try
                                {
                                    result = await progressDialog.RunAsync(async (progress, cancellationToken) =>
                                    {
                                        // --- Etapa 1: Caché de geometría (0% -> 50%) ---
                                        var cacheManager = GeometryCacheManager.Instance;
                                        Action<string, System.Drawing.Color> progressCallback = (msg, color) =>
                                        {
                                            int percent = 0;
                                            if (System.Text.RegularExpressions.Regex.Match(msg, @"(\d+)%") is var match && match.Success)
                                            {
                                                percent = int.Parse(match.Groups[1].Value);
                                            }
                                            // Escalar el progreso a la primera mitad de la barra
                                            progress.Report((percent / 2, msg));
                                        };

                                        var cachedData = await Task.Run(() => cacheManager.EnsureCacheIsValid(doc, view3D, progressCallback), cancellationToken);
                                        cancellationToken.ThrowIfCancellationRequested();

                                        if (!cacheManager.IsCacheValid || cachedData.TriangleCount == 0)
                                        {
                                            throw new InvalidOperationException("No se pudo generar un caché de geometría válido.");
                                        }

                                        // Configuración para el renderizado
                                        var eyePosition = view3D.GetOrientation().EyePosition;
                                        XYZ bottomLeft = viewCorners[0];
                                        XYZ topRight = viewCorners[1];
                                        var upDir = orientation.UpDirection.Normalize();
                                        var rightDir = orientation.ForwardDirection.CrossProduct(upDir).Normalize();
                                        XYZ viewWidthVec = (topRight - bottomLeft).DotProduct(rightDir) * rightDir;
                                        XYZ viewHeightVec = (topRight - bottomLeft).DotProduct(upDir) * upDir;
                                        double minDepth = 0.1;
                                        double maxDepth = AutoDepthRange ? 
                                            (view3D.CropBox.Enabled ? (view3D.CropBox.Max - view3D.CropBox.Min).GetLength() * 1.2 : 100.0) 
                                            : DepthRangeDistance;

                                        var config = new RayTracingConfig
                                        {
                                            EyePosition = new ComputeSharp.Float3((float)eyePosition.X, (float)eyePosition.Y, (float)eyePosition.Z),
                                            ViewDirection = new ComputeSharp.Float3((float)(bottomLeft.X - eyePosition.X), (float)(bottomLeft.Y - eyePosition.Y), (float)(bottomLeft.Z - eyePosition.Z)),
                                            RightDirection = new ComputeSharp.Float3((float)viewWidthVec.X, (float)viewWidthVec.Y, (float)viewWidthVec.Z),
                                            UpDirection = new ComputeSharp.Float3((float)viewHeightVec.X, (float)viewHeightVec.Y, (float)viewHeightVec.Z),
                                            Width = targetWidth, Height = targetHeight, MinDepth = (float)minDepth, MaxDepth = (float)maxDepth
                                        };

                                        // --- Etapa 2: Renderizado de líneas (50% -> 75%) ---
                                        progress.Report((50, "Renderizando líneas con GPU..."));
                                        float[] gpuLineBuffer = await _gpuManager.ExecuteLineRenderAsync(cachedData.GeometryMmf, cachedData.VertexCount, cachedData.TriangleCount, config);
                                        cancellationToken.ThrowIfCancellationRequested();
                                        GenerateImageFromGpuBuffer(gpuLineBuffer, targetWidth, targetHeight, this.OutputPath, timestamp);
                                        
                                        // --- Etapa 3: Renderizado de profundidad (75% -> 95%) ---
                                        progress.Report((75, "Calculando profundidad con GPU..."));
                                        float[] gpuDepthBuffer = await _gpuManager.ExecuteDepthRayTracingFromCacheAsync(cachedData.GeometryMmf, cachedData.VertexCount, cachedData.TriangleCount, config);
                                        cancellationToken.ThrowIfCancellationRequested();
                                        
                                        // --- Etapa 4: Conversión de datos (95% -> 100%) ---
                                        progress.Report((95, "Finalizando..."));
                                        var finalDepthData = new double[targetHeight, targetWidth];
                                        Parallel.For(0, targetHeight, y =>
                                        {
                                            for (int x = 0; x < targetWidth; x++)
                                            {
                                                int idx = y * targetWidth + x;
                                                finalDepthData[y, x] = gpuDepthBuffer[idx];
                                            }
                                        });
                                        
                                        progress.Report((100, "¡Completado!"));
                                        await Task.Delay(500, cancellationToken); // Pausa para ver el mensaje final

                                        return (finalDepthData, true);
                                    });

                                    // Asignar resultados después de que el diálogo se complete
                                    depthData = result.GpuDepthData;
                                    if (result.RenderedOnGpu)
                                    {
                                        // La imagen de render ya fue generada por la GPU, no necesitamos el temporal.
                                        tempRenderPath = "";
                                    }
                                }
                                catch (OperationCanceledException)
                                {
                                    UpdateStatusCallback?.Invoke("Operación cancelada.", Drawing.Color.Orange);
                                    WabiSabiLogger.Log("Exportación GPU cancelada por el usuario.", LogLevel.Info);
                                    return; // Salir del método Execute
                                }
                                catch (Exception ex)
                                {
                                    UpdateStatusCallback?.Invoke($"Error en proceso GPU: {ex.Message}", Drawing.Color.Red);
                                    WabiSabiLogger.LogError("Error durante la operación con ModernProgressDialog", ex);
                                    return; // Salir del método Execute
                                }
                            }
                        }
                        else // Si no es modo Alta/GPU, o si hay fallback
                        {
                            if (DepthQuality == 2 && !(UseGpuAcceleration && _gpuManager?.IsGpuAvailable == true))
                            {
                                UpdateStatusCallback?.Invoke("GPU no disponible, usando modo Normal.", Drawing.Color.Orange);
                                useFallbackMode = true;
                            }

                            // Usar el modo apropiado según la calidad
                            int effectiveQuality = useFallbackMode ? 1 : DepthQuality;

                            _depthExtractorFast ??= new DepthExtractorFast(app, DepthResolution, effectiveQuality == 0 ? 4 : 2);
                            _depthExtractorFast.AutoDepthRange = this.AutoDepthRange;
                            _depthExtractorFast.ManualDepthDistance = this.DepthRangeDistance;
                            _depthExtractorFast.UseGpuAcceleration = this.UseGpuAcceleration;
                            depthData = _depthExtractorFast.ExtractDepthMap(view3D, targetWidth, targetHeight, viewCorners);
                        }
                        // --- FIN DE ACTUALIZACIÓN ---
                    }
                    catch (Exception ex)
                    {
                        UpdateStatusCallback?.Invoke($"Error calculando: {ex.Message}", System.Drawing.Color.Orange);
                        WabiSabiLogger.LogError("Error durante el cálculo de profundidad/líneas", ex);
                    }
                }

                // --- TAREA 4: Crear y encolar el trabajo (sin cambios) ---
                var job = new ExportJob
                {
                    TempRenderPath = tempRenderPath ?? "",
                    TargetWidth = targetWidth,
                    TargetHeight = targetHeight,
                    IsCropActive = isCropActive,
                    OutlineMinU = outline.Min.U, OutlineMinV = outline.Min.V, OutlineMaxU = outline.Max.U, OutlineMaxV = outline.Max.V,
                    FinalOutputPath = this.OutputPath,
                    Timestamp = timestamp.Substring(0, 15),
                    SaveTimestampedRender = this.SaveTimestampedRender,
                    SaveTimestampedDepth = this.SaveTimestampedDepth,
                    ViewName = view3D.Name,
                    ViewScale = view3D.Scale,
                    DetailLevel = view3D.DetailLevel.ToString(),
                    DisplayStyle = view3D.DisplayStyle.ToString(),
                    CamEyeX = orientation.EyePosition.X, CamEyeY = orientation.EyePosition.Y, CamEyeZ = orientation.EyePosition.Z,
                    CamForwardX = orientation.ForwardDirection.X, CamForwardY = orientation.ForwardDirection.Y, CamForwardZ = orientation.ForwardDirection.Z,
                    CamUpX = orientation.UpDirection.X, CamUpY = orientation.UpDirection.Y, CamUpZ = orientation.UpDirection.Z,
                    ProjectName = doc.Title,
                    ProjectPath = doc.PathName,
                    GpuAccelerated = this.UseGpuAcceleration && this.DepthQuality == 2,
                    DepthData = depthData,
                    EventHandler = this
                };

                // Solo encolar si hay algo que procesar
                if (!string.IsNullOrEmpty(job.TempRenderPath) || job.DepthData != null)
                {
                    WabiSabiBridgeApp._exportQueue.Enqueue(job);
                    UpdateStatusCallback?.Invoke($"Encolado: {job.Timestamp}", System.Drawing.Color.CornflowerBlue);
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error crítico en ExportEventHandler.Execute", ex);
                UpdateStatusCallback?.Invoke($"Error al encolar: {ex.Message}", System.Drawing.Color.Red);
            }
        }
        
        // CAMBIO: El método se marca como static y se retiran parámetros no utilizados.
        private static IList<XYZ> GetEffectiveViewCorners(UIView uiView)
        {
            return uiView.GetZoomCorners();
        }

        private void GenerateImageFromGpuBuffer(float[] buffer, int width, int height, string outputPath, string timestamp)
        {
            if (buffer == null || buffer.Length != width * height * 4)
            {
                WabiSabiLogger.LogError($"Buffer inválido: esperado {width * height * 4} elementos, recibido {buffer?.Length ?? 0}");
                return;
            }
            using var finalBitmap = new Drawing.Bitmap(width, height, PixelFormat.Format32bppArgb);
            BitmapData bmpData = finalBitmap.LockBits(new Drawing.Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, finalBitmap.PixelFormat);

            try
            {
                int pixelCount = width * height;
                byte[] byteBuffer = new byte[pixelCount * 4];

                Parallel.For(0, pixelCount, i =>
                {
                    int srcIdx = i * 4;
                    int dstIdx = i * 4;

                    byteBuffer[dstIdx] = (byte)(Math.Max(0, Math.Min(255, buffer[srcIdx + 3] * 255)));     // A
                    byteBuffer[dstIdx + 1] = (byte)(Math.Max(0, Math.Min(255, buffer[srcIdx] * 255)));         // R
                    byteBuffer[dstIdx + 2] = (byte)(Math.Max(0, Math.Min(255, buffer[srcIdx + 1] * 255)));     // G
                    byteBuffer[dstIdx + 3] = (byte)(Math.Max(0, Math.Min(255, buffer[srcIdx + 2] * 255)));     // B
                });

                System.Runtime.InteropServices.Marshal.Copy(byteBuffer, 0, bmpData.Scan0, byteBuffer.Length);
            }
            finally
            {
                finalBitmap.UnlockBits(bmpData);
            }

            string targetFile = Path.Combine(outputPath, "current_render.png");
            finalBitmap.Save(targetFile, System.Drawing.Imaging.ImageFormat.Png);

            if (SaveTimestampedRender)
            {
                finalBitmap.Save(Path.Combine(outputPath, $"render_{timestamp}.png"), System.Drawing.Imaging.ImageFormat.Png);
            }
            WabiSabiLogger.Log($"Imagen de líneas (GPU) exportada: {targetFile}", LogLevel.Debug);
        }

        private void ExportHiddenLineImage(Document doc, View3D view3D, string outputPath, string timestamp,
            int targetWidth, int targetHeight, bool isCropActive)
        {
            WabiSabiLogger.LogDiagnostic("Export",
                $"ExportHiddenLineImage - Res: {targetWidth}x{targetHeight}, Crop: {isCropActive}");

            var options = new ImageExportOptions
            {
                FilePath = Path.Combine(outputPath, $"render_{timestamp}"),
                HLRandWFViewsFileType = ImageFileType.PNG,
                ImageResolution = ImageResolution.DPI_150,
                ZoomType = ZoomFitType.Zoom,
                ExportRange = ExportRange.VisibleRegionOfCurrentView,
            };

            doc.ExportImage(options);

            string generatedFile = Path.Combine(outputPath, $"render_{timestamp}.png");
            string targetFile = Path.Combine(outputPath, "current_render.png");

            if (!File.Exists(generatedFile)) return;

            using var originalImage = Drawing.Image.FromFile(generatedFile);
            using var finalBitmap = new Drawing.Bitmap(targetWidth, targetHeight);

            finalBitmap.SetResolution(originalImage.HorizontalResolution, originalImage.VerticalResolution);
            using (var g = Drawing.Graphics.FromImage(finalBitmap))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;
                g.TextRenderingHint = System.Drawing.Text.TextRenderingHint.AntiAliasGridFit;

                Drawing.Rectangle sourceRect = isCropActive
                    ? new Drawing.Rectangle(
                        (int)(view3D.Outline.Min.U * originalImage.Width),
                        (int)((1 - view3D.Outline.Max.V) * originalImage.Height),
                        (int)((view3D.Outline.Max.U - view3D.Outline.Min.U) * originalImage.Width),
                        (int)((view3D.Outline.Max.V - view3D.Outline.Min.V) * originalImage.Height))
                    : new Drawing.Rectangle(0, 0, originalImage.Width, originalImage.Height);

                float srcAspect = (float)sourceRect.Width / sourceRect.Height;
                float dstAspect = (float)targetWidth / targetHeight;

                Drawing.Rectangle destRect = srcAspect > dstAspect
                    ? new Drawing.Rectangle(0, (targetHeight - (int)(targetWidth / srcAspect)) / 2, targetWidth, (int)(targetWidth / srcAspect))
                    : new Drawing.Rectangle((targetWidth - (int)(targetHeight * srcAspect)) / 2, 0, (int)(targetHeight * srcAspect), targetHeight);

                g.Clear(Drawing.Color.FromArgb(255, 40, 43, 48));
                g.DrawImage(originalImage, destRect, sourceRect, Drawing.GraphicsUnit.Pixel);
            }
            finalBitmap.Save(targetFile, System.Drawing.Imaging.ImageFormat.Png);

            if (!SaveTimestampedRender)
            {
                try { File.Delete(generatedFile); } catch { }
            }
            WabiSabiLogger.Log($"Imagen de líneas ocultas exportada: {targetFile}", LogLevel.Debug);
        }
        
        private void ExportMetadata(Document doc, View3D view3D, string outputPath, string timestamp)
        {
            WabiSabiLogger.LogDiagnostic("Export", "Exportando metadata...");

            var orientation = view3D.GetOrientation();
            var eyePos = orientation.EyePosition;
            var forwardDir = orientation.ForwardDirection;
            var targetPos = eyePos + forwardDir.Multiply(10);
            var upVec = orientation.UpDirection;

            var metadata = new
            {
                timestamp,
                view3D.Name,
                view_type = "3D",
                view3D.Scale,
                detail_level = view3D.DetailLevel.ToString(),
                display_style = view3D.DisplayStyle.ToString(),
                crop_box_active = view3D.CropBoxActive,
                camera = new
                {
                    eye_position = new { x = eyePos.X, y = eyePos.Y, z = eyePos.Z },
                    target_position = new { x = targetPos.X, y = targetPos.Y, z = targetPos.Z },
                    up_vector = new { x = upVec.X, y = upVec.Y, z = upVec.Z }
                },
                project_info = new { name = doc.Title, path = doc.PathName },
                gpu_acceleration = UseGpuAcceleration
            };
            File.WriteAllText(Path.Combine(outputPath, "current_metadata.json"), JsonConvert.SerializeObject(metadata, Formatting.Indented));
            WabiSabiLogger.Log("Metadata exportada", LogLevel.Debug);
        }

        private static void CreateNotificationFile(string outputPath, string timestamp)
        {
            WabiSabiLogger.LogDiagnostic("Export", "Creando archivo de notificación...");
            File.WriteAllText(Path.Combine(outputPath, "last_update.txt"), timestamp);
            WabiSabiLogger.Log("Archivo de notificación creado", LogLevel.Debug);
        }

        public void Dispose() { _depthExtractor?.Dispose(); _depthExtractorFast?.Dispose(); }

        public void GenerateDepthImage(double[,] depthData, int width, int height, string outputPath, string timestamp, bool saveTimestampedDepth)
        {
            using var depthMap = new Drawing.Bitmap(width, height, PixelFormat.Format24bppRgb);
            BitmapData bmpData = depthMap.LockBits(new Drawing.Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, depthMap.PixelFormat);
            unsafe
            {
                byte* ptr = (byte*)bmpData.Scan0;
                Parallel.For(0, height, y =>
                {
                    byte* row = ptr + (y * bmpData.Stride);
                    for (int x = 0; x < width; x++)
                    {
                        byte depthValue = (byte)(Math.Max(0.0, Math.Min(1.0, depthData[y, x])) * 255);
                        row[x * 3] = depthValue; row[x * 3 + 1] = depthValue; row[x * 3 + 2] = depthValue;
                    }
                });
            }
            depthMap.UnlockBits(bmpData);

            var encoderParams = new EncoderParameters(1) { Param = { [0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Quality, 90L) } };
            var pngCodec = ImageCodecInfo.GetImageEncoders().First(c => c.FormatID == System.Drawing.Imaging.ImageFormat.Png.Guid);

            string currentDepthPath = Path.Combine(outputPath, "current_depth.png");
            depthMap.Save(currentDepthPath, pngCodec, encoderParams);

            if (saveTimestampedDepth)
            {
                depthMap.Save(Path.Combine(outputPath, $"depth_{timestamp}.png"), pngCodec, encoderParams);
            }
            WabiSabiLogger.Log($"Mapa de profundidad guardado: {currentDepthPath}", LogLevel.Debug);
        }

        private double CalculateMaxDepth(View3D view3D, XYZ eyePosition, XYZ forwardDirection, XYZ viewCenter)
        {
            if (!AutoDepthRange) return DepthRangeDistance;
            if (!view3D.IsPerspective) return 50.0;

            double distanceToTarget = -1.0;
            if (view3D.CropBox.Enabled)
            {
                var cropCenter = (view3D.CropBox.Min + view3D.CropBox.Max) / 2.0;
                double distanceAlongView = (cropCenter - eyePosition).DotProduct(forwardDirection);
                if (distanceAlongView > 0) distanceToTarget = distanceAlongView;
            }
            if (distanceToTarget < 0)
            {
                double distanceToViewPlane = (viewCenter - eyePosition).DotProduct(forwardDirection);
                if (distanceToViewPlane > 0) distanceToTarget = distanceToViewPlane;
            }

            return distanceToTarget > 0 ? distanceToTarget * 1.2 : (view3D.Outline.Max.U - view3D.Outline.Min.U) / (2.0 * Math.Tan(Math.PI / 6.0)) * 1.5;
        }

        public string GetName() => "WabiSabi Bridge Export Event";
    }
    /// Ventana principal del plugin
    /// </summary>
    public class WabiSabiBridgeWindow : WinForms.Form
    {
        private readonly UIApplication _uiApp;
        private readonly ExternalEvent _externalEvent;
        private readonly ExportEventHandler _eventHandler;
        private readonly WabiSabiConfig _config;

        private WinForms.TextBox _outputPathTextBox = null!;
        private WinForms.CheckBox _autoExportCheckBox = null!;
        private WinForms.CheckBox _exportDepthCheckBox = null!;
        private WinForms.ComboBox _depthResolutionCombo = null!;
        private WinForms.ComboBox _depthQualityCombo = null!;
        private WinForms.TrackBar _depthRangeTrackBar = null!;
        private WinForms.Label _depthRangeValueLabel = null!;
        private WinForms.CheckBox _autoDepthCheckBox = null!;
        private WinForms.CheckBox _gpuAccelerationCheckBox = null!;
        private WinForms.CheckBox _geometryExtractionCheckBox = null!;
        private WinForms.CheckBox _saveRenderCheckBox = null!;
        private WinForms.CheckBox _saveDepthCheckBox = null!;
        private WinForms.Label _statusLabel = null!;
        private WinForms.Label _cacheStatusLabel = null!;
        private WinForms.Button _clearCacheButton = null!;
        private WinForms.Timer _cacheStatusTimer = null!;
        
        public WabiSabiBridgeWindow(UIApplication uiApp, ExternalEvent externalEvent, ExportEventHandler eventHandler)
        {
            _uiApp = uiApp;
            _externalEvent = externalEvent;
            _eventHandler = eventHandler;
            _eventHandler.UpdateStatusCallback = UpdateStatus;
            _config = WabiSabiConfig.Load();
            InitializeComponent();
            CheckGpuStatus();
        }
        
        private void InitializeComponent()
        {
            Text = "WabiSabi Bridge v0.3.3";
            Size = new Drawing.Size(500, 720);
            StartPosition = WinForms.FormStartPosition.CenterScreen;
            FormBorderStyle = WinForms.FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            
            // Usar un panel principal con margen
            var mainPanel = new WinForms.Panel
            {
                Dock = WinForms.DockStyle.Fill,
                Padding = new WinForms.Padding(15),
                BackColor = Drawing.Color.White
            };
            
            // === SECCIÓN 1: Carpeta de salida ===
            var outputGroup = new WinForms.GroupBox
            {
                Text = "Configuración de Salida",
                Location = new Drawing.Point(15, 15),
                Size = new Drawing.Size(455, 80),
                Font = new Drawing.Font("Segoe UI", 9F, Drawing.FontStyle.Bold)
            };
            
            var outputLabel = new WinForms.Label
            {
                Text = "Carpeta:",
                Location = new Drawing.Point(15, 30),
                Size = new Drawing.Size(60, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                TextAlign = Drawing.ContentAlignment.MiddleLeft
            };
            
            _outputPathTextBox = new WinForms.TextBox
            {
                Location = new Drawing.Point(80, 30),
                Size = new Drawing.Size(280, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                Text = _config.OutputPath
            };
            
            var browseButton = new WinForms.Button
            {
                Text = "Examinar...",
                Location = new Drawing.Point(365, 29),
                Size = new Drawing.Size(75, 27),
                Font = new Drawing.Font("Segoe UI", 9F),
                FlatStyle = WinForms.FlatStyle.System
            };
            browseButton.Click += (s, e) => {
                using (var dialog = new WinForms.FolderBrowserDialog { SelectedPath = _outputPathTextBox.Text })
                    if (dialog.ShowDialog() == WinForms.DialogResult.OK) {
                        _outputPathTextBox.Text = dialog.SelectedPath;
                        _config.OutputPath = dialog.SelectedPath;
                        _config.Save();
                    }
            };
            
            outputGroup.Controls.AddRange(new WinForms.Control[] { outputLabel, _outputPathTextBox, browseButton });
            
            // === BOTÓN PRINCIPAL DE EXPORTAR ===
            var exportButton = new WinForms.Button
            {
                Text = "EXPORTAR VISTA ACTUAL",
                Location = new Drawing.Point(15, 105),
                Size = new Drawing.Size(455, 45),
                Font = new Drawing.Font("Segoe UI", 10F, Drawing.FontStyle.Bold),
                BackColor = Drawing.Color.FromArgb(0, 120, 215),
                ForeColor = Drawing.Color.White,
                FlatStyle = WinForms.FlatStyle.Flat,
                Cursor = WinForms.Cursors.Hand
            };
            exportButton.FlatAppearance.BorderSize = 0;
            exportButton.Click += (s, e) => ExportCurrentView();
            
            // === SECCIÓN 2: Opciones de exportación ===
            var exportOptionsGroup = new WinForms.GroupBox
            {
                Text = "Opciones de Profundidad",
                Location = new Drawing.Point(15, 165),
                Size = new Drawing.Size(455, 180),
                Font = new Drawing.Font("Segoe UI", 9F, Drawing.FontStyle.Bold)
            };
            
            _exportDepthCheckBox = new WinForms.CheckBox
            {
                Text = "Exportar mapa de profundidad",
                Location = new Drawing.Point(15, 25),
                Size = new Drawing.Size(200, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                Checked = _config.ExportDepth
            };
            
            var resolutionLabel = new WinForms.Label
            {
                Text = "Resolución:",
                Location = new Drawing.Point(15, 55),
                Size = new Drawing.Size(80, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                TextAlign = Drawing.ContentAlignment.MiddleLeft
            };
            
            _depthResolutionCombo = new WinForms.ComboBox
            {
                Location = new Drawing.Point(100, 55),
                Size = new Drawing.Size(120, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                DropDownStyle = WinForms.ComboBoxStyle.DropDownList
            };
            _depthResolutionCombo.Items.AddRange(new object[] { "256", "512", "1024", "2048" });
            _depthResolutionCombo.Text = _config.DepthResolution.ToString();
            
            var qualityLabel = new WinForms.Label
            {
                Text = "Calidad:",
                Location = new Drawing.Point(240, 55),
                Size = new Drawing.Size(60, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                TextAlign = Drawing.ContentAlignment.MiddleLeft
            };
            
            _depthQualityCombo = new WinForms.ComboBox
            {
                Location = new Drawing.Point(305, 55),
                Size = new Drawing.Size(135, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                DropDownStyle = WinForms.ComboBoxStyle.DropDownList
            };
            _depthQualityCombo.Items.AddRange(new object[] { "Baja", "Media", "Alta (GPU)" });
            _depthQualityCombo.SelectedIndex = _config.DepthQuality;
            
            _gpuAccelerationCheckBox = new WinForms.CheckBox
            {
                Text = "Aceleración GPU",
                Location = new Drawing.Point(15, 90),
                Size = new Drawing.Size(150, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                Checked = _config.UseGpuAcceleration
            };
            
            _geometryExtractionCheckBox = new WinForms.CheckBox
            {
                Text = "Modo Experimental",
                Location = new Drawing.Point(170, 90),
                Size = new Drawing.Size(150, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                ForeColor = Drawing.Color.DarkOrange,
                Checked = _config.UseGeometryExtraction
            };
            
            var timeEstimateLabel = new WinForms.Label
            {
                Name = "timeEstimateLabel",
                Location = new Drawing.Point(15, 120),
                Size = new Drawing.Size(425, 20),
                Font = new Drawing.Font("Segoe UI", 8F, Drawing.FontStyle.Italic),
                ForeColor = Drawing.Color.Gray,
                TextAlign = Drawing.ContentAlignment.MiddleLeft
            };
            
            var gpuStatusLabel = new WinForms.Label
            {
                Name = "gpuStatusLabel",
                Location = new Drawing.Point(15, 145),
                Size = new Drawing.Size(425, 20),
                Font = new Drawing.Font("Segoe UI", 8F),
                ForeColor = Drawing.Color.Gray,
                TextAlign = Drawing.ContentAlignment.MiddleLeft
            };
            
            exportOptionsGroup.Controls.AddRange(new WinForms.Control[] {
                _exportDepthCheckBox, resolutionLabel, _depthResolutionCombo,
                qualityLabel, _depthQualityCombo, _gpuAccelerationCheckBox,
                _geometryExtractionCheckBox, timeEstimateLabel, gpuStatusLabel
            });
            
            // === SECCIÓN 3: Información del caché ===
            var cacheGroup = new WinForms.GroupBox
            {
                Text = "Estado del Caché",
                Location = new Drawing.Point(15, 355),
                Size = new Drawing.Size(455, 80),
                Font = new Drawing.Font("Segoe UI", 9F, Drawing.FontStyle.Bold)
            };
            
            _cacheStatusLabel = new WinForms.Label
            {
                Location = new Drawing.Point(15, 25),
                Size = new Drawing.Size(320, 40),
                Font = new Drawing.Font("Segoe UI", 9F),
                TextAlign = Drawing.ContentAlignment.MiddleLeft,
                Text = "Caché: No inicializado"
            };
            
            _clearCacheButton = new WinForms.Button
            {
                Text = "Limpiar",
                Location = new Drawing.Point(345, 30),
                Size = new Drawing.Size(95, 30),
                Font = new Drawing.Font("Segoe UI", 9F),
                BackColor = Drawing.Color.FromArgb(255, 200, 200),
                FlatStyle = WinForms.FlatStyle.Flat,
                Cursor = WinForms.Cursors.Hand
            };
            _clearCacheButton.FlatAppearance.BorderSize = 0;
            _clearCacheButton.Click += (s, e) => {
                if (WinForms.MessageBox.Show("¿Limpiar el caché de geometría?\n\nEsto forzará la reconstrucción en la próxima exportación.",
                    "Confirmar limpieza", WinForms.MessageBoxButtons.YesNo, WinForms.MessageBoxIcon.Question) == WinForms.DialogResult.Yes)
                {
                    GeometryCacheManager.Instance.InvalidateCache();
                    UpdateStatus("Caché limpiado", Drawing.Color.Orange);
                    UpdateCacheStatus();
                }
            };
            
            cacheGroup.Controls.AddRange(new WinForms.Control[] { _cacheStatusLabel, _clearCacheButton });
            
            // === SECCIÓN 4: Opciones adicionales ===
            var additionalGroup = new WinForms.GroupBox
            {
                Text = "Opciones Adicionales",
                Location = new Drawing.Point(15, 445),
                Size = new Drawing.Size(455, 200),
                Font = new Drawing.Font("Segoe UI", 9F, Drawing.FontStyle.Bold)
            };
            
            _autoExportCheckBox = new WinForms.CheckBox
            {
                Text = "Auto-exportar al mover cámara",
                Location = new Drawing.Point(15, 25),
                Size = new Drawing.Size(250, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                Checked = _config.AutoExport
            };
            
            _autoDepthCheckBox = new WinForms.CheckBox
            {
                Text = "Rango de profundidad automático",
                Location = new Drawing.Point(15, 50),
                Size = new Drawing.Size(250, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                Checked = _config.AutoDepthRange
            };

            var depthRangeLabel = new WinForms.Label
            {
                Text = "Rango Manual:",
                Location = new Drawing.Point(15, 78),
                Size = new Drawing.Size(90, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                TextAlign = Drawing.ContentAlignment.MiddleLeft
            };

            _depthRangeTrackBar = new WinForms.TrackBar
            {
                Location = new Drawing.Point(105, 75),
                Size = new Drawing.Size(250, 25),
                Minimum = 10,
                Maximum = 500,
                TickFrequency = 10,
                Value = (int)_config.DepthRangeDistance,
                Enabled = !_config.AutoDepthRange // Se activa si el modo automático está desactivado
            };

            _depthRangeValueLabel = new WinForms.Label
            {
                Location = new Drawing.Point(365, 78),
                Size = new Drawing.Size(75, 25),
                Font = new Drawing.Font("Segoe UI", 9F, Drawing.FontStyle.Bold),
                TextAlign = Drawing.ContentAlignment.MiddleLeft,
                Text = $"{_depthRangeTrackBar.Value}m",
                Enabled = !_config.AutoDepthRange
            };
            
            _saveRenderCheckBox = new WinForms.CheckBox
            {
                Text = "Guardar renders con timestamp",
                Location = new Drawing.Point(15, 130),
                Size = new Drawing.Size(250, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                Checked = _config.SaveTimestampedRender
            };
            
            _saveDepthCheckBox = new WinForms.CheckBox
            {
                Text = "Guardar profundidad con timestamp",
                Location = new Drawing.Point(15, 155),
                Size = new Drawing.Size(250, 25),
                Font = new Drawing.Font("Segoe UI", 9F),
                Checked = _config.SaveTimestampedDepth
            };
            
            additionalGroup.Controls.AddRange(new WinForms.Control[] {
                _autoExportCheckBox, _autoDepthCheckBox, depthRangeLabel, _depthRangeTrackBar, _depthRangeValueLabel, _saveRenderCheckBox, _saveDepthCheckBox
            });
            
            // === BARRA DE ESTADO ===
            var statusPanel = new WinForms.Panel
            {
                Location = new Drawing.Point(15, 595),
                Size = new Drawing.Size(455, 50),
                BorderStyle = WinForms.BorderStyle.FixedSingle
            };
            
            _statusLabel = new WinForms.Label
            {
                Dock = WinForms.DockStyle.Fill,
                Font = new Drawing.Font("Segoe UI", 10F, Drawing.FontStyle.Bold),
                TextAlign = Drawing.ContentAlignment.MiddleCenter,
                Text = "Listo para exportar",
                ForeColor = Drawing.Color.Green
            };
            
            statusPanel.Controls.Add(_statusLabel);
            
            // Agregar todos los controles al panel principal
            mainPanel.Controls.AddRange(new WinForms.Control[] {
                outputGroup, exportButton, exportOptionsGroup, cacheGroup, additionalGroup, statusPanel
            });
            
            Controls.Add(mainPanel);
            
            // Configurar eventos
            _exportDepthCheckBox.CheckedChanged += (s, e) => {
                _config.ExportDepth = _exportDepthCheckBox.Checked;
                _config.Save();
                UpdateDepthControls();
                UpdateTimeEstimate();
            };
            
            _depthResolutionCombo.SelectedIndexChanged += (s, e) => {
                _config.DepthResolution = int.Parse(_depthResolutionCombo.Text);
                _config.Save();
                UpdateTimeEstimate();
            };
            
            _depthQualityCombo.SelectedIndexChanged += (s, e) => {
                _config.DepthQuality = _depthQualityCombo.SelectedIndex;
                _config.Save();
                UpdateTimeEstimate();
            };
            
            _gpuAccelerationCheckBox.CheckedChanged += (s, e) => {
                _config.UseGpuAcceleration = _gpuAccelerationCheckBox.Checked;
                _config.Save();
                UpdateGpuControls();
                UpdateTimeEstimate();
            };
            
            _geometryExtractionCheckBox.CheckedChanged += (s, e) => {
                _config.UseGeometryExtraction = _geometryExtractionCheckBox.Checked;
                _config.Save();
                UpdateTimeEstimate();
            };
            
            _autoExportCheckBox.CheckedChanged += (s, e) => {
                _config.AutoExport = _autoExportCheckBox.Checked;
                _config.Save();
            };
            
            _autoDepthCheckBox.CheckedChanged += (s, e) => {
                _config.AutoDepthRange = _autoDepthCheckBox.Checked;
                _config.Save();
                _depthRangeTrackBar.Enabled = !_autoDepthCheckBox.Checked;
                _depthRangeValueLabel.Enabled = !_autoDepthCheckBox.Checked;
            };
            
            _saveRenderCheckBox.CheckedChanged += (s, e) => {
                _config.SaveTimestampedRender = _saveRenderCheckBox.Checked;
                _config.Save();
            };
            
            _saveDepthCheckBox.CheckedChanged += (s, e) => {
                _config.SaveTimestampedDepth = _saveDepthCheckBox.Checked;
                _config.Save();
            };

            _depthRangeTrackBar.Scroll += (s, e) => {
                _depthRangeValueLabel.Text = $"{_depthRangeTrackBar.Value}m";
                _config.DepthRangeDistance = _depthRangeTrackBar.Value;
                // Guardar la configuración al soltar el control deslizante para no sobrecargar
            };
            _depthRangeTrackBar.MouseUp += (s, e) => {
                _config.Save();
            };
            
            // Timer para actualizar el estado del caché
            _cacheStatusTimer = new WinForms.Timer { Interval = 1000 };
            _cacheStatusTimer.Tick += (s, e) => UpdateCacheStatus();
            _cacheStatusTimer.Start();
            
            // Inicializar estados
            UpdateDepthControls();
            UpdateGpuControls();
            UpdateTimeEstimate();
            CheckGpuStatus();
            #endregion
        }
        
        public void ExportCurrentView()
        {
            try
            {
                if (InvokeRequired) { Invoke(new Action(ExportCurrentView)); return; }
            
                UpdateStatus("Preparando exportación...", Drawing.Color.Blue);
                
                _eventHandler.OutputPath = _outputPathTextBox.Text;
                _eventHandler.UiApp = _uiApp;
                _eventHandler.ExportDepth = _exportDepthCheckBox.Checked;
                _eventHandler.DepthResolution = _config.DepthResolution;
                _eventHandler.DepthQuality = _config.DepthQuality;
                _eventHandler.AutoDepthRange = _autoDepthCheckBox.Checked;
                _eventHandler.DepthRangeDistance = _depthRangeTrackBar.Value;
                _eventHandler.UseGpuAcceleration = _gpuAccelerationCheckBox.Checked;
                _eventHandler.UseGeometryExtraction = _geometryExtractionCheckBox.Checked;
                _eventHandler.SaveTimestampedRender = _config.SaveTimestampedRender;
                _eventHandler.SaveTimestampedDepth = _config.SaveTimestampedDepth;

                _externalEvent.Raise();
            }
            catch (Exception ex) { UpdateStatus($"Error: {ex.Message}", Drawing.Color.Red); }
        }

        #region Métodos de la UI
        private void UpdateDepthControls() {
            bool enabled = _exportDepthCheckBox.Checked;
            _depthResolutionCombo.Enabled = enabled; _depthQualityCombo.Enabled = enabled;
            _gpuAccelerationCheckBox.Enabled = enabled; UpdateGpuControls();
            UpdateTimeEstimate();
        }
        private void UpdateGpuControls() {
            _geometryExtractionCheckBox.Enabled = _exportDepthCheckBox.Checked && _gpuAccelerationCheckBox.Checked;
            if (!_gpuAccelerationCheckBox.Checked) _geometryExtractionCheckBox.Checked = false;
            UpdateTimeEstimate();
        }
        private void CheckGpuStatus() {
            try {
                using var gpuManager = new GpuAccelerationManager(this);
                // CAMBIO: Se usa coincidencia de patrones para hacer el código más compacto.
                if(Controls.Find("gpuStatusLabel", true).FirstOrDefault() is WinForms.Label gpuStatusLabel) {
                    gpuStatusLabel.Text = gpuManager.IsGpuAvailable ? "GPU: Disponible ✓" : "GPU: No disponible (usando CPU paralela)";
                    gpuStatusLabel.ForeColor = gpuManager.IsGpuAvailable ? Drawing.Color.Green : Drawing.Color.Orange;
                }
            } catch (Exception ex) {
                Autodesk.Revit.UI.TaskDialog.Show("Error de GPU", $"Error al verificar GPU: {ex}");
            }
        }
        private void UpdateTimeEstimate() {
            if (Controls.Find("timeEstimateLabel", true).FirstOrDefault() is not WinForms.Label timeLabel) return;

            if (!_exportDepthCheckBox.Checked) { timeLabel.Text = ""; return; }
            
            int quality = _config.DepthQuality;
            bool useGpu = _config.UseGpuAcceleration;

            // --- LÓGICA DE ESTIMACIÓN CORREGIDA ---
            if (quality == 2) // Calidad "Alta"
            {
                if (useGpu)
                {
                    timeLabel.Text = "Tiempo estimado: Muy rápido (Renderizado de líneas en GPU)";
                }
                else
                {
                    timeLabel.Text = "Tiempo estimado: ~2-5 minutos (Extracción de geometría en CPU)";
                }
                return;
            }
            
            int resIndex = _config.DepthResolution == 256 ? 0 : _config.DepthResolution == 512 ? 1 : _config.DepthResolution == 1024 ? 2 : 3;
            int[,] timeMatrix = useGpu ? new int[,] { { 1, 1, 2 }, { 1, 3, 5 }, { 2, 8, 15 }, { 8, 30, 60 } } 
                                    : new int[,] { { 1, 3, 5 }, { 3, 10, 20 }, { 10, 40, 80 }, { 40, 160, 320 } };
            int seconds = timeMatrix[resIndex, quality] * (_config.UseGeometryExtraction ? 2 : 1);
            
            timeLabel.Text = $"Tiempo estimado: ~{(seconds < 60 ? $"{seconds} segundos" : $"{seconds / 60} minutos")}{(useGpu ? " (GPU)" : " (CPU)")}";
        }
        private void UpdateCacheStatus() {
            try {
                var cacheManager = GeometryCacheManager.Instance;
                if (cacheManager.IsCacheValid) {
                    string sizeInfo = cacheManager.CacheSizeBytes > 1048576 ? $"{cacheManager.CacheSizeBytes / 1048576.0:F1}MB" : $"{cacheManager.CacheSizeBytes / 1024.0:F1}KB";
                    var elapsed = DateTime.Now - cacheManager.LastCacheTime;
                    string timeAgo = elapsed.TotalSeconds < 60 ? "hace un momento" : elapsed.TotalMinutes < 60 ? $"hace {(int)elapsed.TotalMinutes} min" : $"hace {(int)elapsed.TotalHours} h";
                    _cacheStatusLabel.Text = $"Caché: Válido ({cacheManager.VertexCount:N0} V, {cacheManager.TriangleCount:N0} T, {sizeInfo}) - {timeAgo}";
                    _cacheStatusLabel.ForeColor = Drawing.Color.Green; _clearCacheButton.Enabled = true;
                } else {
                    _cacheStatusLabel.Text = "Caché: No válido (se reconstruirá autom.)";
                    _cacheStatusLabel.ForeColor = Drawing.Color.Gray; _clearCacheButton.Enabled = false;
                }
            } catch { _cacheStatusLabel.Text = "Caché: Estado desconocido"; }
        }
        private void UpdateStatus(string message, Drawing.Color color) {
            if (InvokeRequired) { Invoke(new Action(() => UpdateStatus(message, color))); return; }
            _statusLabel.Text = message; _statusLabel.ForeColor = color;
        }
        protected override void OnFormClosing(WinForms.FormClosingEventArgs e) {
            _cacheStatusTimer.Stop(); _cacheStatusTimer.Dispose();
            _eventHandler.Dispose();
            base.OnFormClosing(e);
        }
        #endregion
    }

    

    #region Clase Principal de la Aplicación (Refactorizada y Corregida)

    /// <summary>
    /// Aplicación principal del plugin - Versión corregida con logging mejorado
    /// </summary>
    public class WabiSabiBridgeApp : IExternalApplication
    {
        private static CameraDataExtractionServer? _currentServer;
        private static readonly Dictionary<Document, CameraDataExtractionServer> _documentServers = new();
        private static LockFreeCameraRingBuffer? _cameraBuffer;
        private static MemoryMappedFile? _cameraMmf;
        private static MemoryMappedViewAccessor? _cameraAccessor;
        private static Thread? _streamerThread;
        private static CancellationTokenSource? _cancellationTokenSource;
        private const string MmfName = "WabiSabiBridge_CameraStream";
        internal static ExternalEvent? WabiSabiEvent;
        internal static ExportEventHandler? WabiSabiEventHandler;
        private static readonly WabiSabiConfig _config = WabiSabiConfig.Load();
        private static int _lastKnownSequenceNumber = -1;
        private static int _globalSequenceNumber = -1;
        private static int _lastProcessedSequenceNumber = -1;
        private static long _lastCameraMoveTimestamp = 0;
        private static long _lastExportTimestamp = 0;

        private static System.Windows.Forms.Timer? _viewInvalidationTimer;
        private static System.Windows.Forms.Timer? _directCaptureTimer;
        private static UIApplication? _currentUiApp;
        private static bool _isContinuousCaptureActive = false;
        private static long _lastInvalidationTime = 0;
        private const long INVALIDATION_INTERVAL_MS = 33; // ~30 FPS para invalidación    

        internal static readonly ConcurrentQueue<ExportJob> _exportQueue = new ConcurrentQueue<ExportJob>();
        private static Task? _exportWorker;
        private static CancellationTokenSource? _exportCts;   
        
        private static bool _isInitialized = false;
        private static int _idlingCallCount = 0;
        private static readonly HashSet<Document> _pendingRegistrations = new HashSet<Document>();
        private static ViewOrientation3D? _lastKnownOrientation;
        private static readonly double XYZ_TOLERANCE = 1e-6; // Tolerancia para comparar vectores flotantes

        private static volatile bool _invalidationRequested = false;

        public Result OnStartup(UIControlledApplication application)
        {
            try
            {
                WabiSabiLogger.Log("========================================", LogLevel.Info);
                WabiSabiLogger.Log("WabiSabi Bridge - INICIANDO", LogLevel.Info);
                WabiSabiLogger.Log($"Versión: v0.3.3 Fixed", LogLevel.Info);
                WabiSabiLogger.Log("========================================", LogLevel.Info);
                
                _isInitialized = false;
                
                if (!CreateRibbon(application)) return Result.Failed;
                if (!InitializeEventHandler()) return Result.Failed;
                if (!SubscribeToRevitEvents(application)) return Result.Failed;

                // --- CAMBIO: INICIAMOS EL SISTEMA DE STREAMING SIEMPRE ---
                // La lógica de si se exporta o no, se controlará en OnIdling.
                // Esto evita el ciclo frágil de registrar/desregistrar el servidor.
                InitializeStreamingSystem(); 

                _isInitialized = true;
                WabiSabiLogger.Log("WabiSabi Bridge iniciado con éxito", LogLevel.Info);
                // INICIAR EL HILO CONSUMIDOR DE EXPORTACIONES
                _exportCts = new CancellationTokenSource();
                _exportWorker = Task.Run(() => ProcessExportQueue(_exportCts.Token));
                WabiSabiLogger.Log("Hilo de trabajo de exportación iniciado.", LogLevel.Info);

                _isInitialized = true;
                RunDiagnostics();
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error crítico en OnStartup", ex);
                TaskDialog.Show("Error Crítico WabiSabi", $"Error durante inicio: {ex.Message}");
                return Result.Failed;
            }
        }
        /// <summary>
        /// Este evento se dispara por el Timer en el hilo de la UI, 
        /// garantizando una captura de cámara constante.
        /// </summary>
        private static void DirectCaptureTimer_Tick(object? sender, EventArgs e)
        {
            // Solo capturamos si la opción está activa y tenemos una app de Revit
            if (_isContinuousCaptureActive && _currentUiApp != null)
            {
                CaptureCameraDataDirectly(_currentUiApp.ActiveUIDocument);
            }
            else
            {
                // Si por alguna razón el tick se ejecuta cuando no debe, detenemos el timer.
                _directCaptureTimer?.Stop();
            }
        }

        /// <summary>
        /// Se ejecuta en un hilo de fondo, procesando trabajos de la cola sin bloquear la UI.
        /// </summary>
        private static void ProcessExportQueue(CancellationToken token)
        {
            WabiSabiLogger.Log("Procesador de cola de exportación activo.", LogLevel.Info);
            while (!token.IsCancellationRequested)
            {
                if (_exportQueue.TryDequeue(out ExportJob? job) && job != null)
                {
                    try
                    {
                        WabiSabiLogger.Log($"Procesando trabajo: {job.Timestamp}", LogLevel.Debug);

                        // --- 1. PROCESAR Y GUARDAR IMAGEN RENDERIZADA ---
                        // Solo procesar si hay una imagen de Revit que procesar
                        if (!string.IsNullOrWhiteSpace(job.TempRenderPath) && File.Exists(job.TempRenderPath))
                        {
                            string finalRenderPath = Path.Combine(job.FinalOutputPath, "current_render.png");
                        
                        using (var originalImage = System.Drawing.Image.FromFile(job.TempRenderPath))
                        using (var finalBitmap = new System.Drawing.Bitmap(job.TargetWidth, job.TargetHeight))
                        {
                            finalBitmap.SetResolution(originalImage.HorizontalResolution, originalImage.VerticalResolution);
                            using (var g = System.Drawing.Graphics.FromImage(finalBitmap))
                            {
                                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                                g.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                                
                                System.Drawing.Rectangle sourceRect = job.IsCropActive
                                    ? new System.Drawing.Rectangle(
                                        (int)(job.OutlineMinU * originalImage.Width),
                                        (int)((1 - job.OutlineMaxV) * originalImage.Height),
                                        (int)((job.OutlineMaxU - job.OutlineMinU) * originalImage.Width),
                                        (int)((job.OutlineMaxV - job.OutlineMinV) * originalImage.Height))
                                    : new System.Drawing.Rectangle(0, 0, originalImage.Width, originalImage.Height);

                                float srcAspect = (float)sourceRect.Width / sourceRect.Height;
                                float dstAspect = (float)job.TargetWidth / job.TargetHeight;
                                
                                System.Drawing.Rectangle destRect = srcAspect > dstAspect
                                    ? new System.Drawing.Rectangle(0, (job.TargetHeight - (int)(job.TargetWidth / srcAspect)) / 2, job.TargetWidth, (int)(job.TargetWidth / srcAspect))
                                    : new System.Drawing.Rectangle((job.TargetWidth - (int)(job.TargetHeight * srcAspect)) / 2, 0, (int)(job.TargetHeight * srcAspect), job.TargetHeight);

                                g.Clear(System.Drawing.Color.FromArgb(255, 40, 43, 48));
                                g.DrawImage(originalImage, destRect, sourceRect, System.Drawing.GraphicsUnit.Pixel);
                            }
                            
                            finalBitmap.Save(finalRenderPath, System.Drawing.Imaging.ImageFormat.Png);
                            
                            if (job.SaveTimestampedRender)
                            {
                                finalBitmap.Save(Path.Combine(job.FinalOutputPath, $"render_{job.Timestamp}.png"), System.Drawing.Imaging.ImageFormat.Png);
                            }
                        }
                        
                            try { File.Delete(job.TempRenderPath); } catch { }
                            WabiSabiLogger.Log($"Imagen procesada y guardada: {finalRenderPath}", LogLevel.Debug);
                        }
                        // Si TempRenderPath está vacío, significa que ya se procesó directamente desde GPU

                        // --- 2. PROCESAR Y GUARDAR IMAGEN DE PROFUNDIDAD ---
                        if (job.DepthData != null)
                        {
                            WabiSabiLogger.Log("Procesando datos de profundidad...", LogLevel.Debug);
                            job.EventHandler?.GenerateDepthImage(job.DepthData, job.TargetWidth, job.TargetHeight, 
                                job.FinalOutputPath, job.Timestamp, job.SaveTimestampedDepth);
                            WabiSabiLogger.Log("Imagen de profundidad guardada.", LogLevel.Debug);
                        }

                        // --- 3. GENERAR Y GUARDAR METADATOS ---
                        var targetPos = new { 
                            x = job.CamEyeX + job.CamForwardX * 10, 
                            y = job.CamEyeY + job.CamForwardY * 10, 
                            z = job.CamEyeZ + job.CamForwardZ * 10 
                        };
                        
                        var metadata = new {
                            job.Timestamp,
                            job.ViewName,
                            view_type = "3D",
                            scale = job.ViewScale,
                            detail_level = job.DetailLevel,
                            display_style = job.DisplayStyle,
                            crop_box_active = job.IsCropActive,
                            camera = new {
                                eye_position = new { x = job.CamEyeX, y = job.CamEyeY, z = job.CamEyeZ },
                                target_position = targetPos,
                                up_vector = new { x = job.CamUpX, y = job.CamUpY, z = job.CamUpZ }
                            },
                            project_info = new { name = job.ProjectName, path = job.ProjectPath },
                            gpu_acceleration = job.GpuAccelerated
                        };
                        File.WriteAllText(Path.Combine(job.FinalOutputPath, "current_metadata.json"), JsonConvert.SerializeObject(metadata, Formatting.Indented));
                        WabiSabiLogger.Log("Metadata guardada", LogLevel.Debug);

                        // --- 4. CREAR ARCHIVO DE NOTIFICACIÓN ---
                        File.WriteAllText(Path.Combine(job.FinalOutputPath, "last_update.txt"), job.Timestamp);
                        WabiSabiLogger.Log("Archivo de notificación creado", LogLevel.Debug);

                        WabiSabiLogger.Log($"Trabajo {job.Timestamp} completado.", LogLevel.Info);
                    }
                    catch (Exception ex)
                    {
                        WabiSabiLogger.LogError($"Error procesando el trabajo {job?.Timestamp}", ex);
                    }
                }
                else
                {
                    Thread.Sleep(50);
                }
            }
        }

        public string GetName() => "WabiSabi Bridge Export Event";
        

        /// <summary>
        /// Activa o desactiva dinámicamente el sistema de streaming para la exportación automática.
        /// </summary>
        public static void SetAutoExportEnabled(bool enabled, UIApplication? uiApp)
        {
            WabiSabiLogger.Log($"SetAutoExportEnabled (Arquitectura Timer) llamado con: {enabled}", LogLevel.Info);
            
            _config.AutoExport = enabled;
            _config.Save();
            _currentUiApp = uiApp;
            _isContinuousCaptureActive = enabled;

            if (enabled)
            {
                // Si el timer no existe, lo creamos
                if (_directCaptureTimer == null)
                {
                    _directCaptureTimer = new System.Windows.Forms.Timer();
                    _directCaptureTimer.Interval = 50; // Capturar ~20 veces por segundo
                    _directCaptureTimer.Tick += DirectCaptureTimer_Tick;
                }
                
                // Reiniciamos contadores y empezamos el timer
                _globalSequenceNumber = -1;
                _lastProcessedSequenceNumber = -1;
                _lastCameraMoveTimestamp = 0;
                _directCaptureTimer.Start();
                
                WabiSabiLogger.Log($"Timer de captura directa iniciado ({_directCaptureTimer.Interval}ms).", LogLevel.Info);
            }
            else
            {
                // Si el timer existe, lo detenemos
                _directCaptureTimer?.Stop();
                WabiSabiLogger.Log("Timer de captura directa detenido.", LogLevel.Info);
            }
        }

         /// <summary>
        /// Inicia el sistema de invalidación suave de la vista
        /// </summary>
        private static void StartViewInvalidationSystem()
        {
            try
            {
                // Detener cualquier timer existente
                StopViewInvalidationSystem();
                
                // Crear un timer de Windows Forms con intervalo más corto
                _viewInvalidationTimer = new System.Windows.Forms.Timer
                {
                    Interval = 16  // 16ms = ~60 FPS
                };
                
                _viewInvalidationTimer.Tick += OnInvalidationTimerTick;
                _viewInvalidationTimer.Start();
                
                WabiSabiLogger.Log($"Timer de invalidación iniciado (16ms = 60 FPS)", LogLevel.Info);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error iniciando sistema de invalidación", ex);
            }
        }

        /// <summary>
        /// Detiene el sistema de invalidación
        /// </summary>
        private static void StopViewInvalidationSystem()
        {
            try
            {
                if (_viewInvalidationTimer != null)
                {
                    _viewInvalidationTimer.Stop();
                    _viewInvalidationTimer.Tick -= OnInvalidationTimerTick;
                    _viewInvalidationTimer.Dispose();
                    _viewInvalidationTimer = null;
                    
                    WabiSabiLogger.Log("Timer de invalidación detenido", LogLevel.Info);
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error deteniendo sistema de invalidación", ex);
            }
        }

        /// <summary>
        /// Callback del timer de invalidación
        /// </summary>
        private static void OnInvalidationTimerTick(object? sender, EventArgs e)
        {
            if (_isContinuousCaptureActive && _currentUiApp != null)
            {
                try
                {
                    // Forzar invalidación más agresiva
                    _invalidationRequested = true;
                    
                    // También intentar refrescar directamente si es posible
                    if (_currentUiApp.ActiveUIDocument?.ActiveView is View3D)
                    {
                        // Este es un truco para forzar redibujado
                        _currentUiApp.ActiveUIDocument.RefreshActiveView();
                    }
                }
                catch
                {
                    // Silenciar errores
                }
            }
        }


        /// <summary>
        /// Invalida una región mínima de la vista para forzar redibujado
        /// </summary>
        private static void InvalidateMinimalViewRegion(View3D view3D)
        {
            try
            {
                // Obtener el UIView correspondiente
                var uiView = _currentUiApp?.ActiveUIDocument?.GetOpenUIViews()
                    .FirstOrDefault(v => v.ViewId == view3D.Id);
                    
                if (uiView != null)
                {
                    // TÉCNICA 1: Invalidar un píxel en la esquina
                    var viewRect = uiView.GetWindowRectangle();
                    var invalidRect = new System.Drawing.Rectangle(
                        viewRect.Left, 
                        viewRect.Top, 
                        1, 1  // Solo 1 píxel
                    );
                    
                    // Esta es una operación muy ligera que fuerza redibujado
                    _currentUiApp?.ActiveUIDocument?.RefreshActiveView();
                    
                    // ALTERNATIVA: Usar InvalidateRect de Windows API (más directo)
                    // InvalidateRect(windowHandle, ref invalidRect, false);
                }
            }
            catch
            {
                // Silenciar errores
            }
        }

       
        /// <summary>
        /// Detiene de forma segura el sistema de streaming y libera sus recursos.
        /// </summary>
        private static void ShutdownStreamingSystem()
        {
            try
            {
                WabiSabiLogger.Log("Deteniendo sistema de streaming...", LogLevel.Info);

                // Detener hilo
                _cancellationTokenSource?.Cancel();
                _streamerThread?.Join(TimeSpan.FromSeconds(1));
                _cancellationTokenSource?.Dispose();
                _cancellationTokenSource = null;
                _streamerThread = null;

                // Limpiar recursos de streaming
                _cameraAccessor?.Dispose();
                _cameraAccessor = null;
                _cameraMmf?.Dispose();
                _cameraMmf = null;
                _cameraBuffer = null; // El buffer no es IDisposable

                // Desregistrar todos los servidores
                UnregisterAllServers();

                WabiSabiLogger.Log("Sistema de streaming detenido limpiamente.", LogLevel.Info);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error durante la detención del sistema de streaming", ex);
            }
        }

        private static bool CreateRibbon(UIControlledApplication application)
        {
            try
            {
                WabiSabiLogger.Log("Creando ribbon...", LogLevel.Info);
                
                string tabName = "WabiSabi";
                
                // Crear tab
                try 
                { 
                    application.CreateRibbonTab(tabName);
                    WabiSabiLogger.Log($"Tab '{tabName}' creado", LogLevel.Info);
                } 
                catch (Autodesk.Revit.Exceptions.ArgumentException) 
                { 
                    WabiSabiLogger.Log($"Tab '{tabName}' ya existe", LogLevel.Info);
                }
                
                // Crear panel
                RibbonPanel? panel = null;
                var existingPanels = application.GetRibbonPanels(tabName);
                foreach (var p in existingPanels)
                {
                    if (p.Name == "Bridge")
                    {
                        panel = p;
                        break;
                    }
                }
                
                if (panel == null)
                {
                    panel = application.CreateRibbonPanel(tabName, "Bridge");
                    WabiSabiLogger.Log("Panel 'Bridge' creado", LogLevel.Info);
                }
                
                string thisAssemblyPath = System.Reflection.Assembly.GetExecutingAssembly().Location;
                WabiSabiLogger.Log($"Assembly path: {thisAssemblyPath}", LogLevel.Debug);
                
                // Agregar botón principal
                bool mainButtonExists = false;
                foreach (var item in panel.GetItems())
                {
                    if (item.Name == "WabiSabiBridge")
                    {
                        mainButtonExists = true;
                        break;
                    }
                }
                
                if (!mainButtonExists)
                {
                    var mainButton = new PushButtonData(
                        "WabiSabiBridge", 
                        "WabiSabi\nBridge", 
                        thisAssemblyPath, 
                        "WabiSabiBridge.WabiSabiBridgeCommand")
                    {
                        ToolTip = "Abrir WabiSabi Bridge",
                        LongDescription = "Plugin de exportación de vistas con soporte GPU"
                    };
                    
                    panel.AddItem(mainButton);
                    WabiSabiLogger.Log("Botón principal agregado", LogLevel.Info);
                }
                
                // Agregar separador
                panel.AddSeparator();
                
                // Agregar botón de diagnóstico
                bool diagButtonExists = false;
                foreach (var item in panel.GetItems())
                {
                    if (item.Name == "WabiSabiDiagnostic")
                    {
                        diagButtonExists = true;
                        break;
                    }
                }
                
                if (!diagButtonExists)
                {
                    var diagButton = new PushButtonData(
                        "WabiSabiDiagnostic", 
                        "Diagnóstico", 
                        thisAssemblyPath, 
                        "WabiSabiBridge.WabiSabiDiagnosticCommand")  // <- Nombre correcto de la clase
                    {
                        ToolTip = "Ejecutar diagnóstico del sistema",
                        LongDescription = "Verifica el estado de todos los componentes de WabiSabi Bridge"
                    };
                    
                    panel.AddItem(diagButton);
                    WabiSabiLogger.Log("Botón de diagnóstico agregado", LogLevel.Info);
                }

                return true;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en CreateRibbon", ex);
                return false;
            }
        }

        private static bool InitializeEventHandler()
        {
            try
            {
                WabiSabiLogger.Log("Inicializando event handler...", LogLevel.Info);
                
                WabiSabiEventHandler = new ExportEventHandler();
                WabiSabiEvent = ExternalEvent.Create(WabiSabiEventHandler);
                
                WabiSabiLogger.Log("Event handler inicializado correctamente", LogLevel.Info);
                return true;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error inicializando event handler", ex);
                return false;
            }
        }

        private static void InitializeStreamingSystem()
        {
            try
            {
                WabiSabiLogger.Log($"Inicializando sistema de streaming persistente...", LogLevel.Info);
                
                // --- CAMBIO: ELIMINAMOS LA COMPROBACIÓN DE _config.AutoExport ---
                // El sistema se inicializa siempre.
                
                // Si el sistema ya está funcionando, no hacemos nada.
                if (_cameraBuffer != null && _streamerThread != null && _streamerThread.IsAlive)
                {
                    WabiSabiLogger.Log("El sistema de streaming ya está activo.", LogLevel.Info);
                    return;
                }

                // Crear buffer
                const int BUFFER_CAPACITY = 128;
                _cameraBuffer = new LockFreeCameraRingBuffer(BUFFER_CAPACITY);
                WabiSabiLogger.Log($"Buffer de cámara creado con capacidad {BUFFER_CAPACITY}", LogLevel.Info);

                // Crear memory-mapped file
                if (CreateMemoryMappedFile())
                {
                    StartStreamerThread();
                }
                else
                {
                    WabiSabiLogger.Log("Memory-mapped file falló, el streaming no funcionará.", LogLevel.Warning);
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error inicializando sistema de streaming", ex);
            }
        }

        private static bool CreateMemoryMappedFile()
        {
            try
            {
                // IMPORTANTE: Usar SerializableCameraData, NO CameraData
                int cameraDataSize = System.Runtime.InteropServices.Marshal.SizeOf<SerializableCameraData>();
                int mmfSize = ((cameraDataSize + 4095) / 4096) * 4096;
                
                WabiSabiLogger.Log($"Creando MMF, tamaño: {mmfSize} bytes", LogLevel.Debug);

                try
                {
                    _cameraMmf = MemoryMappedFile.CreateNew(MmfName, mmfSize);
                    WabiSabiLogger.Log($"MMF creado: {MmfName}", LogLevel.Info);
                }
                catch (IOException)
                {
                    // Si el nombre ya existe, crear uno único
                    string uniqueMmfName = $"{MmfName}_{Guid.NewGuid():N}";
                    _cameraMmf = MemoryMappedFile.CreateNew(uniqueMmfName, mmfSize);
                    WabiSabiLogger.Log($"MMF creado con nombre único: {uniqueMmfName}", LogLevel.Info);
                }
                
                _cameraAccessor = _cameraMmf.CreateViewAccessor();
                
                // IMPORTANTE: Escribir SerializableCameraData, NO CameraData
                var defaultData = new SerializableCameraData
                {
                    EyePositionX = 0.0,
                    EyePositionY = 0.0,
                    EyePositionZ = 0.0,
                    ViewDirectionX = 0.0,
                    ViewDirectionY = 0.0,
                    ViewDirectionZ = 1.0,
                    UpDirectionX = 0.0,
                    UpDirectionY = 1.0,
                    UpDirectionZ = 0.0,
                    SequenceNumber = 0,
                    Timestamp = 0
                };
                
                _cameraAccessor.Write(0, ref defaultData);
                
                WabiSabiLogger.Log("MMF inicializado correctamente", LogLevel.Info);
                return true;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error creando MMF", ex);
                
                // Limpiar recursos en caso de error
                _cameraAccessor?.Dispose();
                _cameraAccessor = null;
                _cameraMmf?.Dispose();
                _cameraMmf = null;
                
                return false;
            }
        }

        private static void StartStreamerThread()
        {
            try
            {
                _cancellationTokenSource = new CancellationTokenSource();
                _streamerThread = new Thread(() => ProcessCameraStream(_cancellationTokenSource.Token))
                { 
                    IsBackground = true, 
                    Name = "WabiSabi_CameraStreamer" 
                };
                _streamerThread.Start();
                
                WabiSabiLogger.Log("Hilo de streaming iniciado", LogLevel.Info);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error iniciando hilo de streaming", ex);
            }
        }

        private bool SubscribeToRevitEvents(UIControlledApplication application)
        {
            try
            {
                WabiSabiLogger.Log("Suscribiendo eventos de Revit...", LogLevel.Info);
                
                application.ViewActivated += OnViewActivated;
                application.ControlledApplication.DocumentOpened += OnDocumentOpened;
                application.ControlledApplication.DocumentClosing += OnDocumentClosing;
                application.ControlledApplication.DocumentChanged += OnDocumentChanged;
                application.Idling += OnIdling;
                
                WabiSabiLogger.Log("Eventos de Revit suscritos correctamente", LogLevel.Info);
                return true;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error suscribiendo eventos", ex);
                return false;
            }
        }

        private void OnViewActivated(object? sender, ViewActivatedEventArgs e)
        {
            try
            {
                if (!_isInitialized || e.CurrentActiveView == null) return;
                
                WabiSabiLogger.LogDiagnostic("ViewActivated", $"Vista: {e.CurrentActiveView.Name}, Tipo: {e.CurrentActiveView.ViewType}");
                
                ManageServerRegistration(e.CurrentActiveView);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en OnViewActivated", ex);
            }
        }

        private void OnDocumentOpened(object? sender, DocumentOpenedEventArgs e)
        {
            try
            {
                if (!_isInitialized || e.Document == null) return;
                
                WabiSabiLogger.LogDiagnostic("DocumentOpened", $"Documento: {e.Document.Title}");
                
                var uidoc = new UIDocument(e.Document);
                if (uidoc.ActiveView != null)
                {
                    ManageServerRegistration(uidoc.ActiveView);
                }
                
                GeometryCacheManager.Instance.InvalidateCache();
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en OnDocumentOpened", ex);
            }
        }

        private void OnDocumentClosing(object? sender, DocumentClosingEventArgs e)
        {
            try
            {
                if (!_isInitialized || e.Document == null) return;
                
                WabiSabiLogger.LogDiagnostic("DocumentClosing", $"Documento: {e.Document.Title}");
                
                if (_documentServers.TryGetValue(e.Document, out var serverToClose))
                {
                    UnregisterServer(serverToClose);
                    if (_currentServer == serverToClose) 
                    {
                        _currentServer = null;
                    }
                    _documentServers.Remove(e.Document);
                }
                
                GeometryCacheManager.Instance.InvalidateCache();
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en OnDocumentClosing", ex);
            }
        }
        
        private void OnDocumentChanged(object? sender, DocumentChangedEventArgs e)
        {
            try
            {
                if (!_isInitialized) return;
                
                Document doc = e.GetDocument();
                if (doc != null)
                {
                    int addedCount = e.GetAddedElementIds().Count;
                    int deletedCount = e.GetDeletedElementIds().Count;
                    int modifiedCount = e.GetModifiedElementIds().Count;
                    
                    if (addedCount > 0 || deletedCount > 0 || modifiedCount > 0)
                    {
                        WabiSabiLogger.LogDiagnostic("DocumentChanged", 
                            $"Añadidos: {addedCount}, Eliminados: {deletedCount}, Modificados: {modifiedCount}");
                        
                        GeometryCacheManager.Instance.InvalidateCache();
                    }
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en OnDocumentChanged", ex);
            }
        }

        private static void CaptureCameraDataDirectly(UIDocument? uiDoc)
        {
            if (uiDoc?.Document == null || uiDoc.ActiveView is not View3D view3D || _cameraBuffer == null)
            {
                return;
            }

            try
            {
                var orientation = view3D.GetOrientation();

                // Solo escribimos data nueva si la cámara realmente se movió
                // La función HasOrientationChanged ya existe en la clase
                if (HasOrientationChanged(orientation))
                {
                    // Creamos una copia de la orientación para la siguiente comparación
                    _lastKnownOrientation = new ViewOrientation3D(orientation.EyePosition, orientation.UpDirection, orientation.ForwardDirection);
                    
                    // Creamos los datos de la cámara con un número de secuencia que SÍ se incrementa
                    var cameraData = new CameraData
                    {
                        EyePosition = orientation.EyePosition,
                        ViewDirection = orientation.ForwardDirection,
                        UpDirection = orientation.UpDirection,
                        // Usamos el secuenciador del hilo de streaming para mantener la consistencia
                        SequenceNumber = Interlocked.Increment(ref WabiSabiBridgeApp._globalSequenceNumber),
                        Timestamp = Stopwatch.GetTimestamp()
                    };

                    // Escribimos en el buffer. El otro hilo se encargará de pasarlo al MMF.
                    _cameraBuffer.TryWrite(cameraData);
                }
            }
            catch (Autodesk.Revit.Exceptions.InvalidOperationException)
            {
                // Ignorar errores que ocurren si la vista no está completamente lista
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogDiagnostic("CaptureCamera", $"Error en captura directa: {ex.Message}");
            }
        }

        // WabiSabiBridge.cs -> WabiSabiBridgeApp
        private void OnIdling(object? sender, IdlingEventArgs e)
        {
            // Si la captura continua no está activa, no hacemos nada en Idling.
            if (!_isInitialized || !_isContinuousCaptureActive) return;

            // El único trabajo de Idling es procesar los datos que el Timer ha generado.
            if (_cameraAccessor != null && _cameraAccessor.CanRead)
            {
                ProcessAutoExportLogicContinuous();
            }
        }


        /// <summary>
        /// Versión mejorada de la lógica de auto-export para captura continua
        /// </summary>
        private static void ProcessAutoExportLogicContinuous()
        {
            try
            {
                // 1. LEER DATOS
                _cameraAccessor!.Read(0, out SerializableCameraData serializableData);
                CameraData data = serializableData.ToCameraData();

                // Si no estamos en modo continuo o no hay datos, no hacer nada.
                if (!_isContinuousCaptureActive || data.SequenceNumber < 0) return;
                
                WabiSabiLogger.LogDiagnostic("AutoExport", 
                    $"Leyendo datos - Seq Leída: {data.SequenceNumber}, Última Procesada: {_lastProcessedSequenceNumber}");

                // 2. DETECTAR MOVIMIENTO
                // La cámara se está moviendo si el número de secuencia que leemos es MAYOR
                // que el último que procesamos en la ejecución anterior.
                bool isMoving = data.SequenceNumber > _lastProcessedSequenceNumber;

                if (isMoving)
                {
                    // Si hay movimiento, simplemente actualizamos el timestamp para saber
                    // que la cámara estuvo activa recientemente.
                    _lastCameraMoveTimestamp = Stopwatch.GetTimestamp();
                    WabiSabiLogger.Log($"Movimiento de cámara detectado! Seq: {data.SequenceNumber}", LogLevel.Info);
                }
                else if (_lastCameraMoveTimestamp > 0)
                {
                    // 3. GESTIONAR ESTABILIDAD
                    // Si no hay movimiento (isMoving = false), y tenemos un timestamp de movimiento previo,
                    // significa que el usuario acaba de detenerse. Comprobamos cuánto tiempo ha pasado.
                    long now = Stopwatch.GetTimestamp();
                    long timeSinceMove = (now - _lastCameraMoveTimestamp) / TimeSpan.TicksPerMillisecond;
                    long timeSinceExport = (now - _lastExportTimestamp) / TimeSpan.TicksPerMillisecond;

                    WabiSabiLogger.LogDiagnostic("AutoExport", 
                        $"Verificando estabilidad - Tiempo desde último mov: {timeSinceMove}ms");

                    // Si han pasado más de 500ms de estabilidad y no hemos exportado en el último segundo...
                    if (timeSinceMove > 500 && timeSinceExport > 1000)
                    {
                        WabiSabiLogger.Log($"Exportando después de {timeSinceMove}ms de estabilidad", LogLevel.Info);
                        
                        TriggerAutoExport();
                        
                        // Reiniciamos los timestamps para evitar exportaciones múltiples.
                        //_lastCameraMoveTimestamp = 0;
                        _lastExportTimestamp = now;
                    }
                }

                // 4. ACTUALIZAR ESTADO
                // Al final de todo, actualizamos el número del último paquete que hemos procesado.
                _lastProcessedSequenceNumber = data.SequenceNumber;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en ProcessAutoExportLogicContinuous", ex);
                // Reiniciar estado en caso de error
                _lastCameraMoveTimestamp = 0;
                _lastProcessedSequenceNumber = -1;
            }
        }

        private static bool HasOrientationChanged(ViewOrientation3D newOrientation)
        {
            if (_lastKnownOrientation == null) return true; // Si no teníamos una, ha cambiado.

            // Comparamos los vectores clave.
            if (!newOrientation.EyePosition.IsAlmostEqualTo(_lastKnownOrientation.EyePosition, XYZ_TOLERANCE)) return true;
            if (!newOrientation.ForwardDirection.IsAlmostEqualTo(_lastKnownOrientation.ForwardDirection, XYZ_TOLERANCE)) return true;
            if (!newOrientation.UpDirection.IsAlmostEqualTo(_lastKnownOrientation.UpDirection, XYZ_TOLERANCE)) return true;
            
            return false; // Si todos los vectores son iguales, no ha cambiado.
        }

        

        private static void TriggerAutoExport()
        {
            try
            {
                var currentConfig = WabiSabiConfig.Load();
                if (WabiSabiEventHandler != null && !string.IsNullOrWhiteSpace(currentConfig.OutputPath))
                {
                    WabiSabiLogger.Log("Configurando parámetros para auto-export", LogLevel.Debug);
                    
                    WabiSabiEventHandler.OutputPath = currentConfig.OutputPath;
                    WabiSabiEventHandler.ExportDepth = currentConfig.ExportDepth;
                    WabiSabiEventHandler.DepthResolution = currentConfig.DepthResolution;
                    WabiSabiEventHandler.DepthQuality = currentConfig.DepthQuality;
                    WabiSabiEventHandler.AutoDepthRange = currentConfig.AutoDepthRange;
                    WabiSabiEventHandler.DepthRangeDistance = currentConfig.DepthRangeDistance;
                    WabiSabiEventHandler.UseGpuAcceleration = currentConfig.UseGpuAcceleration;
                    WabiSabiEventHandler.UseGeometryExtraction = currentConfig.UseGeometryExtraction;
                    WabiSabiEventHandler.SaveTimestampedRender = currentConfig.SaveTimestampedRender;
                    WabiSabiEventHandler.SaveTimestampedDepth = currentConfig.SaveTimestampedDepth;
                    
                    WabiSabiEvent?.Raise();
                    WabiSabiLogger.Log("Auto-export disparado exitosamente", LogLevel.Info);
                }
                else
                {
                    WabiSabiLogger.LogError("No se puede disparar auto-export: Handler o OutputPath inválidos");
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en TriggerAutoExport", ex);
            }
        }

        private static void ProcessAutoExportLogic()
        {
            try
            {
                // 1. CONSUMIR: Leemos el último dato que el PRODUCTOR (servidor en tiempo real) ha generado.
                _cameraAccessor!.Read(0, out SerializableCameraData serializableData);
                CameraData data = serializableData.ToCameraData();
                
                // 2. DETECTAR MOVIMIENTO: Si el número de secuencia ha aumentado, significa que
                //    el servidor estuvo activo (el usuario estuvo moviendo la cámara).
                if (data.SequenceNumber > _lastKnownSequenceNumber)
                {
                    // Hemos detectado que hubo movimiento desde la última vez que Idling se ejecutó.
                    WabiSabiLogger.LogDiagnostic("AutoExport", 
                        $"Movimiento de cámara detectado por el servidor. Seq: {data.SequenceNumber} > {_lastKnownSequenceNumber}");
                    
                    // Actualizamos el último número de secuencia que hemos visto y la hora de este "check-in".
                    _lastKnownSequenceNumber = data.SequenceNumber;
                    _lastCameraMoveTimestamp = Stopwatch.GetTimestamp();
                }
                else
                {
                    // 3. GESTIONAR ESTABILIDAD: Si el número de secuencia NO ha aumentado,
                    //    significa que el usuario ha dejado de mover la cámara.
                    //    Ahora comprobamos si ha pasado suficiente tiempo desde el último movimiento detectado.
                    if (_lastCameraMoveTimestamp > 0)
                    {
                        long now = Stopwatch.GetTimestamp();
                        long timeSinceMove = (now - _lastCameraMoveTimestamp) / TimeSpan.TicksPerMillisecond;
                        long timeSinceExport = (now - _lastExportTimestamp) / TimeSpan.TicksPerMillisecond;
                        
                        // Si han pasado 500ms desde el último check-in donde detectamos movimiento...
                        if (timeSinceMove > 500 && timeSinceExport > 100)
                        {
                            WabiSabiLogger.Log($"Disparando auto-export después de {timeSinceMove}ms de estabilidad", LogLevel.Info);
                            
                            // Disparamos la exportación.
                            TriggerAutoExport();
                            
                            // Reiniciamos el timestamp del movimiento. El sistema ahora esperará
                            // a que `data.SequenceNumber` vuelva a ser mayor que el actual.
                            _lastCameraMoveTimestamp = 0;
                            _lastExportTimestamp = now;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en ProcessAutoExportLogic", ex);
                _lastCameraMoveTimestamp = 0;
            }
        }

        private static void ManageServerRegistration(Autodesk.Revit.DB.View view)
        {
            try
            {
                if (view?.Document == null) return;
                
                var doc = view.Document;
                
                WabiSabiLogger.LogDiagnostic("ServerRegistration", 
                    $"Gestionando servidor para doc: {doc.Title}, Vista: {view.Name}");

                // --- CAMBIO: LA LÓGICA DE REGISTRO YA NO DEPENDE DE _config.AutoExport ---
                // Si la vista es 3D, nos aseguramos de que haya un servidor registrado.
                if (view.ViewType == ViewType.ThreeD)
                {
                    if (_documentServers.TryGetValue(doc, out var existingServer))
                    {
                        _currentServer = existingServer;
                        WabiSabiLogger.LogDiagnostic("ServerRegistration", "Servidor existente encontrado y asignado como actual.");
                    }
                    else
                    {
                        // No hay servidor para este documento, lo registramos.
                        WabiSabiLogger.LogDiagnostic("ServerRegistration", "No existe servidor, procediendo a registrar uno nuevo.");
                        RegisterServerForDocument(doc);
                    }
                }
                else
                {
                    WabiSabiLogger.LogDiagnostic("ServerRegistration", 
                        $"No se registra servidor porque la vista '{view.Name}' no es 3D.");
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en ManageServerRegistration", ex);
            }
        }

       private static void RegisterServerForDocument(Document doc)
        {
            try
            {
                if (doc == null || !doc.IsValidObject)
                {
                    WabiSabiLogger.LogError("No se puede registrar servidor: documento nulo o inválido");
                    return;
                }

                if (_cameraBuffer == null)
                {
                    WabiSabiLogger.LogError("No se puede registrar servidor sin buffer de cámara");
                    return;
                }

                if (_documentServers.ContainsKey(doc))
                {
                    WabiSabiLogger.LogDiagnostic("ServerRegistration", $"Ya existe un servidor para '{doc.Title}'.");
                    _currentServer = _documentServers[doc];
                    // La lógica de activación se gestiona externamente, así que solo salimos.
                    return;
                }

                // CAMBIO CLAVE: Obtener el servicio y hacer el CAST al tipo CORRECTO.
                var multiServerService = ExternalServiceRegistry.GetService(
                    ExternalServices.BuiltInExternalServices.DirectContext3DService) as MultiServerService;
                    
                if (multiServerService == null)
                {
                    WabiSabiLogger.LogError("Servicio DirectContext3D (MultiServerService) no disponible.");
                    return;
                }

                var server = new CameraDataExtractionServer(_cameraBuffer);
                
                if (_config.AutoExport)
                {
                    server.ContinuousCapture = true;
                    _isContinuousCaptureActive = true;
                }
                
                try
                {
                    // Ahora usamos la variable `multiServerService` que SÍ tiene estos métodos.
                    multiServerService.AddServer(server);
                    
                    _documentServers[doc] = server;
                    _currentServer = server;
                    _lastKnownSequenceNumber = -1;

                    WabiSabiLogger.Log($"Servidor registrado exitosamente para documento: {doc.Title}", LogLevel.Info);
                    
                    // La llamada a ActivateServerForAllViews ahora funcionará porque le pasamos un objeto válido.
                    ActivateServerForAllViews(doc);
                    
                    if (_config.AutoExport && _viewInvalidationTimer == null)
                    {
                        StartViewInvalidationSystem();
                    }
                }
                catch (Exception ex)
                {
                    WabiSabiLogger.LogDiagnostic("ServerRegistration", $"Error registrando servidor: {ex.Message}");
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError($"Error registrando servidor para '{doc?.Title ?? "documento desconocido"}'", ex);
            }
        }

        /// <summary>
        /// NUEVO MÉTODO: Activa el servidor como servidor activo para todas las vistas 3D
        /// </summary>
        private static void ActivateServerForAllViews(Document doc)
        {
            try
            {
                if (_currentServer == null || doc == null) return;

                var multiServerService = ExternalServiceRegistry.GetService(
                    ExternalServices.BuiltInExternalServices.DirectContext3DService) as MultiServerService;
                    
                if (multiServerService == null)
                {
                    WabiSabiLogger.LogError("No se pudo obtener MultiServerService para activación");
                    return;
                }

                // Se obtiene la lista actual de servidores activos para el documento para no sobrescribir otros.
                var activeServers = multiServerService.GetActiveServerIds(doc);

                // Se añade nuestro servidor a la lista si no está ya.
                if (!activeServers.Contains(_currentServer.GetServerId()))
                {
                    activeServers.Add(_currentServer.GetServerId());
                }
                
                // CAMBIO CRÍTICO: Se llama UNA SOLA VEZ a nivel de documento y con los parámetros en el orden correcto.
                // SetActiveServers(IList<Guid> serverIds, Document document)
                multiServerService.SetActiveServers(activeServers, doc);
                
                WabiSabiLogger.Log($"Servidor {_currentServer.GetServerId()} activado para el documento {doc.Title}", LogLevel.Info);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en ActivateServerForAllViews", ex);
            }
        }
        
        // Método adicional para manejar el registro diferido
        private static void ScheduleServerRegistration(Document doc)
        {
            try
            {
                if (doc == null || _documentServers.ContainsKey(doc))
                {
                    return;
                }

                // Marcar el documento para registro diferido
                WabiSabiLogger.LogDiagnostic("ServerRegistration", 
                    $"Programando registro diferido para '{doc.Title}'");
                
                // El registro real se hará en el siguiente evento Idling cuando tengamos un contexto válido
                _pendingRegistrations.Add(doc);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error programando registro diferido", ex);
            }
        }        
  
        
        private static void UnregisterAllServers()
        {
            try
            {
                foreach (var server in _documentServers.Values) 
                {
                    UnregisterServer(server);
                }
                _documentServers.Clear();
                _currentServer = null;
                
                WabiSabiLogger.Log("Todos los servidores desregistrados", LogLevel.Info);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error desregistrando todos los servidores", ex);
            }
        }

        private static void UnregisterServer(CameraDataExtractionServer server)
        {
            try
            {
                // CAMBIO CLAVE: Obtener el servicio y hacer el CAST al tipo CORRECTO.
                var multiServerService = ExternalServiceRegistry.GetService(
                    ExternalServices.BuiltInExternalServices.DirectContext3DService) as MultiServerService;

                // Ahora el objeto no será nulo y SÍ tiene el método RemoveServer.
                multiServerService?.RemoveServer(server.GetServerId());
                
                WabiSabiLogger.Log($"Servidor desregistrado: {server.GetServerId()}", LogLevel.Info);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error desregistrando servidor", ex);
            }
        }

        public Result OnShutdown(UIControlledApplication application)
        {
            try
            {
                WabiSabiLogger.Log("WabiSabi Bridge - CERRANDO", LogLevel.Info);
                // DETENER EL HILO CONSUMIDOR
                _exportCts?.Cancel();
                _exportWorker?.Wait(TimeSpan.FromSeconds(2)); // Esperar a que termine
                _exportCts?.Dispose();
                WabiSabiLogger.Log("Hilo de trabajo de exportación detenido.", LogLevel.Info);
                // Detener nuestro nuevo timer
                if (_directCaptureTimer != null)
                {
                    _directCaptureTimer.Stop();
                    _directCaptureTimer.Dispose();
                    _directCaptureTimer = null;
                }
                
                // ... (el resto del código de OnShutdown se queda igual)
                StopViewInvalidationSystem();
                ShutdownStreamingSystem();
                CleanupNonStreamingResources();
                
                WabiSabiLogger.Log("WabiSabi Bridge cerrado correctamente", LogLevel.Info);
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error en OnShutdown", ex);
                return Result.Failed;
            }
        }

        private void CleanupNonStreamingResources()
        {
            try
            {
                // Los recursos de streaming ya no se limpian aquí.
                // _cameraAccessor, _cameraMmf, _cancellationTokenSource ya son manejados.
                // UnregisterAllServers también es manejado.
                
                GeometryCacheManager.Instance.Dispose();
                
                WabiSabiEventHandler?.Dispose();
                WabiSabiEventHandler = null;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error limpiando recursos no-streaming", ex);
            }
        }

        private static void ProcessCameraStream(CancellationToken token)
        {
            WabiSabiLogger.Log("Hilo de streaming de cámara iniciado", LogLevel.Info);
            int processedFrames = 0;
            
            while (!token.IsCancellationRequested)
            {
                try
                {
                    bool hasData = false;
                    
                    if (_cameraBuffer != null && _cameraAccessor != null && _cameraAccessor.CanWrite)
                    {
                        if (_cameraBuffer.TryRead(out CameraData data))
                        {
                            // IMPORTANTE: Verificar el tamaño antes de escribir
                            int requiredSize = System.Runtime.InteropServices.Marshal.SizeOf<SerializableCameraData>();
                            if (_cameraAccessor.Capacity >= requiredSize)
                            {
                                // Convertir CameraData a SerializableCameraData
                                var serializableData = SerializableCameraData.FromCameraData(data);
                                
                                // Escribir la estructura serializable
                                _cameraAccessor.Write(0, ref serializableData);
                                processedFrames++;
                                hasData = true;
                                
                                if (processedFrames % 100 == 0)
                                {
                                    WabiSabiLogger.LogDiagnostic("CameraStream", 
                                        $"{processedFrames} frames procesados");
                                }
                            }
                            else
                            {
                                WabiSabiLogger.LogError($"MMF muy pequeño. Requerido: {requiredSize}, Disponible: {_cameraAccessor.Capacity}");
                            }
                        }
                    }
                    
                    if (!hasData)
                    {
                        Thread.Sleep(1);
                    }
                }
                catch (Exception ex)
                {
                    WabiSabiLogger.LogError("Error en hilo de streaming", ex);
                    Thread.Sleep(100);
                }
            }
            
            WabiSabiLogger.Log($"Hilo de streaming terminado. Total frames procesados: {processedFrames}", LogLevel.Info);
        }

        internal static string RunDiagnostics()
        {
            var diag = new System.Text.StringBuilder();
            
            diag.AppendLine($"Estado del sistema WabiSabi:");
            diag.AppendLine($"- Inicializado: {_isInitialized}");
            diag.AppendLine($"- EventHandler: {(WabiSabiEventHandler != null ? "OK" : "NULL")}");
            diag.AppendLine($"- Event: {(WabiSabiEvent != null ? "OK" : "NULL")}");
            diag.AppendLine($"- CameraBuffer: {(_cameraBuffer != null ? "OK" : "NULL")}");
            diag.AppendLine($"- MMF: {(_cameraMmf != null ? "OK" : "NULL")}");
            diag.AppendLine($"- Accessor: {(_cameraAccessor != null ? "OK" : "NULL")}");
            diag.AppendLine($"- CurrentServer: {(_currentServer != null ? "OK" : "NULL")}");
            
            // Agregar estadísticas del servidor si está disponible
            if (_currentServer != null)
            {
                var stats = _currentServer.GetServerStats();
                diag.AppendLine($"- Server Stats: {stats}");
            }
            
            diag.AppendLine($"- StreamerThread: {(_streamerThread != null && _streamerThread.IsAlive ? "VIVO" : "INACTIVO")}");
            diag.AppendLine($"- Servidores registrados: {_documentServers.Count}");
            diag.AppendLine($"- Config.AutoExport: {_config.AutoExport}");
            diag.AppendLine($"- Config.OutputPath: {_config.OutputPath}");
            diag.AppendLine($"- Idling calls: {_idlingCallCount}");
            diag.AppendLine($"- Last sequence: {_lastKnownSequenceNumber}");
            diag.AppendLine($"- Captura continua activa: {_isContinuousCaptureActive}");
            diag.AppendLine($"- Timer invalidación: {(_viewInvalidationTimer != null && _viewInvalidationTimer.Enabled ? "ACTIVO" : "INACTIVO")}");
            
            string result = diag.ToString();
            WabiSabiLogger.Log("=== DIAGNÓSTICO ===\n" + result, LogLevel.Info);
            
            return result;
        }
        
    }
    
    #endregion

    #region Clase de Configuración (Corregida)
    
    public class WabiSabiConfig
    {
        private string _outputPath = string.Empty;
        
        // Propiedades con valores por defecto seguros
        public string OutputPath 
        { 
            get => string.IsNullOrWhiteSpace(_outputPath) ? GetDefaultOutputPath() : _outputPath;
            set => SetOutputPath(value);
        }
        
        public bool AutoExport { get; set; } = false; // Deshabilitado por defecto para evitar problemas
        public bool ExportDepth { get; set; } = true;
        public int DepthResolution { get; set; } = 512;
        public int DepthQuality { get; set; } = 1;
        public bool AutoDepthRange { get; set; } = true;
        public double DepthRangeDistance { get; set; } = 50.0;
        public bool UseGpuAcceleration { get; set; } = true;
        public bool UseGeometryExtraction { get; set; } = false;
        public bool SaveTimestampedRender { get; set; } = true;
        public bool SaveTimestampedDepth { get; set; } = true;

        // Constructor por defecto
        public WabiSabiConfig()
        {
            _outputPath = GetDefaultOutputPath();
        }

        private static string GetDefaultOutputPath()
        {
            try
            {
                string documentsPath = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                if (string.IsNullOrWhiteSpace(documentsPath))
                {
                    // Fallback si Documents no está disponible
                    documentsPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "Documents");
                }
                
                string defaultPath = Path.Combine(documentsPath, "WabiSabiBridge");
                
                // Validar que la ruta sea válida
                if (IsValidPath(defaultPath))
                {
                    return defaultPath;
                }
                else
                {
                    // Último recurso: usar directorio temporal
                    return Path.Combine(Path.GetTempPath(), "WabiSabiBridge");
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error obteniendo ruta por defecto: {ex.Message}");
                return Path.Combine(Path.GetTempPath(), "WabiSabiBridge");
            }
        }

        private void SetOutputPath(string value)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(value))
                {
                    _outputPath = GetDefaultOutputPath();
                    return;
                }

                // Limpiar y validar la ruta
                string cleanPath = value.Trim();
                
                // Expandir la ruta completa
                string fullPath = Path.GetFullPath(cleanPath);
                
                // Validar que sea una ruta válida
                if (IsValidPath(fullPath))
                {
                    _outputPath = fullPath;
                }
                else
                {
                    Debug.WriteLine($"[WabiSabiConfig] Ruta inválida: {fullPath}, usando por defecto.");
                    _outputPath = GetDefaultOutputPath();
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error estableciendo OutputPath: {ex.Message}");
                _outputPath = GetDefaultOutputPath();
            }
        }

        private static bool IsValidPath(string path)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(path)) return false;
                
                // Verificar caracteres inválidos
                char[] invalidChars = Path.GetInvalidPathChars();
                if (path.IndexOfAny(invalidChars) >= 0) return false;
                
                // Intentar crear la ruta para validar
                string? directoryName = Path.GetDirectoryName(path);
                if (directoryName == null) return false;
                
                // Verificar que no sea demasiado larga (Windows tiene límite de ~260 caracteres)
                if (path.Length > 250) return false;
                
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static string ConfigPath
        {
            get
            {
                try
                {
                    string appDataPath = Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData);
                    if (string.IsNullOrWhiteSpace(appDataPath))
                    {
                        // Fallback
                        appDataPath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "AppData", "Roaming");
                    }
                    
                    return Path.Combine(appDataPath, "WabiSabiBridge", "config.json");
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"[WabiSabiConfig] Error obteniendo ruta de configuración: {ex.Message}");
                    return Path.Combine(Path.GetTempPath(), "WabiSabiBridge_config.json");
                }
            }
        }
        
        public static WabiSabiConfig Load()
        {
            try
            {
                string configPath = ConfigPath;
                
                if (File.Exists(configPath))
                {
                    Debug.WriteLine($"[WabiSabiConfig] Cargando configuración desde: {configPath}");
                    
                    string jsonContent = File.ReadAllText(configPath);
                    if (!string.IsNullOrWhiteSpace(jsonContent))
                    {
                        var config = JsonConvert.DeserializeObject<WabiSabiConfig>(jsonContent);
                        if (config != null)
                        {
                            // Validar y corregir configuración cargada
                            config.ValidateAndCorrectConfig();
                            Debug.WriteLine("[WabiSabiConfig] Configuración cargada exitosamente.");
                            return config;
                        }
                    }
                }
                else
                {
                    Debug.WriteLine($"[WabiSabiConfig] Archivo de configuración no existe: {configPath}");
                }
            }
            catch (JsonException ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error de JSON al cargar configuración: {ex.Message}");
            }
            catch (IOException ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error de E/S al cargar configuración: {ex.Message}");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error general cargando configuración: {ex.Message}");
            }
            
            Debug.WriteLine("[WabiSabiConfig] Creando configuración por defecto.");
            var defaultConfig = new WabiSabiConfig();
            defaultConfig.ValidateAndCorrectConfig();
            return defaultConfig;
        }

        private void ValidateAndCorrectConfig()
        {
            try
            {
                // Corregir valores fuera de rango
                if (DepthResolution <= 0) DepthResolution = 512;
                if (DepthResolution > 4096) DepthResolution = 4096;
                
                // Asegurar que sea potencia de 2 cercana
                DepthResolution = GetNearestPowerOfTwo(DepthResolution);
                
                if (DepthQuality < 0) DepthQuality = 0;
                // --- CORRECCIÓN ---
                // El valor máximo para el índice de calidad ahora es 2.
                if (DepthQuality > 2) DepthQuality = 2;
                
                if (DepthRangeDistance <= 0) DepthRangeDistance = 50.0;
                if (DepthRangeDistance > 1000) DepthRangeDistance = 1000.0;
                
                // Validar ruta de salida
                if (string.IsNullOrWhiteSpace(_outputPath) || !IsValidPath(_outputPath))
                {
                    _outputPath = GetDefaultOutputPath();
                }

                Debug.WriteLine("[WabiSabiConfig] Configuración validada y corregida.");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error validando configuración: {ex.Message}");
            }
        }

        private static int GetNearestPowerOfTwo(int value)
        {
            if (value <= 256) return 256;
            if (value <= 512) return 512;
            if (value <= 1024) return 1024;
            if (value <= 2048) return 2048;
            return 4096;
        }
        
        public void Save()
        {
            try
            {
                // Validar antes de guardar
                ValidateAndCorrectConfig();
                
                string configPath = ConfigPath;
                string? configDir = Path.GetDirectoryName(configPath);
                
                // Crear directorio si no existe
                if (!string.IsNullOrEmpty(configDir) && !Directory.Exists(configDir))
                {
                    Directory.CreateDirectory(configDir);
                    Debug.WriteLine($"[WabiSabiConfig] Directorio de configuración creado: {configDir}");
                }
                
                // Serializar con indentación para legibilidad
                string jsonContent = JsonConvert.SerializeObject(this, Formatting.Indented);
                
                // Escribir archivo con manejo de errores
                File.WriteAllText(configPath, jsonContent);
                
                Debug.WriteLine($"[WabiSabiConfig] Configuración guardada en: {configPath}");
            }
            catch (JsonException ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error de JSON al guardar configuración: {ex.Message}");
            }
            catch (IOException ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error de E/S al guardar configuración: {ex.Message}");
            }
            catch (UnauthorizedAccessException ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error de permisos al guardar configuración: {ex.Message}");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WabiSabiConfig] Error general guardando configuración: {ex.Message}");
            }
        }

        /// <summary>
        /// Crea una configuración de depuración/prueba
        /// </summary>
        public static WabiSabiConfig CreateDebugConfig()
        {
            var config = new WabiSabiConfig
            {
                AutoExport = false, // Deshabilitado para debugging
                ExportDepth = true,
                DepthResolution = 512,
                DepthQuality = 1,
                UseGpuAcceleration = true,
                UseGeometryExtraction = false
            };
            
            Debug.WriteLine("[WabiSabiConfig] Configuración de debug creada.");
            return config;
        }

        public override string ToString()
        {
            return $"WabiSabiConfig: AutoExport={AutoExport}, DepthRes={DepthResolution}, GPU={UseGpuAcceleration}, Path='{OutputPath}'";
        }
        

    

    #endregion
    }

    /// <summary>
    /// Contiene toda la información necesaria para procesar una exportación en un hilo secundario.
    /// </summary>
    public class ExportJob
    {
        // Datos de la imagen
        public string TempRenderPath { get; set; } = string.Empty;
        public int TargetWidth { get; set; }
        public int TargetHeight { get; set; }

        // Datos de recorte
        public bool IsCropActive { get; set; }
        public double OutlineMinU { get; set; }
        public double OutlineMinV { get; set; }
        public double OutlineMaxU { get; set; }
        public double OutlineMaxV { get; set; }
        
        // Datos de guardado
        public string FinalOutputPath { get; set; } = string.Empty;
        public string Timestamp { get; set; } = string.Empty;
        public bool SaveTimestampedRender { get; set; }
        public bool SaveTimestampedDepth { get; set; }

        // Datos para Metadata
        public string ViewName { get; set; } = string.Empty;
        public int ViewScale { get; set; }
        public string DetailLevel { get; set; } = string.Empty;
        public string DisplayStyle { get; set; } = string.Empty;
        public double CamEyeX { get; set; }
        public double CamEyeY { get; set; }
        public double CamEyeZ { get; set; }
        public double CamForwardX { get; set; }
        public double CamForwardY { get; set; }
        public double CamForwardZ { get; set; }
        public double CamUpX { get; set; }
        public double CamUpY { get; set; }
        public double CamUpZ { get; set; }
        public string ProjectName { get; set; } = string.Empty;
        public string ProjectPath { get; set; } = string.Empty;
        public bool GpuAccelerated { get; set; }
        public double[,]? DepthData { get; set; }
        public ExportEventHandler? EventHandler { get; set; }
    }
}