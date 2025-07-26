// EN EL NUEVO ARCHIVO: OptimizedAutoExportManager.cs

using Autodesk.Revit.DB;
using ComputeSharp;
using System;
using System.Collections.Concurrent; // Necesario para BlockingCollection
using System.Drawing; // Necesario para Bitmap
using System.Drawing.Imaging; // Necesario para PixelFormat e ImageFormat
using System.IO;
using System.Linq; // Necesario para .First() en guardado de PNG
using System.Threading;
using System.Threading.Tasks;
using WabiSabiBridge.Extractors.Cache; // Asegúrate de que este namespace es correcto
using WabiSabiBridge.Extractors.Gpu;   // Asegúrate de que este namespace es correcto
using WabiSabiBridge.DirectContext;
using Autodesk.Revit.UI;

namespace WabiSabiBridge
{
    /// <summary>
    /// Gestor central para el sistema de auto-exportación optimizado.
    /// Orquesta el monitor de cámara, el caché de geometría y el renderizado en segundo plano.
    /// </summary>
    public class OptimizedAutoExportManager
    {
        // Componentes principales
        private readonly GeometryCacheManager _cacheManager;
        private readonly JournalMonitor _journalMonitor;
        private readonly LockFreeCameraRingBuffer _cameraBuffer;
        
        private CachedGeometryData _currentCache;
        private double _viewAspectRatio;
        private double _horizontalFieldOfView;

        // Control del bucle consumidor
        private CancellationTokenSource? _cts;
        private Task? _consumerTask;

        public bool IsRunning { get; private set; }

        public OptimizedAutoExportManager(string cacheDirectory)
        {
            _cacheManager = GeometryCacheManager.Instance;
            _cameraBuffer = new LockFreeCameraRingBuffer(256);
            _journalMonitor = new JournalMonitor(_cameraBuffer);
            
        }

        public async Task StartAutoExport(UIApplication uiApp, Document doc, View3D view)
        {
            if (IsRunning) return;

            WabiSabiLogger.Log("Iniciando sistema de auto-exportación optimizado...", LogLevel.Info);

            // --- INICIO DE CIRUGÍA 1: OBTENER UIVIEW Y CALCULAR FOV ---
            // Obtenemos el UIDocument activo desde la aplicación
            var uiDoc = uiApp.ActiveUIDocument;
            if (uiDoc == null)
            {
                WabiSabiLogger.LogError("Error fatal: No se pudo obtener el UIDocument activo.");
                return;
            }
            
            // Buscamos la UIView correspondiente a la vista 3D que nos pasaron
            var uiView = uiDoc.GetOpenUIViews().FirstOrDefault(v => v.ViewId == view.Id);
            if (uiView == null)
            {
                WabiSabiLogger.LogError($"Error fatal: No se pudo encontrar una UIView abierta para la vista '{view.Name}'.");
                return;
            }

            // Calcular y almacenar el aspect ratio real de la vista
            var viewRect = uiView.GetWindowRectangle();
            _viewAspectRatio = (double)(viewRect.Right - viewRect.Left) / (viewRect.Bottom - viewRect.Top);

            // Calcular y almacenar el campo de visión horizontal (en radianes)
            // La API no lo provee directamente, así que lo calculamos.
            try
            {
                IList<XYZ> corners = uiView.GetZoomCorners();
                XYZ eyePos = view.GetOrientation().EyePosition;
                
                // Calcular el centro del plano de la vista
                XYZ viewCenter = (corners[0] + corners[1]) / 2.0;

                // Calcular la distancia perpendicular desde el ojo al plano de la vista
                double distanceToPlane = (viewCenter - eyePos).DotProduct(view.GetOrientation().ForwardDirection);

                // Calcular el ancho real del plano de la vista en el espacio del modelo
                double viewWidth = corners[0].DistanceTo(corners[1]); // Distancia diagonal
                // Corregir por el aspect ratio para obtener el ancho horizontal proyectado
                viewWidth = viewWidth / Math.Sqrt(1 + (1 / (_viewAspectRatio * _viewAspectRatio)));
                
                // Usar la tangente para calcular el FOV
                _horizontalFieldOfView = 2 * Math.Atan((viewWidth / 2.0) / distanceToPlane);
                
                WabiSabiLogger.Log($"FOV calculado: {_horizontalFieldOfView * (180.0 / Math.PI):F2} grados", LogLevel.Debug);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.Log("No se pudo calcular el FOV, usando valor por defecto. Error: " + ex.Message, LogLevel.Warning);
                // Usar un valor por defecto razonable si el cálculo falla
                _horizontalFieldOfView = 60.0 * (Math.PI / 180.0); // 60 grados en radianes
            }
            // --- FIN DE CIRUGÍA 1 ---

            IsRunning = true;
            WabiSabiLogger.Log("Verificando caché de geometría...", LogLevel.Info);
            _currentCache = await _cacheManager.EnsureCacheIsValidAsync(doc, view);
            WabiSabiLogger.Log("Caché de geometría listo.", LogLevel.Info);

            _journalMonitor.Start();

            _cts = new CancellationTokenSource();
            _consumerTask = Task.Run(() => ConsumerLoop(_cts.Token));

            WabiSabiLogger.Log("Sistema de auto-exportación activo y corriendo.", LogLevel.Info);
        }

        public void StopAutoExport()
        {
            if (!IsRunning) return;

            WabiSabiLogger.Log("Deteniendo sistema de auto-exportación...", LogLevel.Info);
            _journalMonitor.Stop();
            _cts?.Cancel();
            try { _consumerTask?.Wait(TimeSpan.FromSeconds(1)); } catch { /* Ignorar excepciones de cancelación */ }
            _cts?.Dispose();
            IsRunning = false;
            WabiSabiLogger.Log("Sistema de auto-exportación detenido.", LogLevel.Info);
        }

        public async Task OnModelChanged(Document doc, View3D view)
        {
            if (!IsRunning) return;
            WabiSabiLogger.Log("Modelo modificado, reconstruyendo caché en segundo plano...", LogLevel.Info);
            _currentCache = await _cacheManager.EnsureCacheIsValidAsync(doc, view);
            WabiSabiLogger.Log("Caché reconstruido y actualizado.", LogLevel.Info);
        }

        private async Task ConsumerLoop(CancellationToken token)
        {
            const int TARGET_FPS = 12;
            var frameTime = TimeSpan.FromMilliseconds(1000.0 / TARGET_FPS);
            var lastExportTime = DateTime.MinValue;
            long lastProcessedSeq = -1;

            while (!token.IsCancellationRequested)
            {
                CameraData? latestData = null;
                while (_cameraBuffer.TryRead(out CameraData data))
                {
                    latestData = data;
                }

                if (latestData.HasValue && DateTime.Now - lastExportTime >= frameTime)
                {
                    if (latestData.Value.SequenceNumber > lastProcessedSeq)
                    {
                        lastExportTime = DateTime.Now;
                        lastProcessedSeq = latestData.Value.SequenceNumber;
                        await ExportFrameAsync(latestData.Value);
                    }
                }

                await Task.Delay(10, token);
            }
        }

        private async Task ExportFrameAsync(CameraData cameraData)
        {
            try
            {
                var wabiSabiConfig = WabiSabiConfig.Load(); // Renombrado para evitar confusión
                int resolution = wabiSabiConfig.DepthResolution;

                // --- INICIO DE CIRUGÍA: RECONSTRUCCIÓN COMPLETA DE CÁMARA POR FRAME ---

                // 1. OBTENER DATOS DE CÁMARA EN TIEMPO REAL (del Journal)
                //    Estos tres vectores definen la cámara completa en este instante.
                var eyePos = cameraData.EyePosition;
                var viewDir = cameraData.ViewDirection.Normalize();
                var upDir = cameraData.UpDirection.Normalize();
                var rightDir = viewDir.CrossProduct(upDir).Normalize();

                // 2. OBTENER LA "FORMA" DE LA VISTA (de la configuración inicial)
                //    Calculamos qué tan ancha y alta debe ser la ventana de la cámara.
                double halfFov = _horizontalFieldOfView / 2.0;
                double viewPlaneHalfWidth = Math.Tan(halfFov);
                double viewPlaneHalfHeight = viewPlaneHalfWidth / _viewAspectRatio;

                // 3. RECONSTRUIR EL PLANO DE LA VISTA PARA ESTE FRAME
                //    Colocamos un plano de vista virtual a una unidad de distancia del ojo.
                var viewCenter = eyePos.Add(viewDir);
                //    Y calculamos sus vectores basándonos en la orientación actual.
                var bottomLeft = viewCenter.Subtract(rightDir.Multiply(viewPlaneHalfWidth)).Subtract(upDir.Multiply(viewPlaneHalfHeight));
                var viewWidthVector = rightDir.Multiply(2 * viewPlaneHalfWidth);
                var viewHeightVector = upDir.Multiply(2 * viewPlaneHalfHeight);

                // 4. CONSTRUIR LA CONFIGURACIÓN DEL SHADER CON LOS VECTORES DE ESTE FRAME
                //    Esta lógica ahora es idéntica a la de la exportación manual, garantizando consistencia.
                var rayTracingConfig = new RayTracingConfig
                {
                    EyePosition = new Float3((float)eyePos.X, (float)eyePos.Y, (float)eyePos.Z),
                    // Vector desde el ojo a la esquina inferior izquierda (recién calculada)
                    ViewDirection = new Float3((float)(bottomLeft.X - eyePos.X), (float)(bottomLeft.Y - eyePos.Y), (float)(bottomLeft.Z - eyePos.Z)),
                    // Vector de ancho completo (recién calculado)
                    RightDirection = new Float3((float)viewWidthVector.X, (float)viewWidthVector.Y, (float)viewWidthVector.Z),
                    // Vector de altura completo (recién calculado)
                    UpDirection = new Float3((float)viewHeightVector.X, (float)viewHeightVector.Y, (float)viewHeightVector.Z),
                    Width = resolution,
                    Height = (int)Math.Round(resolution / _viewAspectRatio),
                    MinDepth = 0.1f,
                    MaxDepth = (float)wabiSabiConfig.DepthRangeDistance
                };

                // --- FIN DE CIRUGÍA ---

                // El resto del código que llama al _gpuManager y guarda los resultados permanece igual...
                if (!_currentCache.IsValid)
                {
                    WabiSabiLogger.Log("Esperando a que el caché sea válido para renderizar...", LogLevel.Warning);
                    return;
                }

                // --- INICIO DE CIRUGÍA 3: USAR EL NOMBRE DEL MMF ---

                var gpuManager = WabiSabiBridgeApp.GpuManager;

                if (gpuManager == null || !gpuManager.IsGpuAvailable)
                {
                    WabiSabiLogger.Log("Auto-Export: Gestor de GPU no disponible, omitiendo frame.", LogLevel.Warning);
                    return; 
                }

                // Usamos el nombre del caché (_currentCache.MmfName) en lugar del objeto MMF.
                float[] depthBuffer = await gpuManager.ExecuteDepthRayTracingFromCacheAsync(
                    _currentCache.MmfName, _currentCache.VertexCount, _currentCache.TriangleCount, rayTracingConfig);

                float[] lineBuffer = await gpuManager.ExecuteLineRenderAsync(
                    _currentCache.MmfName, _currentCache.VertexCount, _currentCache.TriangleCount, rayTracingConfig);

                // --- FIN DE CIRUGÍA 3 ---

                _ = SaveResultsAsync(depthBuffer, lineBuffer, rayTracingConfig.Width, rayTracingConfig.Height);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error exportando frame", ex);
            }
        }

        private async Task SaveResultsAsync(float[] depthBuffer, float[] lineBuffer, int width, int height)
        {
            var config = WabiSabiConfig.Load();
            string outputPath = config.OutputPath;
            if (!Directory.Exists(outputPath)) Directory.CreateDirectory(outputPath);

            var saveRenderTask = Task.Run(() =>
            {
                string renderPath = Path.Combine(outputPath, "current_render.png");
                GenerateImageFromGpuBuffer(lineBuffer, width, height, renderPath);
            });

            var saveDepthTask = Task.Run(() =>
            {
                string depthPath = Path.Combine(outputPath, "current_depth.png");
                GenerateDepthImageFromGpuBuffer(depthBuffer, width, height, depthPath);
            });

            await Task.WhenAll(saveRenderTask, saveDepthTask);
            await File.WriteAllTextAsync(Path.Combine(outputPath, "last_update.txt"), DateTime.Now.ToString("o"));
        }

        private void GenerateImageFromGpuBuffer(float[] buffer, int width, int height, string filePath)
        {
            if (buffer == null || buffer.Length != width * height * 4) return;
            using var finalBitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            BitmapData bmpData = finalBitmap.LockBits(new System.Drawing.Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, finalBitmap.PixelFormat);
            try
            {
                int pixelCount = width * height;
                byte[] byteBuffer = new byte[pixelCount * 4];
                Parallel.For(0, pixelCount, i =>
                {
                    int srcIdx = i * 4;
                    int dstIdx = i * 4;
                    byteBuffer[dstIdx] = (byte)(Math.Max(0, Math.Min(255, buffer[srcIdx + 3] * 255)));
                    byteBuffer[dstIdx + 1] = (byte)(Math.Max(0, Math.Min(255, buffer[srcIdx] * 255)));
                    byteBuffer[dstIdx + 2] = (byte)(Math.Max(0, Math.Min(255, buffer[srcIdx + 1] * 255)));
                    byteBuffer[dstIdx + 3] = (byte)(Math.Max(0, Math.Min(255, buffer[srcIdx + 2] * 255)));
                });
                System.Runtime.InteropServices.Marshal.Copy(byteBuffer, 0, bmpData.Scan0, byteBuffer.Length);
            }
            finally { finalBitmap.UnlockBits(bmpData); }
            finalBitmap.Save(filePath, System.Drawing.Imaging.ImageFormat.Png);
        }

        private void GenerateDepthImageFromGpuBuffer(float[] buffer, int width, int height, string filePath)
        {
            using var depthMap = new Bitmap(width, height, PixelFormat.Format24bppRgb);
            BitmapData bmpData = depthMap.LockBits(new System.Drawing.Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, depthMap.PixelFormat);
            unsafe
            {
                byte* ptr = (byte*)bmpData.Scan0;
                Parallel.For(0, height, y =>
                {
                    byte* row = ptr + (y * bmpData.Stride);
                    for (int x = 0; x < width; x++)
                    {
                        byte depthValue = (byte)(Math.Max(0.0, Math.Min(1.0, buffer[y * width + x])) * 255);
                        row[x * 3] = depthValue;
                        row[x * 3 + 1] = depthValue;
                        row[x * 3 + 2] = depthValue;
                    }
                });
            }
            depthMap.UnlockBits(bmpData);
            depthMap.Save(filePath, System.Drawing.Imaging.ImageFormat.Png);
        }

        public AutoExportStatistics GetStatistics()
        {
            var cacheStats = _cacheManager.GetStatistics();
            return new AutoExportStatistics
            {
                IsRunning = this.IsRunning,
                CacheStatistics = cacheStats,
                CurrentMode = "Journal/GPU",
                TargetFPS = 12
            };
        }
    }

    public class AutoExportStatistics
    {
        public bool IsRunning { get; set; }
        public string CurrentMode { get; set; } = "N/A";
        public int TargetFPS { get; set; }
        public CacheStatistics? CacheStatistics { get; set; }
    }
}
