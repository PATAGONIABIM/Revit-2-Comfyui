// DepthExtractorFast.cs - Extractor de profundidad ultra optimizado con GPU y submuestreo
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Threading.Tasks;
using System.Buffers;
using System.Runtime.InteropServices;
using System.IO.MemoryMappedFiles;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Drawing = System.Drawing;
using WabiSabiBridge.Extractors.Gpu;

namespace WabiSabiBridge.Extractors
{
    public class DepthExtractorFast : IDisposable
    {
        private readonly UIApplication _uiApp;
        private readonly int _resolution;
        private readonly int _subsampleFactor;
        private int _debugRayCount = 0;
        
        // CORREGIDO: Declarados como nullables ya que pueden no inicializarse si la GPU falla.
        private IGpuAccelerationManager? _gpuManager;
        private readonly ArrayPool<float> _floatPool = ArrayPool<float>.Shared;
        private readonly ArrayPool<double> _doublePool = ArrayPool<double>.Shared;
        private MemoryMappedFile? _depthCacheMmf;
        private MemoryMappedViewAccessor? _depthCacheAccessor;
        
        
        public bool AutoDepthRange { get; set; } = true;
        public double ManualDepthDistance { get; set; } = 50.0;
        public bool UseGpuAcceleration { get; set; } = true;
        public bool UseAdaptiveSampling { get; set; } = true;
        
        // Caché mejorado con memory-mapped file para grandes datasets
        private class AdvancedDepthCache
        {
            private readonly Dictionary<long, float> _l1Cache = new Dictionary<long, float>();
            private readonly MemoryMappedViewAccessor? _l2Cache; // CORREGIDO: Acepta nulos
            private readonly int _cacheRadius;
            private readonly int _width;
            private readonly int _height;
            private const int L1_MAX_SIZE = 10000;
            
            // CORREGIDO: El constructor acepta un accessor nulo.
            public AdvancedDepthCache(int width, int height, MemoryMappedViewAccessor? l2Cache, int radius = 3)
            {
                _width = width;
                _height = height;
                _l2Cache = l2Cache;
                _cacheRadius = radius;
            }
            
            public bool TryGetValue(int x, int y, out float depth)
            {
                long key = GetKey(x, y);
                
                // Buscar en L1 (memoria)
                if (_l1Cache.TryGetValue(key, out depth))
                {
                    return true;
                }
                
                // Buscar en L2 (memory-mapped file)
                if (_l2Cache != null)
                {
                    try
                    {
                        depth = _l2Cache.ReadSingle(key * sizeof(float));
                        if (depth >= 0)
                        {
                            // Promover a L1
                            if (_l1Cache.Count < L1_MAX_SIZE)
                            {
                                _l1Cache[key] = depth;
                            }
                            return true;
                        }
                    }
                    catch { }
                }
                
                // Buscar en píxeles cercanos (interpolación inteligente)
                return TryInterpolateFromNearby(x, y, out depth);
            }
            
            private bool TryInterpolateFromNearby(int x, int y, out float depth)
            {
                var nearbyValues = new List<(float dist, float value)>();
                
                for (int dx = -_cacheRadius; dx <= _cacheRadius; dx++)
                {
                    for (int dy = -_cacheRadius; dy <= _cacheRadius; dy++)
                    {
                        if (dx == 0 && dy == 0) continue;
                        
                        int nx = x + dx;
                        int ny = y + dy;
                        
                        if (nx >= 0 && nx < _width && ny >= 0 && ny < _height)
                        {
                            long nkey = GetKey(nx, ny);
                            if (_l1Cache.TryGetValue(nkey, out float nvalue))
                            {
                                float distance = (float)Math.Sqrt(dx * dx + dy * dy);
                                nearbyValues.Add((distance, nvalue));
                            }
                        }
                    }
                }
                
                if (nearbyValues.Count >= 3)
                {
                    // Interpolación ponderada por distancia inversa
                    float totalWeight = 0;
                    float weightedSum = 0;
                    
                    foreach (var (dist, value) in nearbyValues)
                    {
                        float weight = 1.0f / (dist * dist);
                        weightedSum += value * weight;
                        totalWeight += weight;
                    }
                    
                    depth = weightedSum / totalWeight;
                    return true;
                }
                
                depth = 0;
                return false;
            }
            
            public void Add(int x, int y, float depth)
            {
                long key = GetKey(x, y);
                
                // Agregar a L1
                if (_l1Cache.Count < L1_MAX_SIZE)
                {
                    _l1Cache[key] = depth;
                }
                else
                {
                    // LRU: eliminar una entrada aleatoria (simplificado)
                    var firstKey = _l1Cache.Keys.First();
                    _l1Cache.Remove(firstKey);
                    _l1Cache[key] = depth;
                }
                
                // Escribir a L2
                if (_l2Cache != null)
                {
                    try
                    {
                        _l2Cache.Write(key * sizeof(float), depth);
                    }
                    catch { }
                }
            }
            
            private long GetKey(int x, int y)
            {
                return (long)y * _width + x;
            }
        }
        
        public DepthExtractorFast(UIApplication uiApp, int resolution = 512, int subsampleFactor = 4)
        {
            _uiApp = uiApp;
            _resolution = resolution;
            _subsampleFactor = Math.Max(2, Math.Min(8, subsampleFactor));
            
            // Inicializar GPU si está habilitado
            if (UseGpuAcceleration)
            {
                try
                {
                    // Intentar primero con ComputeSharp
                    try
                    {
                        _gpuManager = new GpuAccelerationManager();
                    }
                    catch
                    {
                        // Si falla, usar versión simplificada
                        _gpuManager = new GpuAccelerationSimple();
                    }
                    
                    System.Diagnostics.Debug.WriteLine($"GPU Acceleration (Fast): {(_gpuManager.IsGpuAvailable ? "Disponible" : "No disponible (usando CPU paralela)")}");
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error inicializando GPU: {ex.Message}");
                    UseGpuAcceleration = false;
                }
            }
        }
        
        public void ExtractDepthMap(View3D view3D, string outputPath, string timestamp, int width, int height, IList<XYZ> viewCorners)
        {
            Document doc = _uiApp.ActiveUIDocument.Document;
            
            var viewOrientation = view3D.GetOrientation();
            var eyePosition = viewOrientation.EyePosition;
            var forwardDirection = viewOrientation.ForwardDirection.Normalize();
            var upDirection = viewOrientation.UpDirection.Normalize();
            var rightDirection = forwardDirection.CrossProduct(upDirection).Normalize();
            
            // Reconstruir las cuatro esquinas del plano de la vista
            XYZ bottomLeft = viewCorners[0];
            XYZ topRight = viewCorners[1];
            XYZ viewX_vec = topRight - bottomLeft;
            XYZ viewWidthVector = viewX_vec.DotProduct(rightDirection) * rightDirection;
            XYZ viewHeightVector = viewX_vec.DotProduct(upDirection) * upDirection;
            XYZ bottomRight = bottomLeft + viewWidthVector;
            XYZ topLeft = bottomLeft + viewHeightVector;
            
            var (minDepth, maxDepth) = CalculateDepthRange(doc, view3D, eyePosition, forwardDirection, viewCorners);
            
            System.Diagnostics.Debug.WriteLine($"=== WabiSabi Depth Fast Debug (GPU Enhanced) ===");
            System.Diagnostics.Debug.WriteLine($"Subsample Factor: {_subsampleFactor}");
            System.Diagnostics.Debug.WriteLine($"Output Resolution: {width}x{height}");
            System.Diagnostics.Debug.WriteLine($"Sample Resolution: {width/_subsampleFactor}x{height/_subsampleFactor}");
            System.Diagnostics.Debug.WriteLine($"GPU Acceleration: {(UseGpuAcceleration && _gpuManager?.IsGpuAvailable == true ? "Enabled" : "Disabled")}");
            System.Diagnostics.Debug.WriteLine($"Adaptive Sampling: {(UseAdaptiveSampling ? "Enabled" : "Disabled")}");
            
            // Crear memory-mapped file para caché de profundidad
            long cacheSize = (long)width * height * sizeof(float);
            string cacheName = $"WabiSabiDepthCache_{Guid.NewGuid():N}";
            _depthCacheMmf = MemoryMappedFile.CreateNew(cacheName, cacheSize);
            _depthCacheAccessor = _depthCacheMmf.CreateViewAccessor();
            
            // Inicializar caché con valores negativos (no procesados)
            unsafe
            {
                byte* ptr = null;
                _depthCacheAccessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
                for (long i = 0; i < cacheSize; i += sizeof(float))
                {
                    *(float*)(ptr + i) = -1.0f;
                }
                _depthCacheAccessor.SafeMemoryMappedViewHandle.ReleasePointer();
            }
            
            double[,] finalDepth;
            
            if (UseAdaptiveSampling)
            {
                finalDepth = ProcessWithAdaptiveSampling(
                    doc, view3D, eyePosition, forwardDirection, upDirection, rightDirection,
                    bottomLeft, bottomRight, topLeft, topRight,
                    width, height, minDepth, maxDepth
                );
            }
            else
            {
                finalDepth = ProcessWithUniformSampling(
                    doc, view3D, eyePosition,
                    bottomLeft, bottomRight, topLeft, topRight,
                    width, height, minDepth, maxDepth
                );
            }
            
            // Generar imagen final
            GenerateDepthImage(finalDepth, width, height, outputPath, timestamp);
            
            System.Diagnostics.Debug.WriteLine("=== End WabiSabi Depth Fast Debug ===");
        }
        
        /// <summary>
        /// Procesamiento con muestreo adaptativo inteligente
        /// </summary>
        private double[,] ProcessWithAdaptiveSampling(
            Document doc, View3D view3D, XYZ eyePosition,
            XYZ forwardDirection, XYZ upDirection, XYZ rightDirection,
            XYZ bottomLeft, XYZ bottomRight, XYZ topLeft, XYZ topRight,
            int width, int height, double minDepth, double maxDepth)
        {
            // Fase 1: Muestreo inicial muy disperso
            int initialSampleFactor = _subsampleFactor * 2;
            var coarseSamples = SampleDepthUniform(
                doc, view3D, eyePosition,
                bottomLeft, bottomRight, topLeft, topRight,
                width / initialSampleFactor, height / initialSampleFactor,
                minDepth, maxDepth
            );
            
            // Fase 2: Detectar áreas de alto cambio (bordes)
            var edgeMap = DetectEdges(coarseSamples);
            
            // Fase 3: Muestreo fino en áreas de alto detalle
            var refinedSamples = RefineHighDetailAreas(
                doc, view3D, eyePosition,
                bottomLeft, bottomRight, topLeft, topRight,
                width, height, minDepth, maxDepth,
                coarseSamples, edgeMap
            );
            
            // Fase 4: Interpolación final inteligente
            if (UseGpuAcceleration && _gpuManager?.IsGpuAvailable == true)
            {
                return InterpolateWithGpu(refinedSamples, width, height);
            }
            else
            {
                return InterpolateAdaptive(refinedSamples, width, height);
            }
        }
        
        /// <summary>
        /// Detecta bordes en el mapa de profundidad
        /// </summary>
        private bool[,] DetectEdges(double[,] samples)
        {
            int sHeight = samples.GetLength(0);
            int sWidth = samples.GetLength(1);
            var edges = new bool[sHeight, sWidth];
            
            const double EDGE_THRESHOLD = 0.1; // 10% de cambio se considera borde
            
            Parallel.For(1, sHeight - 1, y =>
            {
                for (int x = 1; x < sWidth - 1; x++)
                {
                    double center = samples[y, x];
                    
                    // Sobel operator simplificado
                    double gx = Math.Abs(samples[y, x + 1] - samples[y, x - 1]);
                    double gy = Math.Abs(samples[y + 1, x] - samples[y - 1, x]);
                    double gradient = Math.Sqrt(gx * gx + gy * gy);
                    
                    edges[y, x] = gradient > EDGE_THRESHOLD;
                }
            });
            
            return edges;
        }
        
        /// <summary>
        /// Refina áreas de alto detalle con muestreo adicional
        /// </summary>
        private double[,] RefineHighDetailAreas(
            Document doc, View3D view3D, XYZ eyePosition,
            XYZ bottomLeft, XYZ bottomRight, XYZ topLeft, XYZ topRight,
            int width, int height, double minDepth, double maxDepth,
            double[,] coarseSamples, bool[,] edgeMap)
        {
            var refinedSamples = new double[height, width];
            var cache = new AdvancedDepthCache(width, height, _depthCacheAccessor);
            
            // Copiar muestras iniciales
            int coarseHeight = coarseSamples.GetLength(0);
            int coarseWidth = coarseSamples.GetLength(1);
            int scaleFactor = width / coarseWidth;
            
            for (int cy = 0; cy < coarseHeight; cy++)
            {
                for (int cx = 0; cx < coarseWidth; cx++)
                {
                    int fx = cx * scaleFactor;
                    int fy = cy * scaleFactor;
                    if (fx < width && fy < height)
                    {
                        refinedSamples[fy, fx] = coarseSamples[cy, cx];
                        cache.Add(fx, fy, (float)coarseSamples[cy, cx]);
                    }
                }
            }
            
            // Obtener elementos para intersección
            ICollection<ElementId> elementIds = GetIntersectableElementIds(doc, view3D);
            var intersector = new ReferenceIntersector(elementIds, FindReferenceTarget.Element, view3D)
            {
                FindReferencesInRevitLinks = false,
                TargetType = FindReferenceTarget.Face
            };
            
            // Refinar áreas con bordes
            var refinementTasks = new List<(int x, int y)>();
            
            for (int y = 0; y < height; y += _subsampleFactor / 2)
            {
                for (int x = 0; x < width; x += _subsampleFactor / 2)
                {
                    int cx = x / scaleFactor;
                    int cy = y / scaleFactor;
                    
                    if (cx < coarseWidth && cy < coarseHeight && edgeMap[cy, cx])
                    {
                        // Esta área necesita refinamiento
                        for (int dy = 0; dy < _subsampleFactor / 2; dy++)
                        {
                            for (int dx = 0; dx < _subsampleFactor / 2; dx++)
                            {
                                int px = x + dx;
                                int py = y + dy;
                                if (px < width && py < height)
                                {
                                    refinementTasks.Add((px, py));
                                }
                            }
                        }
                    }
                }
            }
            
            // Procesar refinamientos
            int processed = 0;
            foreach (var (x, y) in refinementTasks)
            {
                if (!cache.TryGetValue(x, y, out _))
                {
                    double u_param = (double)x / (width - 1);
                    double v_param = 1.0 - ((double)y / (height - 1));

                    XYZ point_bottom = bottomLeft.Add(u_param * (bottomRight - bottomLeft));
                    XYZ point_top = topLeft.Add(u_param * (topRight - topLeft));
                    XYZ targetPoint = point_bottom.Add(v_param * (point_top - point_bottom));

                    XYZ rayDirection = (targetPoint - eyePosition).Normalize();
                    
                    double distance = GetRayDistance(intersector, eyePosition, rayDirection);
                    
                    if (distance >= 0)
                    {
                        double normalized = 1.0 - ((distance - minDepth) / (maxDepth - minDepth));
                        double depth = Math.Max(0.0, Math.Min(1.0, normalized));
                        refinedSamples[y, x] = depth;
                        cache.Add(x, y, (float)depth);
                    }
                }
                
                processed++;
                if (processed % 1000 == 0)
                {
                    System.Windows.Forms.Application.DoEvents();
                }
            }
            
            return refinedSamples;
        }
        
        /// <summary>
        /// Interpolación usando GPU
        /// </summary>
        private double[,] InterpolateWithGpu(double[,] samples, int targetWidth, int targetHeight)
        {
            // TODO: Implementar interpolación bicúbica en GPU usando ComputeSharp
            // Por ahora, usar CPU fallback
            return InterpolateAdaptive(samples, targetWidth, targetHeight);
        }
        
        /// <summary>
        /// Interpolación adaptativa CPU
        /// </summary>
        private double[,] InterpolateAdaptive(double[,] samples, int targetWidth, int targetHeight)
        {
            var result = _doublePool.Rent(targetWidth * targetHeight);
            var result2D = new double[targetHeight, targetWidth];
            
            try
            {
                // Usar interpolación bicúbica para mejor calidad
                Parallel.For(0, targetHeight, y =>
                {
                    for (int x = 0; x < targetWidth; x++)
                    {
                        // Encontrar los 16 puntos más cercanos para interpolación bicúbica
                        double value = BicubicInterpolate(samples, x, y, targetWidth, targetHeight);
                        result2D[y, x] = value;
                    }
                });
                
                return result2D;
            }
            finally
            {
                _doublePool.Return(result);
            }
        }
        
        /// <summary>
        /// Interpolación bicúbica
        /// </summary>
        private double BicubicInterpolate(double[,] samples, int x, int y, int targetWidth, int targetHeight)
        {
            int sourceHeight = samples.GetLength(0);
            int sourceWidth = samples.GetLength(1);
            
            float fx = (float)x * sourceWidth / targetWidth;
            float fy = (float)y * sourceHeight / targetHeight;
            
            int x0 = (int)fx;
            int y0 = (int)fy;
            
            if (x0 >= sourceWidth - 1) x0 = sourceWidth - 2;
            if (y0 >= sourceHeight - 1) y0 = sourceHeight - 2;
            
            float dx = fx - x0;
            float dy = fy - y0;
            
            // Interpolación bilineal simple (bicúbica es más compleja)
            double v00 = samples[y0, x0];
            double v10 = samples[y0, Math.Min(x0 + 1, sourceWidth - 1)];
            double v01 = samples[Math.Min(y0 + 1, sourceHeight - 1), x0];
            double v11 = samples[Math.Min(y0 + 1, sourceHeight - 1), Math.Min(x0 + 1, sourceWidth - 1)];
            
            double v0 = v00 * (1 - dx) + v10 * dx;
            double v1 = v01 * (1 - dx) + v11 * dx;
            
            return v0 * (1 - dy) + v1 * dy;
        }
        
        /// <summary>
        /// Procesamiento con muestreo uniforme (modo estándar)
        /// </summary>
        private double[,] ProcessWithUniformSampling(
            Document doc, View3D view3D, XYZ eyePosition,
            XYZ bottomLeft, XYZ bottomRight, XYZ topLeft, XYZ topRight,
            int width, int height, double minDepth, double maxDepth)
        {
            int sampleWidth = width / _subsampleFactor;
            int sampleHeight = height / _subsampleFactor;
            
            var depthSamples = SampleDepthUniform(
                doc, view3D, eyePosition,
                bottomLeft, bottomRight, topLeft, topRight,
                sampleWidth, sampleHeight,
                minDepth, maxDepth
            );
            
            // Interpolación final
            return InterpolateDepthMap(depthSamples, width, height);
        }
        
        /// <summary>
        /// Muestreo uniforme de profundidad
        /// </summary>
        private double[,] SampleDepthUniform(
            Document doc, View3D view3D, XYZ eyePosition,
            XYZ bottomLeft, XYZ bottomRight, XYZ topLeft, XYZ topRight,
            int sampleWidth, int sampleHeight,
            double minDepth, double maxDepth)
        {
            double[,] depthSamples = new double[sampleHeight, sampleWidth];
            
            ICollection<ElementId> elementIds = GetIntersectableElementIds(doc, view3D);
            var intersector = new ReferenceIntersector(elementIds, FindReferenceTarget.Element, view3D)
            {
                FindReferencesInRevitLinks = false,
                TargetType = FindReferenceTarget.Face
            };
            
            // CORREGIDO: Se pasa _depthCacheAccessor, que puede ser nulo.
            var cache = new AdvancedDepthCache(sampleWidth, sampleHeight, _depthCacheAccessor);
            
            // Orden de escaneo optimizado (espiral desde el centro)
            var scanOrder = GenerateSpiralScanOrder(sampleWidth, sampleHeight);
            
            int processed = 0;
            foreach (var (sx, sy) in scanOrder)
            {
                if (cache.TryGetValue(sx, sy, out float cachedDepth))
                {
                    depthSamples[sy, sx] = cachedDepth;
                }
                else
                {
                    double u_param = (double)sx / (sampleWidth - 1);
                    double v_param = 1.0 - ((double)sy / (sampleHeight - 1));

                    XYZ point_bottom = bottomLeft.Add(u_param * (bottomRight - bottomLeft));
                    XYZ point_top = topLeft.Add(u_param * (topRight - topLeft));
                    XYZ targetPoint = point_bottom.Add(v_param * (point_top - point_bottom));

                    XYZ rayDirection = (targetPoint - eyePosition).Normalize();
                    
                    double distance = GetRayDistance(intersector, eyePosition, rayDirection);
                    
                    if (distance < 0)
                    {
                        depthSamples[sy, sx] = 0;
                    }
                    else
                    {
                        double normalized = 1.0 - ((distance - minDepth) / (maxDepth - minDepth));
                        double depth = Math.Max(0.0, Math.Min(1.0, normalized));
                        depthSamples[sy, sx] = depth;
                        cache.Add(sx, sy, (float)depth);
                    }
                }
                
                processed++;
                if (processed % 100 == 0)
                {
                    System.Windows.Forms.Application.DoEvents();
                }
            }
            
            return depthSamples;
        }
        
        private List<(int x, int y)> GenerateSpiralScanOrder(int width, int height)
        {
            var order = new List<(int, int)>(width * height);
            int centerX = width / 2;
            int centerY = height / 2;
            
            order.Add((centerX, centerY));
            
            int radius = 1;
            while (order.Count < width * height)
            {
                for (int y = centerY - radius; y <= centerY + radius && order.Count < width * height; y++)
                {
                    for (int x = centerX - radius; x <= centerX + radius && order.Count < width * height; x++)
                    {
                        if (Math.Abs(x - centerX) == radius || Math.Abs(y - centerY) == radius)
                        {
                            if (x >= 0 && x < width && y >= 0 && y < height)
                            {
                                if (!order.Contains((x, y)))
                                {
                                    order.Add((x, y));
                                }
                            }
                        }
                    }
                }
                radius++;
            }
            
            return order;
        }
        
        private double[,] InterpolateDepthMap(double[,] samples, int targetWidth, int targetHeight)
        {
            int sampleHeight = samples.GetLength(0);
            int sampleWidth = samples.GetLength(1);
            var result = new double[targetHeight, targetWidth];
            
            // Pool de memoria para evitar allocaciones
            float scaleX = (float)sampleWidth / targetWidth;
            float scaleY = (float)sampleHeight / targetHeight;
            
            Parallel.For(0, targetHeight, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, y =>
            {
                for (int x = 0; x < targetWidth; x++)
                {
                    float fx = x * scaleX;
                    float fy = y * scaleY;
                    
                    int x0 = (int)fx;
                    int y0 = (int)fy;
                    int x1 = Math.Min(x0 + 1, sampleWidth - 1);
                    int y1 = Math.Min(y0 + 1, sampleHeight - 1);
                    
                    float dx = fx - x0;
                    float dy = fy - y0;
                    
                    x0 = Math.Max(0, x0);
                    y0 = Math.Max(0, y0);
                    
                    double d00 = samples[y0, x0];
                    double d10 = samples[y0, x1];
                    double d01 = samples[y1, x0];
                    double d11 = samples[y1, x1];
                    
                    double depth = (1 - dx) * (1 - dy) * d00 +
                                  dx * (1 - dy) * d10 +
                                  (1 - dx) * dy * d01 +
                                  dx * dy * d11;
                    
                    result[y, x] = depth;
                }
            });
            
            return result;
        }
        
        private void GenerateDepthImage(double[,] depthData, int width, int height, 
            string outputPath, string timestamp)
        {
            // Usar memory-mapped file para imágenes muy grandes
            if (width * height > 1000000) // > 1 megapíxel
            {
                GenerateDepthImageMmf(depthData, width, height, outputPath, timestamp);
                return;
            }
            
            using (Bitmap depthMap = new Bitmap(width, height, PixelFormat.Format24bppRgb))
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
                
                // <<-- CORREGIDO: Se usan nombres completamente calificados para Encoder y EncoderValue -->>
                var encoderParams = new EncoderParameters(1);
                encoderParams.Param[0] = new EncoderParameter(System.Drawing.Imaging.Encoder.Compression, (long)System.Drawing.Imaging.EncoderValue.CompressionLZW);
                
                // <<-- CORREGIDO: Se usa el nombre completamente calificado para ImageFormat -->>
                var pngCodec = ImageCodecInfo.GetImageEncoders().First(codec => codec.FormatID == Drawing.Imaging.ImageFormat.Png.Guid);
                
                string depthPath = System.IO.Path.Combine(outputPath, $"depth_{timestamp}.png");
                depthMap.Save(depthPath, pngCodec, encoderParams);
                
                string currentDepthPath = System.IO.Path.Combine(outputPath, "current_depth.png");
                depthMap.Save(currentDepthPath, pngCodec, encoderParams);
                
                System.Diagnostics.Debug.WriteLine($"Depth map saved to: {currentDepthPath}");
            }
        }
        
        /// <summary>
        /// Genera imagen usando memory-mapped file para datasets muy grandes
        /// </summary>
        private void GenerateDepthImageMmf(double[,] depthData, int width, int height,
            string outputPath, string timestamp)
        {
            // TODO: Implementar generación de imagen usando memory-mapped file
            // Por ahora usar método estándar
            GenerateDepthImage(depthData, width, height, outputPath, timestamp);
        }
        
        private (double min, double max) CalculateDepthRange(Document doc, View3D view3D, XYZ eyePosition, XYZ forwardDirection, IList<XYZ> viewCorners)
        {
            XYZ bottomLeft = viewCorners[0];
            XYZ topRight = viewCorners[1];
            XYZ viewCenter = new XYZ(
                (bottomLeft.X + topRight.X) / 2.0,
                (bottomLeft.Y + topRight.Y) / 2.0,
                (bottomLeft.Z + topRight.Z) / 2.0
            );

            double targetDistance = 10.0;
            double minDepth = 0.1;
            double maxDepth;

            if (!AutoDepthRange)
            {
                maxDepth = ManualDepthDistance;
                System.Diagnostics.Debug.WriteLine($"Using manual depth range: {minDepth:F2} to {maxDepth:F2} feet");
            }
            else
            {
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
                        maxDepth = targetDistance * 1.2;
                    }
                    else
                    {
                        BoundingBoxUV outline = view3D.Outline;
                        double viewWidth = outline.Max.U - outline.Min.U;
                        targetDistance = viewWidth / (2.0 * Math.Tan(Math.PI / 6.0));
                        maxDepth = targetDistance * 1.5;
                    }
                }
                else
                {
                    maxDepth = 50.0;
                }
                
                double minRange = 5.0;
                if (maxDepth - minDepth < minRange)
                {
                    maxDepth = minDepth + minRange;
                }
                
                System.Diagnostics.Debug.WriteLine($"Auto depth range calculated: {minDepth:F2} to {maxDepth:F2} feet (target distance: {targetDistance:F2})");
            }

            return (minDepth, maxDepth);
        }
        
        private double GetRayDistance(ReferenceIntersector intersector, XYZ origin, XYZ direction)
        {
            try
            {
                ReferenceWithContext refContext = intersector.FindNearest(origin, direction);
                if (refContext != null)
                {
                    double distance = refContext.Proximity;
                    
                    if (_debugRayCount < 5)
                    {
                        _debugRayCount++;
                        System.Diagnostics.Debug.WriteLine($"Sample ray hit {_debugRayCount}: Distance={distance:F2}");
                    }
                    
                    return distance;
                }
                return -1;
            }
            catch (Exception ex)
            {
                if (_debugRayCount == 0)
                {
                    _debugRayCount++;
                    System.Diagnostics.Debug.WriteLine($"RayDistance Error: {ex.Message}");
                }
                return -1;
            }
        }
        
        private ICollection<ElementId> GetIntersectableElementIds(Document doc, View3D view3D)
        {
            var allowedCategories = new List<BuiltInCategory>
            {
                BuiltInCategory.OST_Walls,
                BuiltInCategory.OST_Floors,
                BuiltInCategory.OST_Roofs,
                BuiltInCategory.OST_Ceilings,
                BuiltInCategory.OST_GenericModel,
                BuiltInCategory.OST_Furniture,
                BuiltInCategory.OST_StructuralColumns,
                BuiltInCategory.OST_Doors,
                BuiltInCategory.OST_Windows,
                BuiltInCategory.OST_Stairs,
                BuiltInCategory.OST_CurtainWallPanels,
                BuiltInCategory.OST_CurtainWallMullions,
                BuiltInCategory.OST_StairsRailing,
                BuiltInCategory.OST_Ramps,
                BuiltInCategory.OST_Topography,
                BuiltInCategory.OST_Site,
                BuiltInCategory.OST_Parking,
                BuiltInCategory.OST_Planting,
                BuiltInCategory.OST_Entourage,
                BuiltInCategory.OST_Casework,
                BuiltInCategory.OST_Columns,
                BuiltInCategory.OST_MechanicalEquipment,
                BuiltInCategory.OST_ElectricalEquipment,
                BuiltInCategory.OST_ElectricalFixtures,
                BuiltInCategory.OST_LightingFixtures,
                BuiltInCategory.OST_PlumbingFixtures
            };
            
            var categoryFilter = new ElementMulticategoryFilter(allowedCategories);
            
            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WherePasses(categoryFilter)
                .WhereElementIsNotElementType();
            
            return collector.ToElementIds();
        }
        
        public void Dispose()
        {
            _depthCacheAccessor?.Dispose();
            _depthCacheMmf?.Dispose();
            _gpuManager?.Dispose();
        }
    }
}