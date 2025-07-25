// GpuAccelerationBase.cs - Infraestructura de aceleración GPU para WabiSabi Bridge
using System;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Threading.Tasks;
using ComputeSharp;
using System.Numerics;

// Directivas 'using' necesarias
using WinForms = System.Windows.Forms;
using Autodesk.Revit.UI;
using System.Runtime.InteropServices; // Necesario para MemoryMarshal

namespace WabiSabiBridge.Extractors.Gpu
{
    #region Shaders Originales (Para Mapa de Profundidad)

    [ThreadGroupSize(8, 8, 1)]
    [GeneratedComputeShaderDescriptor]
    [EmbeddedBytecode(8, 8, 1)]
    public readonly partial struct DepthRayTracingShader : IComputeShader
    {
        private readonly ReadOnlyBuffer<Float3> vertices;
        private readonly ReadOnlyBuffer<Int3> triangles;
        private readonly ReadOnlyBuffer<Float3> normals;
        private readonly RayTracingConfig config;
        private readonly ReadWriteBuffer<float> depthBuffer;

        public DepthRayTracingShader(
            ReadOnlyBuffer<Float3> vertices,
            ReadOnlyBuffer<Int3> triangles,
            ReadOnlyBuffer<Float3> normals,
            RayTracingConfig config,
            ReadWriteBuffer<float> depthBuffer)
        {
            this.vertices = vertices;
            this.triangles = triangles;
            this.normals = normals;
            this.config = config;
            this.depthBuffer = depthBuffer;
        }

        public void Execute()
        {
            int2 id = ThreadIds.XY;
            if (id.X >= config.Width || id.Y >= config.Height) return;

            int pixelIndex = id.Y * config.Width + id.X;
            
            float u = (float)id.X / (config.Width - 1);
            float v = 1.0f - ((float)id.Y / (config.Height - 1));
            
            Float3 directionToPixel = config.ViewDirection + (u * config.RightDirection) + (v * config.UpDirection);
            Float3 rayDirection = Hlsl.Normalize(directionToPixel);
            
            float closestDistance = float.MaxValue;
            
            int triangleCount = triangles.Length;
            for (int i = 0; i < triangleCount; i++)
            {
                Int3 tri = triangles[i];
                Float3 v0 = vertices[tri.X];
                Float3 v1 = vertices[tri.Y];
                Float3 v2 = vertices[tri.Z];
                
                float distance = RayTriangleIntersection(config.EyePosition, rayDirection, v0, v1, v2);
                
                if (distance > 0 && distance < closestDistance)
                {
                    closestDistance = distance;
                }
            }
            
            float normalizedDepth = 0.0f;
            if (closestDistance < float.MaxValue)
            {
                normalizedDepth = 1.0f - Hlsl.Saturate(
                    (closestDistance - config.MinDepth) / (config.MaxDepth - config.MinDepth)
                );
            }
            
            depthBuffer[pixelIndex] = normalizedDepth;
        }
        
        private static float RayTriangleIntersection(Float3 origin, Float3 direction, Float3 v0, Float3 v1, Float3 v2)
        {
            const float EPSILON = 0.0000001f;
            Float3 edge1 = v1 - v0;
            Float3 edge2 = v2 - v0;
            Float3 h = Hlsl.Cross(direction, edge2);
            float a = Hlsl.Dot(edge1, h);
            if (a > -EPSILON && a < EPSILON) return -1.0f;
            float f = 1.0f / a;
            Float3 s = origin - v0;
            float u = f * Hlsl.Dot(s, h);
            if (u < 0.0f || u > 1.0f) return -1.0f;
            Float3 q = Hlsl.Cross(s, edge1);
            float v = f * Hlsl.Dot(direction, q);
            if (v < 0.0f || u + v > 1.0f) return -1.0f;
            float t = f * Hlsl.Dot(edge2, q);
            return t > EPSILON ? t : -1.0f;
        }
    }
    #endregion

    #region Nuevos Shaders (Para Renderizado de Líneas)

    // --- PASE 1: Genera un G-Buffer con profundidad y normales ---
    [ThreadGroupSize(8, 8, 1)]
    [GeneratedComputeShaderDescriptor]
    [EmbeddedBytecode(8, 8, 1)]
    public readonly partial struct GBufferGenerationShader : IComputeShader
    {
        private readonly ReadOnlyBuffer<Float3> vertices;
        private readonly ReadOnlyBuffer<Int3> triangles;
        private readonly RayTracingConfig config;
        private readonly ReadWriteBuffer<GpuBufferData> gBuffer;

        public GBufferGenerationShader(
            ReadOnlyBuffer<Float3> vertices,
            ReadOnlyBuffer<Int3> triangles,
            RayTracingConfig config,
            ReadWriteBuffer<GpuBufferData> gBuffer)
        {
            this.vertices = vertices;
            this.triangles = triangles;
            this.config = config;
            this.gBuffer = gBuffer;
        }

        public void Execute()
        {
            int2 id = ThreadIds.XY;
            if (id.X >= config.Width || id.Y >= config.Height) return;

            int pixelIndex = id.Y * config.Width + id.X;

            float u = (float)id.X / (config.Width - 1);
            float v = 1.0f - ((float)id.Y / (config.Height - 1));
            
            Float3 directionToPixel = config.ViewDirection + (u * config.RightDirection) + (v * config.UpDirection);
            Float3 rayDirection = Hlsl.Normalize(directionToPixel);
            
            float closestDistance = float.MaxValue;
            Float3 hitNormal = new Float3(0, 0, 0);
            
            int triangleCount = triangles.Length;
            for (int i = 0; i < triangleCount; i++)
            {
                Int3 tri = triangles[i];
                Float3 v0 = vertices[tri.X];
                Float3 v1 = vertices[tri.Y];
                Float3 v2 = vertices[tri.Z];
                
                float distance = RayTriangleIntersection(config.EyePosition, rayDirection, v0, v1, v2);
                
                if (distance > 0 && distance < closestDistance)
                {
                    closestDistance = distance;
                    Float3 edge1 = v1 - v0;
                    Float3 edge2 = v2 - v0;

                    // --- INICIO DE LA CORRECCIÓN CRÍTICA ---
                    // Calculamos el producto cruz para obtener la normal del triángulo.
                    Float3 faceNormal = Hlsl.Cross(edge1, edge2);

                    // COMPROBACIÓN DE SEGURIDAD: Verificamos si el triángulo es degenerado.
                    // Si la longitud de la normal es casi cero, el triángulo es una línea o un punto.
                    // Intentar normalizarlo causará un crash en el driver de la GPU.
                    if (Hlsl.Length(faceNormal) > 0.0001f)
                    {
                        // Si la normal es válida, la normalizamos.
                        hitNormal = Hlsl.Normalize(faceNormal);
                    }
                    else
                    {
                        // Si el triángulo es degenerado, asignamos una normal por defecto segura (apuntando hacia arriba).
                        // Esto evita el crash y es visualmente imperceptible.
                        hitNormal = new Float3(0, 0, 1);
                    }
                    // --- FIN DE LA CORRECCIÓN CRÍTICA ---
                }
            }
            
            float normalizedDepth = 0.0f;
            if (closestDistance < float.MaxValue)
            {
                normalizedDepth = 1.0f - Hlsl.Saturate((closestDistance - config.MinDepth) / (config.MaxDepth - config.MinDepth));
            }
            
            gBuffer[pixelIndex].Depth = normalizedDepth;
            gBuffer[pixelIndex].WorldNormal = hitNormal;
        }
        
        private static float RayTriangleIntersection(Float3 origin, Float3 direction, Float3 v0, Float3 v1, Float3 v2)
        {
            const float EPSILON = 0.0000001f;
            Float3 edge1 = v1 - v0;
            Float3 edge2 = v2 - v0;
            Float3 h = Hlsl.Cross(direction, edge2);
            float a = Hlsl.Dot(edge1, h);
            if (a > -EPSILON && a < EPSILON) return -1.0f;
            float f = 1.0f / a;
            Float3 s = origin - v0;
            float u = f * Hlsl.Dot(s, h);
            if (u < 0.0f || u > 1.0f) return -1.0f;
            Float3 q = Hlsl.Cross(s, edge1);
            float v = f * Hlsl.Dot(direction, q);
            if (v < 0.0f || u + v > 1.0f) return -1.0f;
            float t = f * Hlsl.Dot(edge2, q);
            return t > EPSILON ? t : -1.0f;
        }
    }

    // --- PASE 2: Procesa el G-Buffer para detectar aristas ---
    [ThreadGroupSize(8, 8, 1)]
    [GeneratedComputeShaderDescriptor]
    [EmbeddedBytecode(8, 8, 1)]
    public readonly partial struct EdgeDetectionShader : IComputeShader
    {
        private readonly ReadWriteBuffer<GpuBufferData> gBuffer;
        private readonly ReadWriteBuffer<Float4> outputImage;
        private readonly int width;
        private readonly int height;
        private readonly float depthThreshold;
        private readonly float normalThreshold;
        private readonly Float4 edgeColor;
        private readonly Float4 backgroundColor;

        public EdgeDetectionShader(
            ReadWriteBuffer<GpuBufferData> gBuffer,
            ReadWriteBuffer<Float4> outputImage,
            int width, int height,
            float depthThreshold, float normalThreshold,
            Float4 edgeColor, Float4 backgroundColor)
        {
            this.gBuffer = gBuffer;
            this.outputImage = outputImage;
            this.width = width;
            this.height = height;
            this.depthThreshold = depthThreshold;
            this.normalThreshold = normalThreshold;
            this.edgeColor = edgeColor;
            this.backgroundColor = backgroundColor;
        }

        public void Execute()
        {
            int2 id = ThreadIds.XY;
            if (id.X >= width || id.Y >= height) return;

            int index = id.Y * width + id.X;
            GpuBufferData centerPixel = gBuffer[index];

            if (centerPixel.Depth <= 0.001f) // Es fondo si no hay geometría
            {
                outputImage[index] = backgroundColor;
                return;
            }

            int leftIdx = id.Y * width + Hlsl.Max(0, id.X - 1);
            int rightIdx = id.Y * width + Hlsl.Min(width - 1, id.X + 1);
            int topIdx = Hlsl.Min(height - 1, id.Y + 1) * width + id.X;
            int bottomIdx = Hlsl.Max(0, id.Y - 1) * width + id.X;

            float depthDelta = 0;
            depthDelta += Hlsl.Abs(gBuffer[leftIdx].Depth - centerPixel.Depth);
            depthDelta += Hlsl.Abs(gBuffer[rightIdx].Depth - centerPixel.Depth);
            depthDelta += Hlsl.Abs(gBuffer[topIdx].Depth - centerPixel.Depth);
            depthDelta += Hlsl.Abs(gBuffer[bottomIdx].Depth - centerPixel.Depth);

            float normalDelta = 0;
            normalDelta += 1.0f - Hlsl.Abs(Hlsl.Dot(gBuffer[leftIdx].WorldNormal, centerPixel.WorldNormal));
            normalDelta += 1.0f - Hlsl.Abs(Hlsl.Dot(gBuffer[rightIdx].WorldNormal, centerPixel.WorldNormal));
            normalDelta += 1.0f - Hlsl.Abs(Hlsl.Dot(gBuffer[topIdx].WorldNormal, centerPixel.WorldNormal));
            normalDelta += 1.0f - Hlsl.Abs(Hlsl.Dot(gBuffer[bottomIdx].WorldNormal, centerPixel.WorldNormal));

            if (depthDelta > depthThreshold || normalDelta > normalThreshold)
            {
                outputImage[index] = edgeColor;
            }
            else
            {
                outputImage[index] = backgroundColor;
            }
        }
    }

    #endregion

    public class GpuAccelerationManager : IGpuAccelerationManager, IDisposable
    {
        private GraphicsDevice? _device;
        private MemoryMappedFile? _geometryMmf;
        private MemoryMappedViewAccessor? _geometryAccessor;
        private readonly ArrayPool<float> _floatPool = ArrayPool<float>.Shared;
        private readonly ArrayPool<int> _intPool = ArrayPool<int>.Shared;
        private readonly WinForms.Form? _owner;

        public bool IsGpuAvailable { get; private set; }

        public GpuAccelerationManager(WinForms.Form? owner = null)
        {
            _owner = owner;
            try
            {
                _device = GraphicsDevice.GetDefault();
                IsGpuAvailable = _device != null;
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error inicializando GPU: {ex.Message}");
                IsGpuAvailable = false;
                string errorMessage = "WabiSabi Bridge no pudo inicializar la GPU para aceleración.\n\n" +
                                      "Causa probable: Faltan componentes de DirectX 12 o hay un problema con el controlador.\n\n" +
                                      $"Error técnico de ComputeSharp:\n{ex.Message}";
                Action showDialogAction = () => Autodesk.Revit.UI.TaskDialog.Show("Error de Aceleración GPU", errorMessage);
                if (_owner != null && _owner.IsHandleCreated && _owner.InvokeRequired)
                {
                    _owner.Invoke(showDialogAction);
                }
                else
                {
                    try { showDialogAction(); }
                    catch { System.Diagnostics.Debug.WriteLine(errorMessage); }
                }
            }
        }
        
        public void CreateGeometrySharedMemory(string name, long sizeInBytes)
        {
             if (_geometryMmf == null)
            {
                _geometryMmf = MemoryMappedFile.CreateNew(name, sizeInBytes);
                _geometryAccessor = _geometryMmf.CreateViewAccessor();
            }
        }
        
        public void WriteGeometryData(float[] vertices, int[] indices, float[]? normals)
        {
            if (_geometryAccessor == null) return;
            long offset = 0;
            _geometryAccessor.WriteArray(offset, vertices, 0, vertices.Length);
            offset += vertices.Length * sizeof(float);
            _geometryAccessor.WriteArray(offset, indices, 0, indices.Length);
            if (normals != null)
            {
                offset += indices.Length * sizeof(int);
                _geometryAccessor.WriteArray(offset, normals, 0, normals.Length);
            }
        }
        
        public async Task<float[]> ExecuteDepthRayTracingAsync(ExtractedGeometry geometry, RayTracingConfig config)
        {
            // Este método ahora se considera un fallback o para un modo específico
            // El código interno no necesita cambiar
            return await ExecuteDepthRayTracingFromCacheAsync(_geometryMmf!, geometry.VertexCount, geometry.TriangleCount, config);
        }
        
        /// <summary>
        /// Ejecuta ray tracing para generar un mapa de profundidad (calidad "Alta").
        /// </summary>
        public async Task<float[]> ExecuteDepthRayTracingFromCacheAsync(MemoryMappedFile cacheFile, int vertexCount, int triangleCount, RayTracingConfig config)
        {
            if (!IsGpuAvailable || _device == null)
            {
                return await Task.Run(() => ExecuteDepthRayTracingFromCacheCpu(cacheFile, vertexCount, triangleCount, config));
            }

            return await Task.Run(() =>
            {
                // CAMBIO: Se usa la sintaxis clásica de C# para compatibilidad.
                var geometryData = LoadGeometryFromCache(cacheFile, vertexCount, triangleCount);

                using (var vertexBuffer = geometryData.vertexBuffer)
                using (var triangleBuffer = geometryData.triangleBuffer)
                using (var normalBuffer = geometryData.normalBuffer)
                {
                    int pixelCount = config.Width * config.Height;
                    using (var depthBuffer = _device.AllocateReadWriteBuffer<float>(pixelCount))
                    {
                        _device.For(config.Width, config.Height, new DepthRayTracingShader(
                            vertexBuffer, triangleBuffer, normalBuffer, config, depthBuffer));

                        float[] results = new float[pixelCount];
                        depthBuffer.CopyTo(results);
                        return results;
                    }
                }
            });
        }
        
        /// <summary>
        /// NUEVO: Ejecuta el renderizado de líneas en dos pases (calidad "GPU Líneas").
        /// </summary>
        public async Task<float[]> ExecuteLineRenderAsync(MemoryMappedFile cacheFile, int vertexCount, int triangleCount, RayTracingConfig config)
        {
            if (!IsGpuAvailable || _device == null)
            {
                return new float[config.Width * config.Height * 4]; // Devuelve buffer vacío
            }

            return await Task.Run(() =>
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();

                var geometryData = LoadGeometryFromCache(cacheFile, vertexCount, triangleCount);

                using (var vertexBuffer = geometryData.vertexBuffer)
                using (var triangleBuffer = geometryData.triangleBuffer)
                using (var normalBuffer = geometryData.normalBuffer)
                {
                    using (var gBuffer = _device.AllocateReadWriteBuffer<GpuBufferData>(config.Width * config.Height))
                    {
                        // --- FASE 1: Generar el G-Buffer ---
                        _device.For(config.Width, config.Height, new GBufferGenerationShader(vertexBuffer, triangleBuffer, config, gBuffer));

                        sw.Stop();
                        System.Diagnostics.Debug.WriteLine($"Pase 1 (G-Buffer) completado en {sw.ElapsedMilliseconds}ms");
                        sw.Restart();

                        // --- FASE 2: Detección de Aristas ---
                        using (var outputImageBuffer = _device.AllocateReadWriteBuffer<Float4>(config.Width * config.Height))
                        {
                            var edgeColor = new Float4(1, 1, 1, 1); // Blanco
                            var backgroundColor = new Float4(40 / 255.0f, 43 / 255.0f, 48 / 255.0f, 1);

                            _device.For(config.Width, config.Height, new EdgeDetectionShader(
                                // CORRECCIÓN FINAL: Se pasa el buffer directamente. ComputeSharp lo convierte implícitamente a solo lectura.
                                gBuffer,
                                outputImageBuffer,
                                config.Width, config.Height,
                                0.05f,
                                0.7f,
                                edgeColor,
                                backgroundColor
                            ));

                            float[] results = new float[config.Width * config.Height * 4];
                            
                            // Esta corrección se mantiene.
                            outputImageBuffer.CopyTo(MemoryMarshal.Cast<float, Float4>(results));

                            sw.Stop();
                            System.Diagnostics.Debug.WriteLine($"Pase 2 (Edge Detection) completado en {sw.ElapsedMilliseconds}ms");

                            return results;
                        }
                    }
                }
            });
                
        }
        
        /// <summary>
        /// NUEVO: Helper para cargar geometría y devolver buffers GPU desechables.
        /// </summary>
        private (ReadOnlyBuffer<Float3> vertexBuffer, ReadOnlyBuffer<Int3> triangleBuffer, ReadOnlyBuffer<Float3> normalBuffer) LoadGeometryFromCache(MemoryMappedFile cacheFile, int vertexCount, int triangleCount)
        {
            int vertexFloatCount = vertexCount * 3;
            int triangleIntCount = triangleCount * 3;

            float[] vertices = _floatPool.Rent(vertexFloatCount);
            int[] triangleIndices = _intPool.Rent(triangleIntCount);
            float[] normals = _floatPool.Rent(vertexFloatCount); // Asumimos que siempre hay normales por simplicidad

            try
            {
                using (var accessor = cacheFile.CreateViewAccessor())
                {
                    long offset = 0;
                    accessor.ReadArray(offset, vertices, 0, vertexFloatCount);
                    offset += (long)vertexFloatCount * sizeof(float);
                    accessor.ReadArray(offset, triangleIndices, 0, triangleIntCount);
                    offset += (long)triangleIntCount * sizeof(int);
                    accessor.ReadArray(offset, normals, 0, vertexFloatCount);
                }

                Float3[] gpuVertices = ConvertToFloat3Array(vertices, vertexCount);
                Int3[] gpuTriangles = new Int3[triangleCount];
                for (int i = 0; i < triangleCount; i++)
                    gpuTriangles[i] = new Int3(triangleIndices[i * 3], triangleIndices[i * 3 + 1], triangleIndices[i * 3 + 2]);
                Float3[] gpuNormals = ConvertToFloat3Array(normals, vertexCount);
                
                var vertexBuffer = _device!.AllocateReadOnlyBuffer(gpuVertices);
                var triangleBuffer = _device!.AllocateReadOnlyBuffer(gpuTriangles);
                var normalBuffer = _device!.AllocateReadOnlyBuffer(gpuNormals);

                return (vertexBuffer, triangleBuffer, normalBuffer);
            }
            finally
            {
                _floatPool.Return(vertices);
                _intPool.Return(triangleIndices);
                _floatPool.Return(normals);
            }
        }
        
        #region Fallbacks de CPU
        private float[] ExecuteDepthRayTracingFromCacheCpu(MemoryMappedFile cacheFile, int vertexCount, int triangleCount, RayTracingConfig config)
        {
            int pixelCount = config.Width * config.Height;
            float[] depthBuffer = new float[pixelCount];
            float[] vertices = _floatPool.Rent(vertexCount * 3);
            int[] triangles = _intPool.Rent(triangleCount * 3);
            try
            {
                using (var accessor = cacheFile.CreateViewAccessor())
                {
                    accessor.ReadArray(0, vertices, 0, vertices.Length);
                    accessor.ReadArray((long)vertices.Length * sizeof(float), triangles, 0, triangles.Length);
                }
                var eyePosition = new Vector3(config.EyePosition.X, config.EyePosition.Y, config.EyePosition.Z);
                var viewDirection = new Vector3(config.ViewDirection.X, config.ViewDirection.Y, config.ViewDirection.Z);
                var rightDirection = new Vector3(config.RightDirection.X, config.RightDirection.Y, config.RightDirection.Z);
                var upDirection = new Vector3(config.UpDirection.X, config.UpDirection.Y, config.UpDirection.Z);

                Parallel.For(0, pixelCount, pixelIndex =>
                {
                    int x = pixelIndex % config.Width;
                    int y = pixelIndex / config.Width;
                    float u = (float)x / (config.Width - 1);
                    float v = 1.0f - ((float)y / (config.Height - 1));
                    var rayDir = Vector3.Normalize(viewDirection + u * rightDirection + v * upDirection);

                    float closestDistance = float.MaxValue;
                    for (int i = 0; i < triangleCount; i++)
                    {
                        var v0 = new Vector3(vertices[triangles[i * 3] * 3], vertices[triangles[i * 3] * 3 + 1], vertices[triangles[i * 3] * 3 + 2]);
                        var v1 = new Vector3(vertices[triangles[i * 3 + 1] * 3], vertices[triangles[i * 3 + 1] * 3 + 1], vertices[triangles[i * 3 + 1] * 3 + 2]);
                        var v2 = new Vector3(vertices[triangles[i * 3 + 2] * 3], vertices[triangles[i * 3 + 2] * 3 + 1], vertices[triangles[i * 3 + 2] * 3 + 2]);
                        float distance = RayTriangleIntersectionCpu(eyePosition, rayDir, v0, v1, v2);
                        if (distance > 0 && distance < closestDistance) closestDistance = distance;
                    }
                    if (closestDistance < float.MaxValue)
                    {
                        float normalized = (closestDistance - config.MinDepth) / (config.MaxDepth - config.MinDepth);
                        depthBuffer[pixelIndex] = 1.0f - Math.Max(0, Math.Min(1, normalized));
                    }
                    else
                    {
                        depthBuffer[pixelIndex] = 0;
                    }
                });
            }
            finally
            {
                _floatPool.Return(vertices);
                _intPool.Return(triangles);
            }
            return depthBuffer;
        }

        private static float RayTriangleIntersectionCpu(Vector3 origin, Vector3 direction, Vector3 v0, Vector3 v1, Vector3 v2)
        {
            const float EPSILON = 0.0000001f;
            var edge1 = v1 - v0;
            var edge2 = v2 - v0;
            var h = Vector3.Cross(direction, edge2);
            float a = Vector3.Dot(edge1, h);
            if (a > -EPSILON && a < EPSILON) return -1.0f;
            float f = 1.0f / a;
            var s = origin - v0;
            float u = f * Vector3.Dot(s, h);
            if (u < 0.0f || u > 1.0f) return -1.0f;
            var q = Vector3.Cross(s, edge1);
            float v = f * Vector3.Dot(direction, q);
            if (v < 0.0f || u + v > 1.0f) return -1.0f;
            float t = f * Vector3.Dot(edge2, q);
            return t > EPSILON ? t : -1.0f;
        }
        #endregion

        private Float3[] ConvertToFloat3Array(float[] floats, int count)
        {
            Float3[] result = new Float3[count];
            for (int i = 0; i < count; i++)
                result[i] = new Float3(floats[i * 3], floats[i * 3 + 1], floats[i * 3 + 2]);
            return result;
        }
        
        public void Dispose()
        {
            _geometryAccessor?.Dispose();
            _geometryMmf?.Dispose();
            _device?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}