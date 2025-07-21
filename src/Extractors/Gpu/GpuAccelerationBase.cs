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

namespace WabiSabiBridge.Extractors.Gpu
{
    // ... (El código del Shader y la interfaz no cambian, se mantiene igual) ...
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
            
            // --- INICIO DE LA CORRECCIÓN DEL SHADER (MÉTODO ROBUSTO) ---

            // Coordenadas de interpolación en el rango [0, 1]
            float u = (float)id.X / (config.Width - 1);
            // Invertimos Y para que coincida con el sistema de coordenadas de la imagen
            float v = 1.0f - ((float)id.Y / (config.Height - 1));
            
            // Calculamos la dirección del rayo interpolando desde la esquina inferior izquierda.
            // ViewDirection   = Vector desde el Ojo a la Esquina Inferior Izquierda.
            // RightDirection  = Vector del ancho total de la vista.
            // UpDirection     = Vector de la altura total de la vista.
            Float3 directionToPixel = config.ViewDirection + (u * config.RightDirection) + (v * config.UpDirection);
            Float3 rayDirection = Hlsl.Normalize(directionToPixel);

            // --- FIN DE LA CORRECCIÓN ---
            
            float closestDistance = float.MaxValue;
            
            int triangleCount = triangles.Length;
            for (int i = 0; i < triangleCount; i++)
            {
                Int3 tri = triangles[i];
                Float3 v0 = vertices[tri.X];
                Float3 v1 = vertices[tri.Y];
                Float3 v2 = vertices[tri.Z];
                
                float distance = RayTriangleIntersection(
                    config.EyePosition, rayDirection,
                    v0, v1, v2
                );
                
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
        
        private static float RayTriangleIntersection(
            Float3 origin, Float3 direction,
            Float3 v0, Float3 v1, Float3 v2)
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
            
            if (t > EPSILON) return t;
            
            return -1.0f;
        }
    }


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

                // <<-- CORRECCIÓN AQUÍ: Especificamos la ruta completa para desambiguar -->>
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
        
        public async Task<float[]> ExecuteDepthRayTracingAsync(
            ExtractedGeometry geometry,
            RayTracingConfig config)
        {
            if (!IsGpuAvailable || _device == null || _geometryAccessor == null)
            {
                return await Task.Run(() => ExecuteDepthRayTracingCpu(geometry, config));
            }
            
            return await Task.Run(() =>
            {
                int vertexFloatCount = geometry.VertexCount * 3;
                int triangleIntCount = geometry.TriangleCount * 3;
                
                float[] vertices = _floatPool.Rent(vertexFloatCount);
                int[] triangleIndices = _intPool.Rent(triangleIntCount);
                float[]? normals = geometry.NormalsOffset > 0 ? _floatPool.Rent(vertexFloatCount) : null;
                
                try
                {
                    _geometryAccessor.ReadArray(geometry.VerticesOffset, vertices, 0, vertexFloatCount);
                    _geometryAccessor.ReadArray(geometry.IndicesOffset, triangleIndices, 0, triangleIntCount);
                    if (normals != null)
                    {
                        _geometryAccessor.ReadArray(geometry.NormalsOffset, normals, 0, vertexFloatCount);
                    }
                    
                    Float3[] gpuVertices = new Float3[geometry.VertexCount];
                    for (int i = 0; i < geometry.VertexCount; i++)
                        gpuVertices[i] = new Float3(vertices[i * 3], vertices[i * 3 + 1], vertices[i * 3 + 2]);
                    
                    Int3[] gpuTriangles = new Int3[geometry.TriangleCount];
                    for (int i = 0; i < geometry.TriangleCount; i++)
                        gpuTriangles[i] = new Int3(triangleIndices[i * 3], triangleIndices[i * 3 + 1], triangleIndices[i * 3 + 2]);
                    
                    using var vertexBuffer = _device.AllocateReadOnlyBuffer(gpuVertices);
                    using var triangleBuffer = _device.AllocateReadOnlyBuffer(gpuTriangles);
                    using var normalBuffer = normals != null ? 
                        _device.AllocateReadOnlyBuffer(ConvertToFloat3Array(normals, geometry.VertexCount)) : 
                        _device.AllocateReadOnlyBuffer<Float3>(1);
                    
                    int pixelCount = config.Width * config.Height;
                    using var depthBuffer = _device.AllocateReadWriteBuffer<float>(pixelCount);
                    
                    _device.For(config.Width, config.Height, new DepthRayTracingShader(
                        vertexBuffer, triangleBuffer, normalBuffer, config, depthBuffer));
                    
                    float[] results = new float[pixelCount];
                    depthBuffer.CopyTo(results);
                    return results;
                }
                finally
                {
                    _floatPool.Return(vertices);
                    _intPool.Return(triangleIndices);
                    if (normals != null) _floatPool.Return(normals);
                }
            });
        }
        
        private float[] ExecuteDepthRayTracingCpu(ExtractedGeometry geometry, RayTracingConfig config)
        {
            int pixelCount = config.Width * config.Height;
            float[] depthBuffer = new float[pixelCount];

            float[] vertices = _floatPool.Rent(geometry.VertexCount * 3);
            int[] triangles = _intPool.Rent(geometry.TriangleCount * 3);

            try
            {
                _geometryAccessor!.ReadArray(geometry.VerticesOffset, vertices, 0, vertices.Length);
                _geometryAccessor!.ReadArray(geometry.IndicesOffset, triangles, 0, triangles.Length);

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

                    var rayDir = Vector3.Normalize(
                        viewDirection +
                        (u - 0.5f) * 2.0f * rightDirection +
                        (v - 0.5f) * 2.0f * upDirection);

                    float closestDistance = float.MaxValue;

                    for (int i = 0; i < geometry.TriangleCount; i++)
                    {
                        var v0 = new Vector3(vertices[triangles[i * 3] * 3], vertices[triangles[i * 3] * 3 + 1], vertices[triangles[i * 3] * 3 + 2]);
                        var v1 = new Vector3(vertices[triangles[i * 3 + 1] * 3], vertices[triangles[i * 3 + 1] * 3 + 1], vertices[triangles[i * 3 + 1] * 3 + 2]);
                        var v2 = new Vector3(vertices[triangles[i * 3 + 2] * 3], vertices[triangles[i * 3 + 2] * 3 + 1], vertices[triangles[i * 3 + 2] * 3 + 2]);
                        
                        float distance = RayTriangleIntersectionCpu(eyePosition, rayDir, v0, v1, v2);
                        if (distance > 0 && distance < closestDistance)
                        {
                            closestDistance = distance;
                        }
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

        /// <summary>
        /// Ejecuta ray tracing usando geometría desde un caché MMF existente
        /// </summary>
        public async Task<float[]> ExecuteDepthRayTracingFromCacheAsync(
            MemoryMappedFile cacheFile,
            int vertexCount,
            int triangleCount,
            RayTracingConfig config)
        {
            if (!IsGpuAvailable || _device == null)
            {
                return await Task.Run(() => ExecuteDepthRayTracingFromCacheCpu(
                    cacheFile, vertexCount, triangleCount, config));
            }

            return await Task.Run(() =>
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();

                int vertexFloatCount = vertexCount * 3;
                int triangleIntCount = triangleCount * 3;

                float[] vertices = _floatPool.Rent(vertexFloatCount);
                int[] triangleIndices = _intPool.Rent(triangleIntCount);
                float[] normals = _floatPool.Rent(vertexFloatCount);

                try
                {
                    // Leer datos desde el caché MMF
                    using (var accessor = cacheFile.CreateViewAccessor())
                    {
                        long offset = 0;

                        // Leer vértices
                        accessor.ReadArray(offset, vertices, 0, vertexFloatCount);
                        offset += vertexFloatCount * sizeof(float);

                        // Leer índices
                        accessor.ReadArray(offset, triangleIndices, 0, triangleIntCount);
                        offset += triangleIntCount * sizeof(int);

                        // Leer normales
                        accessor.ReadArray(offset, normals, 0, vertexFloatCount);
                    }

                    sw.Stop();
                    System.Diagnostics.Debug.WriteLine($"Caché leído en {sw.ElapsedMilliseconds}ms");
                    sw.Restart();

                    // Convertir a formato GPU
                    Float3[] gpuVertices = new Float3[vertexCount];
                    for (int i = 0; i < vertexCount; i++)
                    {
                        gpuVertices[i] = new Float3(
                            vertices[i * 3],
                            vertices[i * 3 + 1],
                            vertices[i * 3 + 2]);
                    }

                    Int3[] gpuTriangles = new Int3[triangleCount];
                    for (int i = 0; i < triangleCount; i++)
                    {
                        gpuTriangles[i] = new Int3(
                            triangleIndices[i * 3],
                            triangleIndices[i * 3 + 1],
                            triangleIndices[i * 3 + 2]);
                    }

                    Float3[] gpuNormals = new Float3[vertexCount];
                    for (int i = 0; i < vertexCount; i++)
                    {
                        gpuNormals[i] = new Float3(
                            normals[i * 3],
                            normals[i * 3 + 1],
                            normals[i * 3 + 2]);
                    }

                    sw.Stop();
                    System.Diagnostics.Debug.WriteLine($"Datos convertidos en {sw.ElapsedMilliseconds}ms");
                    sw.Restart();

                    // Crear buffers GPU
                    using var vertexBuffer = _device.AllocateReadOnlyBuffer(gpuVertices);
                    using var triangleBuffer = _device.AllocateReadOnlyBuffer(gpuTriangles);
                    using var normalBuffer = _device.AllocateReadOnlyBuffer(gpuNormals);

                    int pixelCount = config.Width * config.Height;
                    using var depthBuffer = _device.AllocateReadWriteBuffer<float>(pixelCount);

                    // Ejecutar shader
                    _device.For(config.Width, config.Height, new DepthRayTracingShader(
                        vertexBuffer, triangleBuffer, normalBuffer, config, depthBuffer));

                    // Obtener resultados
                    float[] results = new float[pixelCount];
                    depthBuffer.CopyTo(results);

                    sw.Stop();
                    System.Diagnostics.Debug.WriteLine($"GPU ray tracing completado en {sw.ElapsedMilliseconds}ms");

                    return results;
                }
                finally
                {
                    _floatPool.Return(vertices);
                    _intPool.Return(triangleIndices);
                    _floatPool.Return(normals);
                }
            });
        }

        /// <summary>
        /// Versión CPU del ray tracing desde caché (fallback)
        /// </summary>
        private float[] ExecuteDepthRayTracingFromCacheCpu(
            MemoryMappedFile cacheFile,
            int vertexCount,
            int triangleCount,
            RayTracingConfig config)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();

            int pixelCount = config.Width * config.Height;
            float[] depthBuffer = new float[pixelCount];

            float[] vertices = _floatPool.Rent(vertexCount * 3);
            int[] triangles = _intPool.Rent(triangleCount * 3);

            try
            {
                // Leer desde caché
                using (var accessor = cacheFile.CreateViewAccessor())
                {
                    long offset = 0;
                    accessor.ReadArray(offset, vertices, 0, vertices.Length);
                    offset += vertices.Length * sizeof(float);
                    accessor.ReadArray(offset, triangles, 0, triangles.Length);
                }

                sw.Stop();
                System.Diagnostics.Debug.WriteLine($"CPU: Caché leído en {sw.ElapsedMilliseconds}ms");
                sw.Restart();

                var eyePosition = new Vector3(config.EyePosition.X, config.EyePosition.Y, config.EyePosition.Z);
                var viewDirection = new Vector3(config.ViewDirection.X, config.ViewDirection.Y, config.ViewDirection.Z);
                var rightDirection = new Vector3(config.RightDirection.X, config.RightDirection.Y, config.RightDirection.Z);
                var upDirection = new Vector3(config.UpDirection.X, config.UpDirection.Y, config.UpDirection.Z);

                // Ray tracing paralelo en CPU
                Parallel.For(0, pixelCount, new ParallelOptions
                {
                    MaxDegreeOfParallelism = Environment.ProcessorCount
                },
                pixelIndex =>
                {
                    int x = pixelIndex % config.Width;
                    int y = pixelIndex / config.Width;

                    float u = (float)x / (config.Width - 1);
                    float v = 1.0f - ((float)y / (config.Height - 1));

                    var rayDir = Vector3.Normalize(
                        viewDirection +
                        (u - 0.5f) * 2.0f * rightDirection +
                        (v - 0.5f) * 2.0f * upDirection);

                    float closestDistance = float.MaxValue;

                    // Probar intersección con cada triángulo
                    for (int i = 0; i < triangleCount; i++)
                    {
                        int idx0 = triangles[i * 3] * 3;
                        int idx1 = triangles[i * 3 + 1] * 3;
                        int idx2 = triangles[i * 3 + 2] * 3;

                        var v0 = new Vector3(vertices[idx0], vertices[idx0 + 1], vertices[idx0 + 2]);
                        var v1 = new Vector3(vertices[idx1], vertices[idx1 + 1], vertices[idx1 + 2]);
                        var v2 = new Vector3(vertices[idx2], vertices[idx2 + 1], vertices[idx2 + 2]);

                        float distance = RayTriangleIntersectionCpu(eyePosition, rayDir, v0, v1, v2);
                        if (distance > 0 && distance < closestDistance)
                        {
                            closestDistance = distance;
                        }
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

                sw.Stop();
                System.Diagnostics.Debug.WriteLine($"CPU ray tracing completado en {sw.ElapsedMilliseconds}ms");

                return depthBuffer;
            }
            finally
            {
                _floatPool.Return(vertices);
                _intPool.Return(triangles);
            }
        }
        
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