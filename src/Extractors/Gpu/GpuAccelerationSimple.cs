// GpuAccelerationSimple.cs - Archivo COMPLETO Y CORREGIDO FINAL
using System;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Threading.Tasks;
using WabiSabiBridge.Extractors.Gpu;

namespace WabiSabiBridge.Extractors.Gpu
{
    /// <summary>
    /// Implementación de respaldo de la aceleración GPU que utiliza paralelización de CPU.
    /// Esta clase implementa la interfaz IGpuAccelerationManager para ser intercambiable.
    /// </summary>
    public class GpuAccelerationSimple : IGpuAccelerationManager, IDisposable
    {
        private MemoryMappedFile? _geometryMmf;
        private MemoryMappedViewAccessor? _geometryAccessor;
        private readonly ArrayPool<float> _floatPool = ArrayPool<float>.Shared;
        private readonly ArrayPool<int> _intPool = ArrayPool<int>.Shared;

        public bool IsGpuAvailable { get; private set; } = false;

        public GpuAccelerationSimple()
        {
            System.Diagnostics.Debug.WriteLine("Gestor de GPU en modo Simple: Usando paralelización de CPU como respaldo.");
        }

        public void CreateGeometrySharedMemory(string name, long sizeInBytes)
        {
            try
            {
                _geometryMmf = MemoryMappedFile.CreateNew(name, sizeInBytes);
                _geometryAccessor = _geometryMmf.CreateViewAccessor();
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error creando MMF en modo Simple: {ex.Message}");
            }
        }

        public void WriteGeometryData(float[] vertices, int[] indices, float[]? normals)
        {
            if (_geometryAccessor == null) return;
            long offset = 0;
            try
            {
                _geometryAccessor.WriteArray(offset, vertices, 0, vertices.Length);
                offset += (long)vertices.Length * sizeof(float);
                _geometryAccessor.WriteArray(offset, indices, 0, indices.Length);
                if (normals != null)
                {
                    offset += (long)indices.Length * sizeof(int);
                    _geometryAccessor.WriteArray(offset, normals, 0, normals.Length);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error escribiendo a MMF en modo Simple: {ex.Message}");
            }
        }

        public Task<float[]> ExecuteLineRenderAsync(string mmfName, int vertexCount, int triangleCount, RayTracingConfig config)
        {
            System.Diagnostics.Debug.WriteLine("ADVERTENCIA: GpuAccelerationSimple (CPU Fallback) no soporta el renderizado de líneas. Se devolverá una imagen vacía.");
            int bufferSize = config.Width * config.Height * 4;
            return Task.FromResult(new float[bufferSize]);
        }

        // --- IMPLEMENTACIÓN DEL MÉTODO QUE FALTABA ---
        // Este método usa el _geometryAccessor que ya fue creado y escrito.
        public async Task<float[]> ExecuteDepthRayTracingAsync(ExtractedGeometry geometry, RayTracingConfig config)
        {
            return await Task.Run(() =>
            {
                int pixelCount = config.Width * config.Height;
                var depthBuffer = new float[pixelCount];
                float[]? vertices = null;
                int[]? triangles = null;

                try
                {
                    if (_geometryAccessor == null) return depthBuffer;

                    int vertexFloatCount = geometry.VertexCount * 3;
                    int triangleIntCount = geometry.TriangleCount * 3;
                    vertices = _floatPool.Rent(vertexFloatCount);
                    triangles = _intPool.Rent(triangleIntCount);

                    // Leer desde el accessor existente usando los offsets
                    _geometryAccessor.ReadArray(geometry.VerticesOffset, vertices, 0, vertexFloatCount);
                    _geometryAccessor.ReadArray(geometry.IndicesOffset, triangles, 0, triangleIntCount);

                    return RayTraceCpuLogic(config, geometry.TriangleCount, vertices, triangles);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error en CPU fallback ray tracing (accessor): {ex.Message}");
                    return depthBuffer;
                }
                finally
                {
                    if (vertices != null) _floatPool.Return(vertices);
                    if (triangles != null) _intPool.Return(triangles);
                }
            });
        }

        // --- IMPLEMENTACIÓN DEL OTRO MÉTODO REQUERIDO POR LA INTERFAZ ---
        // Este método abre un MMF a partir de su nombre.
        public async Task<float[]> ExecuteDepthRayTracingFromCacheAsync(string mmfName, int vertexCount, int triangleCount, RayTracingConfig config)
        {
            return await Task.Run(() =>
            {
                int pixelCount = config.Width * config.Height;
                var depthBuffer = new float[pixelCount];
                float[]? vertices = null;
                int[]? triangles = null;

                try
                {
                    int vertexFloatCount = vertexCount * 3;
                    int triangleIntCount = triangleCount * 3;
                    vertices = _floatPool.Rent(vertexFloatCount);
                    triangles = _intPool.Rent(triangleIntCount);

                    using (var cacheFile = MemoryMappedFile.OpenExisting(mmfName))
                    using (var accessor = cacheFile.CreateViewAccessor())
                    {
                        accessor.ReadArray(0, vertices, 0, vertexFloatCount);
                        accessor.ReadArray((long)vertexFloatCount * sizeof(float), triangles, 0, triangleIntCount);
                    }

                    return RayTraceCpuLogic(config, triangleCount, vertices, triangles);
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error en CPU ray tracing desde caché: {ex.Message}");
                    return depthBuffer;
                }
                finally
                {
                    if (vertices != null) _floatPool.Return(vertices);
                    if (triangles != null) _intPool.Return(triangles);
                }
            });
        }
        
        /// <summary>
        /// Lógica central de trazado de rayos en CPU, compartida por ambos métodos de la interfaz.
        /// </summary>
        private float[] RayTraceCpuLogic(RayTracingConfig config, int triangleCount, float[] vertices, int[] triangles)
        {
            int pixelCount = config.Width * config.Height;
            var depthBuffer = new float[pixelCount];

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

                Vector3 directionToPixel = viewDirection + (u * rightDirection) + (v * upDirection);
                Vector3 rayDir = Vector3.Normalize(directionToPixel);

                float closestDistance = float.MaxValue;

                for (int i = 0; i < triangleCount; i++)
                {
                    int i0 = triangles[i * 3] * 3;
                    int i1 = triangles[i * 3 + 1] * 3;
                    int i2 = triangles[i * 3 + 2] * 3;

                    var v0 = new Vector3(vertices[i0], vertices[i0 + 1], vertices[i0 + 2]);
                    var v1 = new Vector3(vertices[i1], vertices[i1 + 1], vertices[i1 + 2]);
                    var v2 = new Vector3(vertices[i2], vertices[i2 + 1], vertices[i2 + 2]);

                    float distance = RayTriangleIntersection(eyePosition, rayDir, v0, v1, v2);
                    if (distance > 0 && distance < closestDistance)
                    {
                        closestDistance = distance;
                    }
                }

                if (closestDistance < float.MaxValue)
                {
                    float normalized = (closestDistance - config.MinDepth) / (config.MaxDepth - config.MinDepth);
                    depthBuffer[pixelIndex] = 1.0f - Math.Clamp(normalized, 0.0f, 1.0f);
                }
                else
                {
                    depthBuffer[pixelIndex] = 0;
                }
            });

            return depthBuffer;
        }

        private static float RayTriangleIntersection(Vector3 origin, Vector3 direction, Vector3 v0, Vector3 v1, Vector3 v2)
        {
            const float EPSILON = 0.000001f;
            Vector3 edge1 = v1 - v0;
            Vector3 edge2 = v2 - v0;
            Vector3 h = Vector3.Cross(direction, edge2);
            float a = Vector3.Dot(edge1, h);

            if (a > -EPSILON && a < EPSILON) return -1.0f;

            float f = 1.0f / a;
            Vector3 s = origin - v0;
            float u = f * Vector3.Dot(s, h);

            if (u < 0.0f || u > 1.0f) return -1.0f;

            Vector3 q = Vector3.Cross(s, edge1);
            float v = f * Vector3.Dot(direction, q);

            if (v < 0.0f || u + v > 1.0f) return -1.0f;
            
            float t = f * Vector3.Dot(edge2, q);
            return t > EPSILON ? t : -1.0f;
        }

        public void Dispose()
        {
            _geometryAccessor?.Dispose();
            _geometryMmf?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}