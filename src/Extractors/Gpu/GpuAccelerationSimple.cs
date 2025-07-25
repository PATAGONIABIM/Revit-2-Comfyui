// GpuAccelerationSimple.cs - Implementación simplificada sin dependencias complejas
using System;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using System.Threading.Tasks;
using WabiSabiBridge.Extractors.Gpu;
using ComputeSharp;
using System.Numerics; // CORREGIDO: Usar System.Numerics para los cálculos en CPU

namespace WabiSabiBridge.Extractors.Gpu
{
    /// <summary>
    /// Implementación simplificada de aceleración GPU usando solo paralelización CPU
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
            System.Diagnostics.Debug.WriteLine("GPU simplificada: Usando solo paralelización CPU");
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
                System.Diagnostics.Debug.WriteLine($"Error creando MMF: {ex.Message}");
            }
        }
        
        public void WriteGeometryData(float[] vertices, int[] indices, float[]? normals)
        {
            if (_geometryAccessor == null) return;
            
            long offset = 0;
            try
            {
                _geometryAccessor.WriteArray(offset, vertices, 0, vertices.Length);
                offset += vertices.Length * sizeof(float);
                _geometryAccessor.WriteArray(offset, indices, 0, indices.Length);
                if (normals != null)
                {
                    offset += indices.Length * sizeof(int);
                    _geometryAccessor.WriteArray(offset, normals, 0, normals.Length);
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"Error escribiendo a MMF: {ex.Message}");
            }
        }

        /// <summary>
        /// Implementación de respaldo para el renderizado de líneas. No está soportado en el modo simple,
        /// por lo que devuelve una imagen vacía para evitar que la aplicación se bloquee.
        /// </summary>
        public Task<float[]> ExecuteLineRenderAsync(MemoryMappedFile cacheFile, int vertexCount, int triangleCount, RayTracingConfig config)
        {
            // Advertir al desarrollador que este modo no está soportado.
            System.Diagnostics.Debug.WriteLine("ADVERTENCIA: GpuAccelerationSimple (CPU Fallback) no soporta el renderizado de líneas. Se devolverá una imagen vacía.");
            
            // Crear un buffer de imagen del tamaño correcto (RGBA, 4 floats por píxel)
            int bufferSize = config.Width * config.Height * 4;
            float[] emptyImageBuffer = new float[bufferSize];
            
            // (Opcional) Podríamos llenar el buffer con un color de fondo aquí si quisiéramos.
            // Por ahora, devolverlo vacío (negro) está bien.

            // Devolver el buffer vacío dentro de una tarea completada.
            return Task.FromResult(emptyImageBuffer);
        }
        
        public async Task<float[]> ExecuteDepthRayTracingAsync(
            ExtractedGeometry geometry,
            RayTracingConfig config)
        {
            return await Task.Run(() =>
            {
                int pixelCount = config.Width * config.Height;
                float[] depthBuffer = new float[pixelCount];
                
                float[]? vertices = null;
                int[]? triangles = null;
                
                try
                {
                    if (_geometryAccessor == null) return depthBuffer;

                    int vertexFloatCount = geometry.VertexCount * 3;
                    int triangleIntCount = geometry.TriangleCount * 3;
                    
                    vertices = _floatPool.Rent(vertexFloatCount);
                    triangles = _intPool.Rent(triangleIntCount);
                    
                    _geometryAccessor.ReadArray(geometry.VerticesOffset, vertices, 0, vertexFloatCount);
                    _geometryAccessor.ReadArray(geometry.IndicesOffset, triangles, 0, triangleIntCount);

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
                        
                        Vector3 rayDir = Vector3.Normalize(
                            viewDirection +
                            (u - 0.5f) * 2.0f * rightDirection +
                            (v - 0.5f) * 2.0f * upDirection
                        );
                        
                        float closestDistance = float.MaxValue;
                        
                        for (int i = 0; i < geometry.TriangleCount; i++)
                        {
                            int i0 = triangles[i * 3] * 3;
                            int i1 = triangles[i * 3 + 1] * 3;
                            int i2 = triangles[i * 3 + 2] * 3;
                            
                            var v0 = new Vector3(vertices[i0], vertices[i0 + 1], vertices[i0 + 2]);
                            var v1 = new Vector3(vertices[i1], vertices[i1 + 1], vertices[i1 + 2]);
                            var v2 = new Vector3(vertices[i2], vertices[i2 + 1], vertices[i2 + 2]);
                            
                            float distance = RayTriangleIntersection(
                                eyePosition, rayDir, v0, v1, v2
                            );
                            
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
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error en CPU fallback ray tracing: {ex.Message}");
                }
                finally
                {
                    if (vertices != null) _floatPool.Return(vertices);
                    if (triangles != null) _intPool.Return(triangles);
                }
                
                return depthBuffer;
            });
        }
        
        /// <summary>
        /// Ejecuta ray tracing usando geometría desde un caché MMF existente (versión CPU)
        /// </summary>
        public async Task<float[]> ExecuteDepthRayTracingFromCacheAsync(
            MemoryMappedFile cacheFile, 
            int vertexCount, 
            int triangleCount, 
            RayTracingConfig config)
        {
            return await Task.Run(() =>
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                
                int pixelCount = config.Width * config.Height;
                float[] depthBuffer = new float[pixelCount];
                
                float[]? vertices = null;
                int[]? triangles = null;
                
                try
                {
                    int vertexFloatCount = vertexCount * 3;
                    int triangleIntCount = triangleCount * 3;
                    
                    vertices = _floatPool.Rent(vertexFloatCount);
                    triangles = _intPool.Rent(triangleIntCount);
                    
                    // Leer datos desde el caché MMF
                    using (var accessor = cacheFile.CreateViewAccessor())
                    {
                        long offset = 0;
                        
                        // Leer vértices
                        accessor.ReadArray(offset, vertices, 0, vertexFloatCount);
                        offset += vertexFloatCount * sizeof(float);
                        
                        // Leer índices
                        accessor.ReadArray(offset, triangles, 0, triangleIntCount);
                        
                        // No leemos normales en la versión simple
                    }
                    
                    sw.Stop();
                    System.Diagnostics.Debug.WriteLine($"Simple GPU: Caché leído en {sw.ElapsedMilliseconds}ms");
                    sw.Restart();
                    
                    var eyePosition = new Vector3(config.EyePosition.X, config.EyePosition.Y, config.EyePosition.Z);
                    var viewDirection = new Vector3(config.ViewDirection.X, config.ViewDirection.Y, config.ViewDirection.Z);
                    var rightDirection = new Vector3(config.RightDirection.X, config.RightDirection.Y, config.RightDirection.Z);
                    var upDirection = new Vector3(config.UpDirection.X, config.UpDirection.Y, config.UpDirection.Z);
                    
                    // Ray tracing paralelo optimizado
                    int numThreads = Environment.ProcessorCount;
                    int rowsPerThread = config.Height / numThreads;
                    
                    Parallel.For(0, numThreads, threadId =>
                    {
                        int startRow = threadId * rowsPerThread;
                        int endRow = (threadId == numThreads - 1) ? config.Height : startRow + rowsPerThread;
                        
                        for (int y = startRow; y < endRow; y++)
                        {
                            for (int x = 0; x < config.Width; x++)
                            {
                                int pixelIndex = y * config.Width + x;
                                
                                float u = (float)x / (config.Width - 1);
                                float v = 1.0f - ((float)y / (config.Height - 1));
                                
                                Vector3 rayDir = Vector3.Normalize(
                                    viewDirection +
                                    (u - 0.5f) * 2.0f * rightDirection +
                                    (v - 0.5f) * 2.0f * upDirection
                                );
                                
                                float closestDistance = float.MaxValue;
                                
                                // Optimización: procesar triángulos en bloques para mejor caché CPU
                                const int BLOCK_SIZE = 16;
                                for (int blockStart = 0; blockStart < triangleCount; blockStart += BLOCK_SIZE)
                                {
                                    int blockEnd = Math.Min(blockStart + BLOCK_SIZE, triangleCount);
                                    
                                    for (int i = blockStart; i < blockEnd; i++)
                                    {
                                        int idx0 = triangles[i * 3] * 3;
                                        int idx1 = triangles[i * 3 + 1] * 3;
                                        int idx2 = triangles[i * 3 + 2] * 3;
                                        
                                        var v0 = new Vector3(vertices[idx0], vertices[idx0 + 1], vertices[idx0 + 2]);
                                        var v1 = new Vector3(vertices[idx1], vertices[idx1 + 1], vertices[idx1 + 2]);
                                        var v2 = new Vector3(vertices[idx2], vertices[idx2 + 1], vertices[idx2 + 2]);
                                        
                                        float distance = RayTriangleIntersection(eyePosition, rayDir, v0, v1, v2);
                                        if (distance > 0 && distance < closestDistance)
                                        {
                                            closestDistance = distance;
                                        }
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
                            }
                        }
                    });
                    
                    sw.Stop();
                    System.Diagnostics.Debug.WriteLine($"Simple GPU: Ray tracing completado en {sw.ElapsedMilliseconds}ms");
                    
                    return depthBuffer;
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error en Simple GPU ray tracing desde caché: {ex.Message}");
                    return depthBuffer;
                }
                finally
                {
                    if (vertices != null) _floatPool.Return(vertices);
                    if (triangles != null) _intPool.Return(triangles);
                }
            });
        }

        private static float RayTriangleIntersection(Vector3 origin, Vector3 direction, Vector3 v0, Vector3 v1, Vector3 v2)
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

        public void Dispose()
        {
            _geometryAccessor?.Dispose();
            _geometryMmf?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}