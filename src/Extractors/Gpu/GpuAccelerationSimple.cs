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