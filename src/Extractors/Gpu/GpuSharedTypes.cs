// GpuSharedTypes.cs - Tipos compartidos para aceleración GPU
using System;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ComputeSharp;

namespace WabiSabiBridge.Extractors.Gpu
{
    /// <summary>
    /// Interfaz común para gestores de aceleración GPU con soporte de caché
    /// </summary>
    public interface IGpuAccelerationManager : IDisposable
    {
        bool IsGpuAvailable { get; }
        void CreateGeometrySharedMemory(string name, long sizeInBytes);
        void WriteGeometryData(float[] vertices, int[] indices, float[] normals);
        Task<float[]> ExecuteDepthRayTracingAsync(ExtractedGeometry geometry, RayTracingConfig config);
        
        // NUEVO: Método optimizado que usa el caché directamente
        Task<float[]> ExecuteDepthRayTracingFromCacheAsync(
            MemoryMappedFile cacheFile, 
            int vertexCount, 
            int triangleCount, 
            RayTracingConfig config);

        // NUEVO: Método para ejecutar el renderizado de líneas en la GPU
        Task<float[]> ExecuteLineRenderAsync(
            MemoryMappedFile cacheFile,
            int vertexCount,
            int triangleCount,
            RayTracingConfig config);
    }
    
    /// <summary>
    /// Datos de geometría extraídos de Revit para procesamiento GPU
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct ExtractedGeometry
    {
        public int TriangleCount;
        public int VertexCount;
        // Los datos reales se almacenan en memory-mapped file
        public long VerticesOffset;
        public long IndicesOffset;
        public long NormalsOffset;
    }

    /// <summary>
    /// Configuración de ray tracing para GPU
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct RayTracingConfig
    {
        public Float3 EyePosition;
        public Float3 ViewDirection;
        public Float3 UpDirection;
        public Float3 RightDirection;
        public int Width;
        public int Height;
        public float MinDepth;
        public float MaxDepth;
    }

    /// <summary>
    /// Estructura para almacenar los datos del G-Buffer (profundidad y normal).
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct GpuBufferData
    {
        public float Depth;
        public Float3 WorldNormal;
    }
}