// GpuSharedTypes.cs - Tipos compartidos para aceleración GPU
using System;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using ComputeSharp; // <<-- AÑADIDO para usar sus tipos nativos

namespace WabiSabiBridge.Extractors.Gpu
{
    /// <summary>
    /// Interfaz común para gestores de aceleración GPU
    /// </summary>
    public interface IGpuAccelerationManager : IDisposable
    {
        bool IsGpuAvailable { get; }
        void CreateGeometrySharedMemory(string name, long sizeInBytes);
        void WriteGeometryData(float[] vertices, int[] indices, float[] normals);
        Task<float[]> ExecuteDepthRayTracingAsync(ExtractedGeometry geometry, RayTracingConfig config);
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
        public Float3 EyePosition;      // <<-- CORREGIDO
        public Float3 ViewDirection;    // <<-- CORREGIDO
        public Float3 UpDirection;      // <<-- CORREGIDO
        public Float3 RightDirection;   // <<-- CORREGIDO
        public int Width;
        public int Height;
        public float MinDepth;
        public float MaxDepth;
    }
    
    // <<-- ELIMINADAS las estructuras float3 e int3 personalizadas para evitar ambigüedad -->>
}
