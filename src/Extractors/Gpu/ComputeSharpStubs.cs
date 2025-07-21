// ComputeSharpStubs.cs
// Este archivo proporciona definiciones manuales para los atributos de ComputeSharp.
// Se usa como un workaround (solución alternativa) cuando el generador de código fuente
// del paquete NuGet no se ejecuta correctamente en el entorno de compilación,
// lo que causa errores CS0246 (tipo o espacio de nombres no encontrado).

using System;

namespace ComputeSharp
{
    /// <summary>
    /// Definición manual del atributo ThreadGroupSize para satisfacer al compilador.
    /// </summary>
    [AttributeUsage(AttributeTargets.Struct, AllowMultiple = false)]
    internal sealed class ThreadGroupSizeAttribute : Attribute
    {
        /// <summary>
        /// Inicializa una nueva instancia de la clase <see cref="ThreadGroupSizeAttribute"/>.
        /// </summary>
        /// <param name="x">El tamaño del grupo de hilos en el eje X.</param>
        /// <param name="y">El tamaño del grupo de hilos en el eje Y.</param>
        /// <param name="z">El tamaño del grupo de hilos en el eje Z.</param>
        public ThreadGroupSizeAttribute(int x, int y, int z)
        {
        }
    }

    /// <summary>
    /// Definición manual del atributo GeneratedComputeShaderDescriptor para satisfacer al compilador.
    /// </summary>
    [AttributeUsage(AttributeTargets.Struct, AllowMultiple = false)]
    internal sealed class GeneratedComputeShaderDescriptorAttribute : Attribute
    {
    }
}