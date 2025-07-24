// LockFreeCameraRingBuffer.cs - Versión corregida con manejo robusto de errores
using System;
using System.Threading;
using System.Diagnostics;

namespace WabiSabiBridge.DirectContext
{
    /// <summary>
    /// Ring buffer thread-safe para datos de cámara con manejo robusto de errores
    /// </summary>
    public class LockFreeCameraRingBuffer
    {
        private readonly CameraData[] _buffer;
        private volatile int _writeIndex;
        private volatile int _readIndex;
        private readonly int _capacityMask;
        private readonly int _capacity;
        private volatile int _count; // Contador de elementos para debugging

        public LockFreeCameraRingBuffer(int capacity)
        {
            // Validación mejorada
            if (capacity <= 0)
            {
                Debug.WriteLine($"[LockFreeCameraRingBuffer] Error: capacidad <= 0: {capacity}");
                throw new ArgumentException("La capacidad debe ser mayor que cero.", nameof(capacity));
            }

            // Verificar si es potencia de 2
            if ((capacity & (capacity - 1)) != 0)
            {
                Debug.WriteLine($"[LockFreeCameraRingBuffer] Error: capacidad no es potencia de 2: {capacity}");
                
                // Auto-corregir a la siguiente potencia de 2
                int correctedCapacity = GetNextPowerOfTwo(capacity);
                Debug.WriteLine($"[LockFreeCameraRingBuffer] Corrigiendo capacidad de {capacity} a {correctedCapacity}");
                capacity = correctedCapacity;
            }
            
            _capacity = capacity;
            _capacityMask = capacity - 1;
            _buffer = new CameraData[capacity];
            _writeIndex = 0;
            _readIndex = 0;
            _count = 0;

            Debug.WriteLine($"[LockFreeCameraRingBuffer] Inicializado con capacidad: {capacity}");
        }

        /// <summary>
        /// Encuentra la siguiente potencia de 2 mayor o igual al valor dado
        /// </summary>
        private static int GetNextPowerOfTwo(int value)
        {
            if (value <= 0) return 2;
            
            value--;
            value |= value >> 1;
            value |= value >> 2;
            value |= value >> 4;
            value |= value >> 8;
            value |= value >> 16;
            return value + 1;
        }

        public bool TryWrite(CameraData data)
        {
            try
            {
                int currentWrite = _writeIndex;
                int nextWrite = (currentWrite + 1) & _capacityMask;

                // Verificar si hay espacio
                if (nextWrite == _readIndex)
                {
                    // Buffer lleno - podríamos implementar estrategia de overflow
                    return false;
                }

                // Validar datos antes de escribir
                if (!IsValidCameraData(data))
                {
                    Debug.WriteLine("[LockFreeCameraRingBuffer] Datos de cámara inválidos rechazados.");
                    return false;
                }

                // Escribir datos
                _buffer[currentWrite] = data;
                
                // Barrera de memoria para asegurar que los datos se escriban antes de actualizar el índice
                Thread.MemoryBarrier(); 
                
                // Actualizar índice de escritura
                _writeIndex = nextWrite;
                
                // Incrementar contador (para debugging)
                Interlocked.Increment(ref _count);
                
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[LockFreeCameraRingBuffer] Error en TryWrite: {ex.Message}");
                return false;
            }
        }

        public bool TryRead(out CameraData data)
        {
            data = default;
            
            try
            {
                int currentRead = _readIndex;
                
                // Verificar si hay datos disponibles
                if (currentRead == _writeIndex)
                {
                    return false; // Buffer vacío
                }

                // Leer datos
                data = _buffer[currentRead];
                
                // Barrera de memoria para asegurar que la lectura se complete antes de actualizar el índice
                Thread.MemoryBarrier();
                
                // Actualizar índice de lectura
                _readIndex = (currentRead + 1) & _capacityMask;
                
                // Decrementar contador (para debugging)
                Interlocked.Decrement(ref _count);
                
                // Validar datos leídos
                if (!IsValidCameraData(data))
                {
                    Debug.WriteLine("[LockFreeCameraRingBuffer] Datos de cámara leídos son inválidos.");
                    data = default;
                    return false;
                }
                
                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[LockFreeCameraRingBuffer] Error en TryRead: {ex.Message}");
                data = default;
                return false;
            }
        }

        /// <summary>
        /// Valida que los datos de cámara sean razonables
        /// </summary>
        private static bool IsValidCameraData(CameraData data)
        {
            try
            {
                // Verificar que las posiciones no sean NaN o infinitas
                if (double.IsNaN(data.EyePosition.X) || double.IsNaN(data.EyePosition.Y) || double.IsNaN(data.EyePosition.Z) ||
                    double.IsInfinity(data.EyePosition.X) || double.IsInfinity(data.EyePosition.Y) || double.IsInfinity(data.EyePosition.Z))
                {
                    return false;
                }

                if (double.IsNaN(data.ViewDirection.X) || double.IsNaN(data.ViewDirection.Y) || double.IsNaN(data.ViewDirection.Z) ||
                    double.IsInfinity(data.ViewDirection.X) || double.IsInfinity(data.ViewDirection.Y) || double.IsInfinity(data.ViewDirection.Z))
                {
                    return false;
                }

                if (double.IsNaN(data.UpDirection.X) || double.IsNaN(data.UpDirection.Y) || double.IsNaN(data.UpDirection.Z) ||
                    double.IsInfinity(data.UpDirection.X) || double.IsInfinity(data.UpDirection.Y) || double.IsInfinity(data.UpDirection.Z))
                {
                    return false;
                }

                // Verificar que el timestamp sea razonable
                if (data.Timestamp <= 0)
                {
                    return false;
                }

                // Verificar que el número de secuencia sea válido
                if (data.SequenceNumber < 0)
                {
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[LockFreeCameraRingBuffer] Error validando datos: {ex.Message}");
                return false;
            }
        }

        /// <summary>
        /// Obtiene estadísticas del buffer para debugging
        /// </summary>
        public BufferStats GetStats()
        {
            try
            {
                return new BufferStats
                {
                    Capacity = _capacity,
                    Count = _count,
                    WriteIndex = _writeIndex,
                    ReadIndex = _readIndex,
                    IsFull = ((_writeIndex + 1) & _capacityMask) == _readIndex,
                    IsEmpty = _writeIndex == _readIndex
                };
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[LockFreeCameraRingBuffer] Error obteniendo estadísticas: {ex.Message}");
                return new BufferStats { Capacity = _capacity, Count = -1 };
            }
        }

        /// <summary>
        /// Limpia el buffer (útil para debugging)
        /// </summary>
        public void Clear()
        {
            try
            {
                _writeIndex = 0;
                _readIndex = 0;
                _count = 0;
                Thread.MemoryBarrier();
                Debug.WriteLine("[LockFreeCameraRingBuffer] Buffer limpiado.");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[LockFreeCameraRingBuffer] Error limpiando buffer: {ex.Message}");
            }
        }

        public struct BufferStats
        {
            public int Capacity;
            public int Count;
            public int WriteIndex;
            public int ReadIndex;
            public bool IsFull;
            public bool IsEmpty;

            public override string ToString()
            {
                return $"Capacity: {Capacity}, Count: {Count}, WriteIdx: {WriteIndex}, ReadIdx: {ReadIndex}, Full: {IsFull}, Empty: {IsEmpty}";
            }
        }
    }
}
