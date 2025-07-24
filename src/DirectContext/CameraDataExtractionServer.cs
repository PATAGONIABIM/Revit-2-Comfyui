// CameraDataExtractionServer.cs - Versión corregida con RenderScene
using System;
using System.Diagnostics;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.DirectContext3D;
using Autodesk.Revit.DB.ExternalService;
using View = Autodesk.Revit.DB.View;

namespace WabiSabiBridge.DirectContext
{
    public class CameraDataExtractionServer : IDirectContext3DServer
    {
        private readonly LockFreeCameraRingBuffer _cameraBuffer;
        private int _instanceSequenceNumber = 0;
        private readonly Guid _serverId;
        private readonly string _serverName;
        private int _drawCallCount = 0;
        private int _successfulExtractions = 0;
        private int _failedExtractions = 0;
        private readonly object _statsLock = new object();
        
        // Campos para mantener el servidor activo
        private ViewOrientation3D? _lastOrientation;
        private long _lastExtractionTime = 0;
        private const long FORCE_UPDATE_INTERVAL_MS = 16; // ~60 FPS
        private bool _isCapturingContinuously = false;
        private DateTime _lastDrawTime = DateTime.MinValue;

        public CameraDataExtractionServer(LockFreeCameraRingBuffer cameraBuffer)
        {
            _cameraBuffer = cameraBuffer ?? throw new ArgumentNullException(nameof(cameraBuffer));
            _serverId = Guid.NewGuid();
            _serverName = $"WabiSabi_CameraExtractor_{_serverId:N}";
            
            WabiSabiLogger.Log($"CameraDataExtractionServer creado: {_serverName}", LogLevel.Info);
        }

        // Propiedad para activar/desactivar captura continua
        public bool ContinuousCapture 
        { 
            get => _isCapturingContinuously;
            set
            {
                _isCapturingContinuously = value;
                WabiSabiLogger.Log($"Captura continua: {value}", LogLevel.Info);
            }
        }

        public bool CanExecute(View view)
        {
            try
            {
                // Más permisivo para asegurar que se ejecute
                bool canExecute = view?.ViewType == ViewType.ThreeD;
                
                // Log inicial para confirmar que se está preguntando
                if (_drawCallCount == 0 && canExecute)
                {
                    WabiSabiLogger.Log($"CanExecute: Primera llamada para vista {view?.Name}", LogLevel.Info);
                }
                
                return canExecute;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError($"Error en CanExecute: {ex.Message}");
                return false;
            }
        }

        // NUEVO: Implementar método que indica que queremos updates continuos
        public bool NeedsUpdate(View view)
        {
            // Si estamos en modo captura continua, siempre necesitamos actualizaciones
            return _isCapturingContinuously;
        }
        
        private bool ShouldExtractCameraData()
        {
            // Siempre extraer en las primeras llamadas para establecer estado inicial
            if (_drawCallCount <= 5) return true;

            // En modo continuo, extraer según intervalo
            if (_isCapturingContinuously)
            {
                long currentTime = Stopwatch.GetTimestamp();
                long elapsedMs = (currentTime - _lastExtractionTime) * 1000 / Stopwatch.Frequency;
                
                if (elapsedMs >= FORCE_UPDATE_INTERVAL_MS)
                {
                    return true;
                }
            }
            
            // Si no tenemos orientación previa, extraer
            if (_lastOrientation == null) return true;

            // Controlar frecuencia para no saturar
            var timeSinceLastDraw = DateTime.Now - _lastDrawTime;
            return timeSinceLastDraw.TotalMilliseconds >= 16; // ~60 FPS max
        }

        private bool HasOrientationChanged(ViewOrientation3D newOrientation)
        {
            if (_lastOrientation == null) return true;

            const double TOLERANCE = 1e-6;

            // Comparar posición del ojo
            if (!AreXYZEqual(_lastOrientation.EyePosition, newOrientation.EyePosition, TOLERANCE))
                return true;

            // Comparar dirección de vista
            if (!AreXYZEqual(_lastOrientation.ForwardDirection, newOrientation.ForwardDirection, TOLERANCE))
                return true;

            // Comparar dirección arriba
            if (!AreXYZEqual(_lastOrientation.UpDirection, newOrientation.UpDirection, TOLERANCE))
                return true;

            return false;
        }

        private static bool AreXYZEqual(XYZ a, XYZ b, double tolerance)
        {
            return Math.Abs(a.X - b.X) < tolerance &&
                   Math.Abs(a.Y - b.Y) < tolerance &&
                   Math.Abs(a.Z - b.Z) < tolerance;
        }

        private static ViewOrientation3D CopyOrientation(ViewOrientation3D orientation)
        {
            // Crear una copia para evitar referencias a objetos que puedan cambiar
            return new ViewOrientation3D(
                new XYZ(orientation.EyePosition.X, orientation.EyePosition.Y, orientation.EyePosition.Z),
                new XYZ(orientation.UpDirection.X, orientation.UpDirection.Y, orientation.UpDirection.Z),
                new XYZ(orientation.ForwardDirection.X, orientation.ForwardDirection.Y, orientation.ForwardDirection.Z)
            );
        }

        private bool ExtractCameraData(View3D view3D, ViewOrientation3D orientation)
        {
            try
            {
                if (!ValidateOrientationData(orientation))
                {
                    return false;
                }

                // Crear datos de cámara
                var cameraData = CreateCameraData(orientation);

                // Escribir al buffer con validación
                bool written = WriteToBufferSafely(cameraData);
                
                if (written && (_successfulExtractions == 1 || _successfulExtractions % 20 == 0))
                {
                    WabiSabiLogger.LogDiagnostic("CameraData", 
                        $"Datos escritos - Seq: {cameraData.SequenceNumber}, " +
                        $"Pos: ({cameraData.EyePosition.X:F2}, {cameraData.EyePosition.Y:F2}, {cameraData.EyePosition.Z:F2})");
                }
                
                return written;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError($"Error en ExtractCameraData: {ex.Message}");
                return false;
            }
        }

        private ViewOrientation3D? GetViewOrientationSafely(View3D view3D)
        {
            try
            {
                return view3D.GetOrientation();
            }
            catch (Autodesk.Revit.Exceptions.InvalidOperationException ex)
            {
                WabiSabiLogger.LogError($"Error obteniendo orientación (InvalidOperation): {ex.Message}");
                return null;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError($"Error obteniendo orientación: {ex.Message}");
                return null;
            }
        }

        private static bool ValidateOrientationData(ViewOrientation3D orientation)
        {
            try
            {
                if (orientation.EyePosition == null || orientation.ForwardDirection == null || orientation.UpDirection == null)
                {
                    WabiSabiLogger.LogDiagnostic("CameraServer", "ValidateOrientation: Datos nulos.");
                    return false;
                }

                // Validar que los valores no sean NaN o infinitos
                if (!IsValidXYZ(orientation.EyePosition) || 
                    !IsValidXYZ(orientation.ForwardDirection) || 
                    !IsValidXYZ(orientation.UpDirection))
                {
                    WabiSabiLogger.LogDiagnostic("CameraServer", "ValidateOrientation: Valores inválidos.");
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError($"Error en ValidateOrientation: {ex.Message}");
                return false;
            }
        }

        private static bool IsValidXYZ(XYZ point)
        {
            try
            {
                return !double.IsNaN(point.X) && !double.IsNaN(point.Y) && !double.IsNaN(point.Z) &&
                       !double.IsInfinity(point.X) && !double.IsInfinity(point.Y) && !double.IsInfinity(point.Z);
            }
            catch (Exception)
            {
                return false;
            }
        }

        private CameraData CreateCameraData(ViewOrientation3D orientation)
        {
            int currentSeq = System.Threading.Interlocked.Increment(ref _instanceSequenceNumber);

            return new CameraData
            {
                EyePosition = orientation.EyePosition,
                ViewDirection = orientation.ForwardDirection,
                UpDirection = orientation.UpDirection,
                SequenceNumber = currentSeq,
                Timestamp = Stopwatch.GetTimestamp()
            };
        }

        private bool WriteToBufferSafely(CameraData cameraData)
        {
            try
            {
                if (!_cameraBuffer.TryWrite(cameraData))
                {
                    if (cameraData.SequenceNumber % 20 == 0)
                    {
                        var stats = _cameraBuffer.GetStats();
                        WabiSabiLogger.LogDiagnostic("CameraServer", 
                            $"Buffer lleno. Stats: {stats}");
                    }
                    return false;
                }

                return true;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error escribiendo al buffer", ex);
                return false;
            }
        }

        private void IncrementSuccessfulExtractions()
        {
            lock (_statsLock)
            {
                _successfulExtractions++;
            }
        }

        private void IncrementFailedExtractions()
        {
            lock (_statsLock)
            {
                _failedExtractions++;
            }
        }

        // CAMBIO: Este método debe retornar true para forzar más actualizaciones
        public bool UseInTransparentPass(View view)
        {
            // Retornar true para obtener más llamadas
            return true;
        }

        public Guid GetServerId() => _serverId;
        
        public string GetName() => _serverName;

        public string GetServerName() => _serverName;

        public string GetDescription() => "Servidor optimizado para extracción continua de datos de cámara en tiempo real para WabiSabi Bridge";

        public string GetVendorId() => "WabiSabi";
        
        public ExternalServiceId GetServiceId()
        {
            return ExternalServices.BuiltInExternalServices.DirectContext3DService;
        }
        
        public string GetApplicationId() => "WabiSabiBridge_v0.3";
        
        public string GetSourceId() => _serverId.ToString("N");

        // NUEVO: Método para debugging - verificar si el servidor está activo
        public bool IsActive()
        {
            try
            {
                var service = ExternalServiceRegistry.GetService(
                    ExternalServices.BuiltInExternalServices.DirectContext3DService) as MultiServerService;
                    
                if (service != null)
                {
                    var activeServers = service.GetActiveServerIds();
                    return activeServers.Contains(_serverId);
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogDiagnostic("ServerStatus", $"Error verificando estado activo: {ex.Message}");
            }
            
            return false;
        }
        
        // CAMBIO: Retornar un bounding box grande para asegurar que siempre se ejecute
        public Outline? GetBoundingBox(View view)
        {
            // Retornar null indica que el servidor debe ser llamado siempre
            // independientemente de la región visible
            return null;
        }

        public bool UsesHandles()
        {
            return false;
        } 
        
        public ServerStats GetServerStats()
        {
            lock (_statsLock)
            {
                return new ServerStats
                {
                    ServerId = _serverId,
                    DrawCallCount = _drawCallCount,
                    SuccessfulExtractions = _successfulExtractions,
                    FailedExtractions = _failedExtractions,
                    SuccessRate = _drawCallCount > 0 ? (double)_successfulExtractions / _drawCallCount : 0,
                    BufferStats = _cameraBuffer?.GetStats() ?? default
                };
            }
        }

        public struct ServerStats
        {
            public Guid ServerId;
            public int DrawCallCount;
            public int SuccessfulExtractions;
            public int FailedExtractions;
            public double SuccessRate;
            public LockFreeCameraRingBuffer.BufferStats BufferStats;

            public override string ToString()
            {
                return $"Server {ServerId:N}: Calls={DrawCallCount}, Success={SuccessfulExtractions}, Failed={FailedExtractions}, Rate={SuccessRate:P1}, {BufferStats}";
            }
        }

        // CAMBIO CRÍTICO CORREGIDO: La firma del método ahora es correcta, recibiendo View y DisplayStyle.
        // Todo el trabajo de extracción se realiza aquí.
        public void RenderScene(View view, DisplayStyle displayStyle)
        {
            lock (_statsLock)
            {
                _drawCallCount++;
            }

            try
            {
                // Log crítico para confirmar que el método se está llamando
                if (_drawCallCount == 1)
                {
                    WabiSabiLogger.Log("¡RenderScene llamado por PRIMERA VEZ!", LogLevel.Info);
                }
                else if (_drawCallCount % 10 == 0)
                {
                    WabiSabiLogger.Log($"RenderScene llamado: {_drawCallCount} veces", LogLevel.Info);
                }

                if (view == null || view.ViewType != ViewType.ThreeD)
                {
                    if (_drawCallCount == 1)
                    {
                        WabiSabiLogger.LogError("RenderScene: Vista nula o no es 3D");
                    }
                    return;
                }

                var view3D = view as View3D;
                if (view3D == null) return;

                // Extraer datos de cámara
                bool shouldExtract = ShouldExtractCameraData();
                
                if (shouldExtract)
                {
                    ViewOrientation3D? currentOrientation = GetViewOrientationSafely(view3D);
                    
                    if (currentOrientation != null && ExtractCameraData(view3D, currentOrientation))
                    {
                        _lastOrientation = CopyOrientation(currentOrientation);
                        _lastExtractionTime = Stopwatch.GetTimestamp();
                        _lastDrawTime = DateTime.Now;
                        
                        IncrementSuccessfulExtractions();
                        
                        // Log de éxito
                        if (_successfulExtractions == 1)
                        {
                            WabiSabiLogger.Log($"¡PRIMERA extracción exitosa! Seq: {_instanceSequenceNumber}", LogLevel.Info);
                        }
                        else if (_successfulExtractions % 10 == 0)
                        {
                            WabiSabiLogger.Log($"Extracción exitosa #{_successfulExtractions}, Seq: {_instanceSequenceNumber}", LogLevel.Info);
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error crítico en RenderScene", ex);
                IncrementFailedExtractions();
            }
            
            // IMPORTANTE: No crear ningún buffer gráfico, solo extraer datos
            // Retornar inmediatamente para minimizar overhead
        }
    }

    #region Helper Classes (Assuming these definitions exist elsewhere)
    
    

    #endregion
}