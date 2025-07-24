// SerializableCameraData.cs - Estructura serializable para Memory-Mapped Files
using System.Runtime.InteropServices;

namespace WabiSabiBridge.DirectContext
{
    /// <summary>
    /// Versión serializable de CameraData que usa solo tipos de valor
    /// para ser compatible con Memory-Mapped Files
    /// </summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct SerializableCameraData
    {
        // Coordenadas de EyePosition
        public double EyePositionX;
        public double EyePositionY;
        public double EyePositionZ;
        
        // Coordenadas de ViewDirection
        public double ViewDirectionX;
        public double ViewDirectionY;
        public double ViewDirectionZ;
        
        // Coordenadas de UpDirection
        public double UpDirectionX;
        public double UpDirectionY;
        public double UpDirectionZ;
        
        // Metadatos
        public int SequenceNumber;
        public long Timestamp;
        
        /// <summary>
        /// Convierte de CameraData a SerializableCameraData
        /// </summary>
        public static SerializableCameraData FromCameraData(CameraData data)
        {
            return new SerializableCameraData
            {
                EyePositionX = data.EyePosition.X,
                EyePositionY = data.EyePosition.Y,
                EyePositionZ = data.EyePosition.Z,
                
                ViewDirectionX = data.ViewDirection.X,
                ViewDirectionY = data.ViewDirection.Y,
                ViewDirectionZ = data.ViewDirection.Z,
                
                UpDirectionX = data.UpDirection.X,
                UpDirectionY = data.UpDirection.Y,
                UpDirectionZ = data.UpDirection.Z,
                
                SequenceNumber = data.SequenceNumber,
                Timestamp = data.Timestamp
            };
        }
        
        /// <summary>
        /// Convierte de SerializableCameraData a CameraData
        /// </summary>
        public CameraData ToCameraData()
        {
            return new CameraData
            {
                EyePosition = new Autodesk.Revit.DB.XYZ(EyePositionX, EyePositionY, EyePositionZ),
                ViewDirection = new Autodesk.Revit.DB.XYZ(ViewDirectionX, ViewDirectionY, ViewDirectionZ),
                UpDirection = new Autodesk.Revit.DB.XYZ(UpDirectionX, UpDirectionY, UpDirectionZ),
                SequenceNumber = SequenceNumber,
                Timestamp = Timestamp
            };
        }
    }
}