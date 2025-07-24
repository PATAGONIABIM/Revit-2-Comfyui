// In file: DirectContext/CameraData.cs
using Autodesk.Revit.DB;
using System.Runtime.InteropServices;

namespace WabiSabiBridge.DirectContext
{
    // CORRECCIÓN: Esta es la única definición de CameraData que usará la aplicación.
    // Se ha simplificado para coincidir con lo que el servidor realmente extrae.
    [StructLayout(LayoutKind.Sequential)]
    public struct CameraData
    {
        public XYZ EyePosition;
        public XYZ ViewDirection;
        public XYZ UpDirection;
        public int SequenceNumber;
        public long Timestamp;
    }
}