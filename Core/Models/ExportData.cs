// ===== Core/Models/ExportData.cs =====
using System;
using System.Collections.Generic;

namespace WabiSabiRevitBridge.Core.Models
{
    /// <summary>
    /// Representa los datos exportados desde Revit
    /// </summary>
    public class ExportData
    {
        public Guid Id { get; set; } = Guid.NewGuid();
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        public string ViewName { get; set; }
        public ExportType Type { get; set; }
        public byte[] ImageData { get; set; }
        public Dictionary<string, object> Metadata { get; set; } = new Dictionary<string, object>();
        public ImageFormat Format { get; set; }
        public int Width { get; set; }
        public int Height { get; set; }
        public int BitDepth { get; set; }
        
        // Performance metrics
        public long ProcessingTimeMs { get; set; }
        public long TransferTimeMs { get; set; }
    }

    public enum ExportType
    {
        HiddenLine,
        Depth,
        Segmentation,
        Normal,
        Metadata,
        Combined
    }

    public enum ImageFormat
    {
        PNG,
        EXR,
        JPEG,
        WebP,
        Raw
    }
}

