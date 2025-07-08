// ===== Core/Interfaces/IDataExtractor.cs =====
using System.Threading;
using System.Threading.Tasks;
using Autodesk.Revit.DB;

namespace WabiSabiRevitBridge.Core.Interfaces
{
    /// <summary>
    /// Interfaz base para extractores de datos
    /// </summary>
    public interface IDataExtractor
    {
        string ExtractorName { get; }
        ExportType ExportType { get; }
        bool IsEnabled { get; set; }
        
        Task<ExtractorResult> ExtractAsync(View3D view, ExtractorOptions options, CancellationToken cancellationToken = default);
        bool CanExtract(View3D view);
        void Configure(ExtractorConfiguration config);
    }

    public class ExtractorOptions
    {
        public int Width { get; set; } = 1920;
        public int Height { get; set; } = 1080;
        public int Quality { get; set; } = 90;
        public bool UseCache { get; set; } = true;
        public bool ParallelProcessing { get; set; } = true;
        public int TileSize { get; set; } = 256;
    }

    public class ExtractorResult
    {
        public bool Success { get; set; }
        public ExportData Data { get; set; }
        public string ErrorMessage { get; set; }
        public long ProcessingTimeMs { get; set; }
        public Dictionary<string, object> Diagnostics { get; set; } = new Dictionary<string, object>();
    }

    public class ExtractorConfiguration
    {
        public Dictionary<string, object> Settings { get; set; } = new Dictionary<string, object>();
    }
}
