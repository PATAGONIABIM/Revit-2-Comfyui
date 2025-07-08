
// ===== Core/Models/WabiSabiConfiguration.cs =====
using System;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace WabiSabiRevitBridge.Core.Models
{
    [JsonObject(MemberSerialization.OptIn)]
    public class WabiSabiConfiguration
    {
        [JsonProperty("version")]
        public string Version { get; set; } = "1.0.0";
        
        [JsonProperty("communication")]
        public CommunicationSettings Communication { get; set; } = new CommunicationSettings();
        
        [JsonProperty("export")]
        public ExportSettings Export { get; set; } = new ExportSettings();
        
        [JsonProperty("performance")]
        public PerformanceSettings Performance { get; set; } = new PerformanceSettings();
        
        [JsonProperty("ui")]
        public UISettings UI { get; set; } = new UISettings();
        
        [JsonProperty("profiles")]
        public Dictionary<string, Profile> Profiles { get; set; } = new Dictionary<string, Profile>();
    }

    public class CommunicationSettings
    {
        [JsonProperty("mode")]
        public CommunicationMode Mode { get; set; } = CommunicationMode.Auto;
        
        [JsonProperty("filePath")]
        public string FilePath { get; set; } = @"C:\Temp\WabiSabi";
        
        [JsonProperty("pipeName")]
        public string PipeName { get; set; } = "WabiSabiRevitBridge";
        
        [JsonProperty("sharedMemoryName")]
        public string SharedMemoryName { get; set; } = "WabiSabiSharedMem";
        
        [JsonProperty("heartbeatIntervalMs")]
        public int HeartbeatIntervalMs { get; set; } = 1000;
        
        [JsonProperty("connectionTimeoutMs")]
        public int ConnectionTimeoutMs { get; set; } = 5000;
    }

    public class ExportSettings
    {
        [JsonProperty("enabledExtractors")]
        public List<string> EnabledExtractors { get; set; } = new List<string> 
        { 
            "HiddenLine", 
            "Depth", 
            "Segmentation", 
            "Metadata" 
        };
        
        [JsonProperty("resolution")]
        public Resolution Resolution { get; set; } = new Resolution { Width = 1920, Height = 1080 };
        
        [JsonProperty("imageFormat")]
        public ImageFormat ImageFormat { get; set; } = ImageFormat.PNG;
        
        [JsonProperty("compressionQuality")]
        public int CompressionQuality { get; set; } = 90;
        
        [JsonProperty("depthPrecision")]
        public int DepthPrecision { get; set; } = 16;
    }

    public class PerformanceSettings
    {
        [JsonProperty("operationMode")]
        public OperationMode OperationMode { get; set; } = OperationMode.Manual;
        
        [JsonProperty("autoModeThrottlingFps")]
        public int AutoModeThrottlingFps { get; set; } = 5;
        
        [JsonProperty("enableCache")]
        public bool EnableCache { get; set; } = true;
        
        [JsonProperty("cacheSizeMB")]
        public int CacheSizeMB { get; set; } = 512;
        
        [JsonProperty("parallelProcessing")]
        public bool ParallelProcessing { get; set; } = true;
        
        [JsonProperty("tileSize")]
        public int TileSize { get; set; } = 256;
        
        [JsonProperty("adaptiveSampling")]
        public bool AdaptiveSampling { get; set; } = true;
    }

    public class UISettings
    {
        [JsonProperty("showPreview")]
        public bool ShowPreview { get; set; } = true;
        
        [JsonProperty("previewSize")]
        public int PreviewSize { get; set; } = 256;
        
        [JsonProperty("showNotifications")]
        public bool ShowNotifications { get; set; } = true;
        
        [JsonProperty("logLevel")]
        public LogLevel LogLevel { get; set; } = LogLevel.Info;
        
        [JsonProperty("theme")]
        public string Theme { get; set; } = "Dark";
    }

    public class Profile
    {
        [JsonProperty("name")]
        public string Name { get; set; }
        
        [JsonProperty("description")]
        public string Description { get; set; }
        
        [JsonProperty("settings")]
        public WabiSabiConfiguration Settings { get; set; }
    }

    public class Resolution
    {
        public int Width { get; set; }
        public int Height { get; set; }
    }

    public enum CommunicationMode
    {
        Auto,
        File,
        NamedPipe,
        SharedMemory
    }

    public enum OperationMode
    {
        Manual,
        Automatic,
        Batch
    }

    public enum LogLevel
    {
        Debug,
        Info,
        Warning,
        Error
    }
}

