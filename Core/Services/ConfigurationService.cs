// ===== Core/Services/ConfigurationService.cs =====
using System;
using System.IO;
using System.Threading.Tasks;
using Newtonsoft.Json;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;

namespace WabiSabiRevitBridge.Core.Services
{
    public class ConfigurationService : IConfigurationService
    {
        private readonly string _configBasePath;
        private readonly string _mainConfigFile;
        private readonly string _profilesDirectory;
        private WabiSabiConfiguration _currentConfig;
        private readonly object _configLock = new object();

        public ConfigurationService()
        {
            _configBasePath = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
                "WabiSabiRevitBridge"
            );
            
            _mainConfigFile = Path.Combine(_configBasePath, "config.json");
            _profilesDirectory = Path.Combine(_configBasePath, "profiles");
            
            EnsureDirectoriesExist();
        }

        private void EnsureDirectoriesExist()
        {
            Directory.CreateDirectory(_configBasePath);
            Directory.CreateDirectory(_profilesDirectory);
        }

        public WabiSabiConfiguration GetConfiguration()
        {
            lock (_configLock)
            {
                return _currentConfig ?? (_currentConfig = LoadConfigurationAsync().GetAwaiter().GetResult());
            }
        }

        public async Task SaveConfigurationAsync(WabiSabiConfiguration config)
        {
            if (config == null)
                throw new ArgumentNullException(nameof(config));

            try
            {
                var json = JsonConvert.SerializeObject(config, Formatting.Indented);
                await File.WriteAllTextAsync(_mainConfigFile, json);
                
                lock (_configLock)
                {
                    _currentConfig = config;
                }
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException($"Failed to save configuration: {ex.Message}", ex);
            }
        }

        public async Task<WabiSabiConfiguration> LoadConfigurationAsync()
        {
            try
            {
                if (!File.Exists(_mainConfigFile))
                {
                    var defaultConfig = CreateDefaultConfiguration();
                    await SaveConfigurationAsync(defaultConfig);
                    return defaultConfig;
                }

                var json = await File.ReadAllTextAsync(_mainConfigFile);
                var config = JsonConvert.DeserializeObject<WabiSabiConfiguration>(json);
                
                // Validate and migrate if needed
                config = ValidateAndMigrateConfiguration(config);
                
                lock (_configLock)
                {
                    _currentConfig = config;
                }
                
                return config;
            }
            catch (Exception ex)
            {
                // If loading fails, return default configuration
                var defaultConfig = CreateDefaultConfiguration();
                lock (_configLock)
                {
                    _currentConfig = defaultConfig;
                }
                return defaultConfig;
            }
        }

        public void ResetToDefaults()
        {
            var defaultConfig = CreateDefaultConfiguration();
            SaveConfigurationAsync(defaultConfig).GetAwaiter().GetResult();
        }

        public string[] GetProfileNames()
        {
            var profiles = Directory.GetFiles(_profilesDirectory, "*.json");
            var names = new string[profiles.Length];
            
            for (int i = 0; i < profiles.Length; i++)
            {
                names[i] = Path.GetFileNameWithoutExtension(profiles[i]);
            }
            
            return names;
        }

        public async Task<WabiSabiConfiguration> LoadProfileAsync(string profileName)
        {
            var profilePath = Path.Combine(_profilesDirectory, $"{profileName}.json");
            
            if (!File.Exists(profilePath))
                throw new FileNotFoundException($"Profile '{profileName}' not found");

            var json = await File.ReadAllTextAsync(profilePath);
            return JsonConvert.DeserializeObject<WabiSabiConfiguration>(json);
        }

        public async Task SaveProfileAsync(string profileName, WabiSabiConfiguration config)
        {
            if (string.IsNullOrWhiteSpace(profileName))
                throw new ArgumentException("Profile name cannot be empty", nameof(profileName));

            var profilePath = Path.Combine(_profilesDirectory, $"{profileName}.json");
            var json = JsonConvert.SerializeObject(config, Formatting.Indented);
            await File.WriteAllTextAsync(profilePath, json);
        }

        private WabiSabiConfiguration CreateDefaultConfiguration()
        {
            var config = new WabiSabiConfiguration
            {
                Version = "1.0.0",
                Communication = new CommunicationSettings
                {
                    Mode = CommunicationMode.Auto,
                    FilePath = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "WabiSabi"),
                    PipeName = "WabiSabiRevitBridge",
                    SharedMemoryName = "WabiSabiSharedMem",
                    HeartbeatIntervalMs = 1000,
                    ConnectionTimeoutMs = 5000
                },
                Export = new ExportSettings
                {
                    EnabledExtractors = new List<string> { "HiddenLine", "Depth", "Segmentation", "Metadata" },
                    Resolution = new Resolution { Width = 1920, Height = 1080 },
                    ImageFormat = ImageFormat.PNG,
                    CompressionQuality = 90,
                    DepthPrecision = 16
                },
                Performance = new PerformanceSettings
                {
                    OperationMode = OperationMode.Manual,
                    AutoModeThrottlingFps = 5,
                    EnableCache = true,
                    CacheSizeMB = 512,
                    ParallelProcessing = true,
                    TileSize = 256,
                    AdaptiveSampling = true
                },
                UI = new UISettings
                {
                    ShowPreview = true,
                    PreviewSize = 256,
                    ShowNotifications = true,
                    LogLevel = LogLevel.Info,
                    Theme = "Dark"
                }
            };

            // Add default profiles
            config.Profiles["High Quality"] = new Profile
            {
                Name = "High Quality",
                Description = "Maximum quality for final renders",
                Settings = new WabiSabiConfiguration
                {
                    Export = new ExportSettings
                    {
                        Resolution = new Resolution { Width = 3840, Height = 2160 },
                        CompressionQuality = 100,
                        DepthPrecision = 32
                    },
                    Performance = new PerformanceSettings
                    {
                        AdaptiveSampling = false,
                        ParallelProcessing = true
                    }
                }
            };

            config.Profiles["Real-time"] = new Profile
            {
                Name = "Real-time",
                Description = "Optimized for real-time feedback",
                Settings = new WabiSabiConfiguration
                {
                    Export = new ExportSettings
                    {
                        Resolution = new Resolution { Width = 1280, Height = 720 },
                        CompressionQuality = 70
                    },
                    Performance = new PerformanceSettings
                    {
                        OperationMode = OperationMode.Automatic,
                        AutoModeThrottlingFps = 15,
                        AdaptiveSampling = true
                    }
                }
            };

            return config;
        }

        private WabiSabiConfiguration ValidateAndMigrateConfiguration(WabiSabiConfiguration config)
        {
            // Ensure all required properties are set
            if (config.Communication == null)
                config.Communication = new CommunicationSettings();
            
            if (config.Export == null)
                config.Export = new ExportSettings();
            
            if (config.Performance == null)
                config.Performance = new PerformanceSettings();
            
            if (config.UI == null)
                config.UI = new UISettings();
            
            if (config.Profiles == null)
                config.Profiles = new Dictionary<string, Profile>();

            // Validate extractors
            if (config.Export.EnabledExtractors == null || config.Export.EnabledExtractors.Count == 0)
            {
                config.Export.EnabledExtractors = new List<string> { "HiddenLine", "Depth", "Segmentation", "Metadata" };
            }

            return config;
        }
    }
}
