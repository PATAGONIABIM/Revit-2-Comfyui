
// ===== UI/ViewModels/MainViewModel.cs =====
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using WabiSabiRevitBridge.Communication;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;
using WabiSabiRevitBridge.Core.Services;
using WabiSabiRevitBridge.Extractors;
using WabiSabiRevitBridge.UI.Commands;

namespace WabiSabiRevitBridge.UI.ViewModels
{
    public class MainViewModel : INotifyPropertyChanged
    {
        private readonly UIDocument _uiDocument;
        private readonly DependencyContainer _container;
        private readonly IConfigurationService _configService;
        private readonly CommunicationManager _communicationManager;
        private readonly List<IDataExtractor> _extractors;
        private readonly DispatcherTimer _updateTimer;
        private readonly StringBuilder _debugLog;
        
        private View3D _currentView;
        private WabiSabiConfiguration _configuration;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isInitialized;

        #region Properties

        private string _connectionStatus = "Disconnected";
        public string ConnectionStatus
        {
            get => _connectionStatus;
            set
            {
                _connectionStatus = value;
                OnPropertyChanged();
            }
        }

        private string _performanceMetrics = "Ready";
        public string PerformanceMetrics
        {
            get => _performanceMetrics;
            set
            {
                _performanceMetrics = value;
                OnPropertyChanged();
            }
        }

        private string _channelInfo = "None";
        public string ChannelInfo
        {
            get => _channelInfo;
            set
            {
                _channelInfo = value;
                OnPropertyChanged();
            }
        }

        private bool _isProcessing;
        public bool IsProcessing
        {
            get => _isProcessing;
            set
            {
                _isProcessing = value;
                OnPropertyChanged();
            }
        }

        private bool _isSettingsOpen;
        public bool IsSettingsOpen
        {
            get => _isSettingsOpen;
            set
            {
                _isSettingsOpen = value;
                OnPropertyChanged();
            }
        }

        // Operation Mode
        private bool _isManualMode = true;
        public bool IsManualMode
        {
            get => _isManualMode;
            set
            {
                _isManualMode = value;
                OnPropertyChanged();
                if (value) UpdateOperationMode(OperationMode.Manual);
            }
        }

        private bool _isAutoMode;
        public bool IsAutoMode
        {
            get => _isAutoMode;
            set
            {
                _isAutoMode = value;
                OnPropertyChanged();
                if (value) UpdateOperationMode(OperationMode.Automatic);
            }
        }

        private bool _isBatchMode;
        public bool IsBatchMode
        {
            get => _isBatchMode;
            set
            {
                _isBatchMode = value;
                OnPropertyChanged();
                if (value) UpdateOperationMode(OperationMode.Batch);
            }
        }

        private bool _isRunning;
        public bool IsRunning
        {
            get => _isRunning;
            set
            {
                _isRunning = value;
                OnPropertyChanged();
            }
        }

        // Export Settings
        private int _exportWidth = 1920;
        public int ExportWidth
        {
            get => _exportWidth;
            set
            {
                _exportWidth = value;
                OnPropertyChanged();
                UpdateExportSettings();
            }
        }

        private int _exportHeight = 1080;
        public int ExportHeight
        {
            get => _exportHeight;
            set
            {
                _exportHeight = value;
                OnPropertyChanged();
                UpdateExportSettings();
            }
        }

        private int _quality = 90;
        public int Quality
        {
            get => _quality;
            set
            {
                _quality = value;
                OnPropertyChanged();
                UpdateExportSettings();
            }
        }

        // Data Selection
        private bool _exportHiddenLines = true;
        public bool ExportHiddenLines
        {
            get => _exportHiddenLines;
            set
            {
                _exportHiddenLines = value;
                OnPropertyChanged();
                UpdateExtractorState("HiddenLine", value);
            }
        }

        private bool _exportDepth = true;
        public bool ExportDepth
        {
            get => _exportDepth;
            set
            {
                _exportDepth = value;
                OnPropertyChanged();
                UpdateExtractorState("Depth", value);
            }
        }

        private bool _exportSegmentation = true;
        public bool ExportSegmentation
        {
            get => _exportSegmentation;
            set
            {
                _exportSegmentation = value;
                OnPropertyChanged();
                UpdateExtractorState("Segmentation", value);
            }
        }

        private bool _exportNormals;
        public bool ExportNormals
        {
            get => _exportNormals;
            set
            {
                _exportNormals = value;
                OnPropertyChanged();
                UpdateExtractorState("Normal", value);
            }
        }

        private bool _exportMetadata = true;
        public bool ExportMetadata
        {
            get => _exportMetadata;
            set
            {
                _exportMetadata = value;
                OnPropertyChanged();
                UpdateExtractorState("Metadata", value);
            }
        }

        // Performance Settings
        private bool _enableCache = true;
        public bool EnableCache
        {
            get => _enableCache;
            set
            {
                _enableCache = value;
                OnPropertyChanged();
                UpdatePerformanceSettings();
            }
        }

        private bool _parallelProcessing = true;
        public bool ParallelProcessing
        {
            get => _parallelProcessing;
            set
            {
                _parallelProcessing = value;
                OnPropertyChanged();
                UpdatePerformanceSettings();
            }
        }

        private bool _adaptiveSampling = true;
        public bool AdaptiveSampling
        {
            get => _adaptiveSampling;
            set
            {
                _adaptiveSampling = value;
                OnPropertyChanged();
                UpdatePerformanceSettings();
            }
        }

        private int _autoModeThrottleFps = 5;
        public int AutoModeThrottleFps
        {
            get => _autoModeThrottleFps;
            set
            {
                _autoModeThrottleFps = value;
                OnPropertyChanged();
                UpdatePerformanceSettings();
            }
        }

        // Preview
        private BitmapImage _previewImage;
        public BitmapImage PreviewImage
        {
            get => _previewImage;
            set
            {
                _previewImage = value;
                OnPropertyChanged();
                OnPropertyChanged(nameof(HasPreview));
            }
        }

        public bool HasPreview => PreviewImage != null;

        // Collections
        public ObservableCollection<string> CommunicationModes { get; }
        public ObservableCollection<ProfileViewModel> Profiles { get; }

        private string _selectedCommunicationMode;
        public string SelectedCommunicationMode
        {
            get => _selectedCommunicationMode;
            set
            {
                _selectedCommunicationMode = value;
                OnPropertyChanged();
                UpdateCommunicationMode();
            }
        }

        private ProfileViewModel _selectedProfile;
        public ProfileViewModel SelectedProfile
        {
            get => _selectedProfile;
            set
            {
                _selectedProfile = value;
                OnPropertyChanged();
            }
        }

        // Debug
        public string DebugLog => _debugLog.ToString();

        #endregion

        #region Commands

        public ICommand ToggleSettingsCommand { get; }
        public ICommand UpdateCommand { get; }
        public ICommand PauseCommand { get; }
        public ICommand StopCommand { get; }
        public ICommand LoadProfileCommand { get; }
        public ICommand SaveProfileCommand { get; }
        public ICommand ClearLogCommand { get; }
        public ICommand ExportLogCommand { get; }
        public ICommand ShowHelpCommand { get; }

        #endregion

        public MainViewModel(UIDocument uiDocument, DependencyContainer container)
        {
            _uiDocument = uiDocument ?? throw new ArgumentNullException(nameof(uiDocument));
            _container = container ?? throw new ArgumentNullException(nameof(container));
            
            _configService = _container.Resolve<IConfigurationService>();
            _communicationManager = _container.Resolve<CommunicationManager>();
            _extractors = new List<IDataExtractor>();
            _debugLog = new StringBuilder();
            
            // Initialize collections
            CommunicationModes = new ObservableCollection<string> 
            { 
                "Auto", "File", "Named Pipe", "Shared Memory" 
            };
            Profiles = new ObservableCollection<ProfileViewModel>();
            
            // Initialize commands
            ToggleSettingsCommand = new RelayCommand(() => IsSettingsOpen = !IsSettingsOpen);
            UpdateCommand = new RelayCommand(async () => await ExecuteUpdateAsync(), () => !IsProcessing);
            PauseCommand = new RelayCommand(ExecutePause, () => IsRunning);
            StopCommand = new RelayCommand(ExecuteStop, () => IsRunning);
            LoadProfileCommand = new RelayCommand(async () => await LoadProfileAsync(), () => SelectedProfile != null);
            SaveProfileCommand = new RelayCommand(async () => await SaveProfileAsync());
            ClearLogCommand = new RelayCommand(() => { _debugLog.Clear(); OnPropertyChanged(nameof(DebugLog)); });
            ExportLogCommand = new RelayCommand(async () => await ExportLogAsync());
            ShowHelpCommand = new RelayCommand(ShowHelp);
            
            // Setup timer for auto mode
            _updateTimer = new DispatcherTimer();
            _updateTimer.Tick += async (s, e) => await AutoUpdateAsync();
            
            // Subscribe to communication events
            _communicationManager.ConnectionStatusChanged += OnConnectionStatusChanged;
            _communicationManager.ChannelChanged += OnChannelChanged;
        }

        public async Task InitializeAsync()
        {
            try
            {
                LogDebug("Initializing WabiSabi Bridge...");
                
                // Load configuration
                _configuration = await _configService.LoadConfigurationAsync();
                ApplyConfiguration();
                
                // Initialize extractors
                InitializeExtractors();
                
                // Load profiles
                await LoadProfilesAsync();
                
                // Get current 3D view
                _currentView = _uiDocument.ActiveView as View3D;
                if (_currentView == null)
                {
                    LogWarning("No active 3D view found");
                }
                
                // Connect to communication channel
                await ConnectAsync();
                
                _isInitialized = true;
                LogDebug("Initialization complete");
            }
            catch (Exception ex)
            {
                LogError($"Initialization failed: {ex.Message}");
            }
        }

        private void InitializeExtractors()
        {
            // Register extractors
            _extractors.Add(new HiddenLineExtractor());
            _extractors.Add(new DepthExtractor());
            _extractors.Add(new SegmentationExtractor());
            _extractors.Add(new NormalMapExtractor());
            _extractors.Add(new MetadataExtractor());
            
            // Configure based on settings
            foreach (var extractor in _extractors)
            {
                var isEnabled = _configuration.Export.EnabledExtractors.Contains(extractor.ExtractorName);
                extractor.IsEnabled = isEnabled;
            }
        }

        private async Task ConnectAsync()
        {
            try
            {
                var mode = ParseCommunicationMode(SelectedCommunicationMode);
                var connected = await _communicationManager.ConnectAsync(mode);
                
                if (connected)
                {
                    LogDebug($"Connected via {_communicationManager.ActiveChannelName}");
                }
                else
                {
                    LogWarning("Failed to establish connection");
                }
            }
            catch (Exception ex)
            {
                LogError($"Connection error: {ex.Message}");
            }
        }

        private async Task ExecuteUpdateAsync()
        {
            if (_currentView == null)
            {
                LogWarning("No active 3D view");
                return;
            }
            
            IsProcessing = true;
            _cancellationTokenSource = new CancellationTokenSource();
            
            try
            {
                LogDebug("Starting export...");
                var stopwatch = Stopwatch.StartNew();
                
                // Create export options
                var options = new ExtractorOptions
                {
                    Width = ExportWidth,
                    Height = ExportHeight,
                    Quality = Quality,
                    UseCache = EnableCache,
                    ParallelProcessing = ParallelProcessing
                };
                
                // Run enabled extractors
                var tasks = new List<Task<ExtractorResult>>();
                
                foreach (var extractor in _extractors.Where(e => e.IsEnabled))
                {
                    tasks.Add(extractor.ExtractAsync(_currentView, options, _cancellationTokenSource.Token));
                }
                
                var results = await Task.WhenAll(tasks);
                
                // Send results
                var successCount = 0;
                foreach (var result in results)
                {
                    if (result.Success)
                    {
                        var sent = await _communicationManager.SendDataAsync(result.Data);
                        if (sent)
                        {
                            successCount++;
                            UpdatePreview(result.Data);
                        }
                    }
                    else
                    {
                        LogWarning($"Extraction failed: {result.ErrorMessage}");
                    }
                }
                
                stopwatch.Stop();
                PerformanceMetrics = $"{successCount} exports in {stopwatch.ElapsedMilliseconds}ms";
                LogDebug($"Export completed: {successCount}/{results.Length} successful");
                
                if (IsAutoMode)
                {
                    StartAutoMode();
                }
            }
            catch (Exception ex)
            {
                LogError($"Export error: {ex.Message}");
            }
            finally
            {
                IsProcessing = false;
                _cancellationTokenSource?.Dispose();
                _cancellationTokenSource = null;
            }
        }

        private void UpdatePreview(ExportData data)
        {
            if (data.Type == ExportType.HiddenLine && data.ImageData != null)
            {
                try
                {
                    var bitmap = new BitmapImage();
                    bitmap.BeginInit();
                    bitmap.StreamSource = new MemoryStream(data.ImageData);
                    bitmap.CacheOption = BitmapCacheOption.OnLoad;
                    bitmap.EndInit();
                    bitmap.Freeze();
                    
                    PreviewImage = bitmap;
                }
                catch
                {
                    // Ignore preview errors
                }
            }
        }

        private void StartAutoMode()
        {
            if (!IsAutoMode || IsRunning)
                return;
            
            var intervalMs = (int)(1000.0 / AutoModeThrottleFps);
            _updateTimer.Interval = TimeSpan.FromMilliseconds(intervalMs);
            _updateTimer.Start();
            IsRunning = true;
            LogDebug($"Auto mode started at {AutoModeThrottleFps} FPS");
        }

        private async Task AutoUpdateAsync()
        {
            if (!IsProcessing)
            {
                await ExecuteUpdateAsync();
            }
        }

        private void ExecutePause()
        {
            _updateTimer.Stop();
            IsRunning = false;
            LogDebug("Auto mode paused");
        }

        private void ExecuteStop()
        {
            _updateTimer.Stop();
            IsRunning = false;
            _cancellationTokenSource?.Cancel();
            LogDebug("Export stopped");
        }

        public void UpdateView(View3D view)
        {
            _currentView = view;
            LogDebug($"View changed to: {view.Name}");
        }

        public void OnDocumentChanged(DocumentChangedEventArgs args)
        {
            // Handle document changes in auto mode
            if (IsAutoMode && IsRunning && !IsProcessing)
            {
                // Trigger update on next timer tick
            }
        }

        public void Cleanup()
        {
            _updateTimer?.Stop();
            _cancellationTokenSource?.Cancel();
            _communicationManager.DisconnectAsync().Wait();
            
            // Save configuration
            _configService.SaveConfigurationAsync(_configuration).Wait();
        }

        #region Configuration Methods

        private void ApplyConfiguration()
        {
            // Apply settings to UI
            SelectedCommunicationMode = _configuration.Communication.Mode.ToString();
            ExportWidth = _configuration.Export.Resolution.Width;
            ExportHeight = _configuration.Export.Resolution.Height;
            Quality = _configuration.Export.CompressionQuality;
            
            EnableCache = _configuration.Performance.EnableCache;
            ParallelProcessing = _configuration.Performance.ParallelProcessing;
            AdaptiveSampling = _configuration.Performance.AdaptiveSampling;
            AutoModeThrottleFps = _configuration.Performance.AutoModeThrottlingFps;
            
            // Set operation mode
            switch (_configuration.Performance.OperationMode)
            {
                case OperationMode.Manual:
                    IsManualMode = true;
                    break;
                case OperationMode.Automatic:
                    IsAutoMode = true;
                    break;
                case OperationMode.Batch:
                    IsBatchMode = true;
                    break;
            }
        }

        private void UpdateExportSettings()
        {
            if (_configuration != null)
            {
                _configuration.Export.Resolution.Width = ExportWidth;
                _configuration.Export.Resolution.Height = ExportHeight;
                _configuration.Export.CompressionQuality = Quality;
            }
        }

        private void UpdatePerformanceSettings()
        {
            if (_configuration != null)
            {
                _configuration.Performance.EnableCache = EnableCache;
                _configuration.Performance.ParallelProcessing = ParallelProcessing;
                _configuration.Performance.AdaptiveSampling = AdaptiveSampling;
                _configuration.Performance.AutoModeThrottlingFps = AutoModeThrottleFps;
            }
        }

        private void UpdateOperationMode(OperationMode mode)
        {
            if (_configuration != null)
            {
                _configuration.Performance.OperationMode = mode;
            }
            
            // Stop auto mode if changing away from it
            if (mode != OperationMode.Automatic && IsRunning)
            {
                ExecuteStop();
            }
        }

        private void UpdateExtractorState(string extractorType, bool enabled)
        {
            var extractor = _extractors.FirstOrDefault(e => e.ExtractorName.Contains(extractorType));
            if (extractor != null)
            {
                extractor.IsEnabled = enabled;
            }
            
            // Update configuration
            if (_configuration != null)
            {
                if (enabled && !_configuration.Export.EnabledExtractors.Contains(extractorType))
                {
                    _configuration.Export.EnabledExtractors.Add(extractorType);
                }
                else if (!enabled)
                {
                    _configuration.Export.EnabledExtractors.Remove(extractorType);
                }
            }
        }

        private void UpdateCommunicationMode()
        {
            var mode = ParseCommunicationMode(SelectedCommunicationMode);
            if (_configuration != null)
            {
                _configuration.Communication.Mode = mode;
            }
            
            // Reconnect with new mode
            Task.Run(async () => await ConnectAsync());
        }

        private CommunicationMode ParseCommunicationMode(string mode)
        {
            return mode switch
            {
                "File" => CommunicationMode.File,
                "Named Pipe" => CommunicationMode.NamedPipe,
                "Shared Memory" => CommunicationMode.SharedMemory,
                _ => CommunicationMode.Auto
            };
        }

        #endregion

        #region Profile Methods

        private async Task LoadProfilesAsync()
        {
            try
            {
                var profileNames = _configService.GetProfileNames();
                
                Profiles.Clear();
                
                // Add built-in profiles
                Profiles.Add(new ProfileViewModel 
                { 
                    Name = "High Quality", 
                    Description = "Maximum quality for final renders" 
                });
                Profiles.Add(new ProfileViewModel 
                { 
                    Name = "Real-time", 
                    Description = "Optimized for real-time feedback" 
                });
                
                // Add user profiles
                foreach (var name in profileNames)
                {
                    Profiles.Add(new ProfileViewModel 
                    { 
                        Name = name, 
                        Description = "User profile" 
                    });
                }
            }
            catch (Exception ex)
            {
                LogError($"Failed to load profiles: {ex.Message}");
            }
        }

        private async Task LoadProfileAsync()
        {
            if (SelectedProfile == null)
                return;
            
            try
            {
                var profile = await _configService.LoadProfileAsync(SelectedProfile.Name);
                _configuration = profile;
                ApplyConfiguration();
                LogDebug($"Loaded profile: {SelectedProfile.Name}");
            }
            catch (Exception ex)
            {
                LogError($"Failed to load profile: {ex.Message}");
            }
        }

        private async Task SaveProfileAsync()
        {
            // In a real implementation, show a dialog to get profile name
            var profileName = $"Profile_{DateTime.Now:yyyyMMdd_HHmmss}";
            
            try
            {
                await _configService.SaveProfileAsync(profileName, _configuration);
                await LoadProfilesAsync();
                LogDebug($"Saved profile: {profileName}");
            }
            catch (Exception ex)
            {
                LogError($"Failed to save profile: {ex.Message}");
            }
        }

        #endregion

        #region Event Handlers

        private void OnConnectionStatusChanged(object sender, ConnectionStatusEventArgs e)
        {
            ConnectionStatus = e.IsConnected ? "Connected" : "Disconnected";
            ChannelInfo = e.ChannelName ?? "None";
        }

        private void OnChannelChanged(object sender, ChannelChangedEventArgs e)
        {
            ChannelInfo = e.ChannelName;
            LogDebug($"Channel changed to: {e.ChannelName}");
        }

        #endregion

        #region Logging Methods

        private void LogDebug(string message)
        {
            Log("DEBUG", message);
        }

        private void LogWarning(string message)
        {
            Log("WARN", message);
        }

        private void LogError(string message)
        {
            Log("ERROR", message);
        }

        private void Log(string level, string message)
        {
            var timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            var logEntry = $"[{timestamp}] [{level}] {message}\n";
            
            _debugLog.AppendLine(logEntry);
            
            // Keep log size reasonable
            if (_debugLog.Length > 100000)
            {
                _debugLog.Remove(0, 50000);
            }
            
            OnPropertyChanged(nameof(DebugLog));
        }

        private async Task ExportLogAsync()
        {
            try
            {
                var logPath = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                    $"WabiSabi_Log_{DateTime.Now:yyyyMMdd_HHmmss}.txt"
                );
                
                await File.WriteAllTextAsync(logPath, _debugLog.ToString());
                LogDebug($"Log exported to: {logPath}");
            }
            catch (Exception ex)
            {
                LogError($"Failed to export log: {ex.Message}");
            }
        }

        private void ShowHelp()
        {
            // In a real implementation, show help dialog or open documentation
            Process.Start(new ProcessStartInfo
            {
                FileName = "https://github.com/yourusername/wabisabi-bridge/wiki",
                UseShellExecute = true
            });
        }

        #endregion

        #region INotifyPropertyChanged

        public event PropertyChangedEventHandler PropertyChanged;

        protected virtual void OnPropertyChanged([CallerMemberName] string propertyName = null)
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        #endregion
    }

    public class ProfileViewModel
    {
        public string Name { get; set; }
        public string Description { get; set; }
    }
}
