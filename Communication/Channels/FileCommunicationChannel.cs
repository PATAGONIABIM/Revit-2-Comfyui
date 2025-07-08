
// ===== Communication/Channels/FileCommunicationChannel.cs =====
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using MessagePack;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;

namespace WabiSabiRevitBridge.Communication.Channels
{
    /// <summary>
    /// File-based communication channel for compatibility
    /// </summary>
    public class FileCommunicationChannel : ICommunicationChannel
    {
        private readonly string _basePath;
        private FileSystemWatcher _watcher;
        private string _currentPath;
        private readonly object _lock = new object();
        private long _bytesSent;
        private long _bytesReceived;
        
        public string ChannelName => "File System";
        public bool IsConnected { get; private set; }
        
        public event EventHandler<ConnectionStateChangedEventArgs> ConnectionStateChanged;
        public event EventHandler<MessageReceivedEventArgs> MessageReceived;

        public FileCommunicationChannel(string basePath)
        {
            _basePath = basePath ?? throw new ArgumentNullException(nameof(basePath));
        }

        public async Task<bool> ConnectAsync(string connectionString)
        {
            return await Task.Run(() =>
            {
                try
                {
                    _currentPath = connectionString ?? _basePath;
                    
                    // Ensure directory exists
                    Directory.CreateDirectory(_currentPath);
                    
                    // Set up file watcher for incoming messages
                    _watcher = new FileSystemWatcher(_currentPath)
                    {
                        Filter = "*.msg",
                        NotifyFilter = NotifyFilters.FileName | NotifyFilters.LastWrite
                    };
                    
                    _watcher.Created += OnFileCreated;
                    _watcher.EnableRaisingEvents = true;
                    
                    IsConnected = true;
                    OnConnectionStateChanged(true, "Connected to file system");
                    
                    return true;
                }
                catch (Exception ex)
                {
                    OnConnectionStateChanged(false, $"Connection failed: {ex.Message}");
                    return false;
                }
            });
        }

        public async Task DisconnectAsync()
        {
            await Task.Run(() =>
            {
                lock (_lock)
                {
                    if (_watcher != null)
                    {
                        _watcher.EnableRaisingEvents = false;
                        _watcher.Created -= OnFileCreated;
                        _watcher.Dispose();
                        _watcher = null;
                    }
                    
                    IsConnected = false;
                    OnConnectionStateChanged(false, "Disconnected");
                }
            });
        }

        public async Task<bool> SendDataAsync(ExportData data)
        {
            if (!IsConnected || data == null)
                return false;

            try
            {
                // Create timestamp-based filename
                var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff");
                var fileName = $"{data.Type}_{timestamp}";
                
                // Save image data
                var imagePath = Path.Combine(_currentPath, $"{fileName}.{GetFileExtension(data.Format)}");
                await File.WriteAllBytesAsync(imagePath, data.ImageData);
                
                // Save metadata
                var metadataPath = Path.Combine(_currentPath, $"{fileName}_metadata.json");
                var metadataJson = Newtonsoft.Json.JsonConvert.SerializeObject(data.Metadata, Newtonsoft.Json.Formatting.Indented);
                await File.WriteAllTextAsync(metadataPath, metadataJson);
                
                // Create notification file
                var message = new CommunicationMessage
                {
                    MessageType = "ExportData",
                    Headers = new Dictionary<string, object>
                    {
                        ["ImagePath"] = imagePath,
                        ["MetadataPath"] = metadataPath,
                        ["ExportType"] = data.Type.ToString(),
                        ["ViewName"] = data.ViewName
                    }
                };
                
                await SendMessageAsync(message);
                
                Interlocked.Add(ref _bytesSent, data.ImageData.Length);
                
                return true;
            }
            catch (Exception ex)
            {
                // Log error
                return false;
            }
        }

        public async Task<bool> SendMessageAsync(CommunicationMessage message)
        {
            if (!IsConnected || message == null)
                return false;

            try
            {
                var timestamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss_fff");
                var messagePath = Path.Combine(_currentPath, $"msg_{timestamp}.msg");
                
                var messageBytes = MessagePackSerializer.Serialize(message);
                await File.WriteAllBytesAsync(messagePath, messageBytes);
                
                Interlocked.Add(ref _bytesSent, messageBytes.Length);
                
                return true;
            }
            catch
            {
                return false;
            }
        }

        public async Task<CommunicationMessage> ReceiveMessageAsync(TimeSpan timeout)
        {
            // File-based channel doesn't support synchronous receive
            // Messages are delivered via events
            await Task.Delay(timeout);
            return null;
        }

        public double GetLatencyMs()
        {
            // File system latency is variable, return average
            return 50.0;
        }

        public long GetBytesSent() => _bytesSent;
        public long GetBytesReceived() => _bytesReceived;

        private void OnFileCreated(object sender, FileSystemEventArgs e)
        {
            Task.Run(async () =>
            {
                try
                {
                    // Wait a bit for file to be fully written
                    await Task.Delay(100);
                    
                    var bytes = await File.ReadAllBytesAsync(e.FullPath);
                    var message = MessagePackSerializer.Deserialize<CommunicationMessage>(bytes);
                    
                    Interlocked.Add(ref _bytesReceived, bytes.Length);
                    
                    MessageReceived?.Invoke(this, new MessageReceivedEventArgs { Message = message });
                    
                    // Delete message file after processing
                    try { File.Delete(e.FullPath); } catch { }
                }
                catch
                {
                    // Ignore invalid message files
                }
            });
        }

        private string GetFileExtension(ImageFormat format)
        {
            return format switch
            {
                ImageFormat.PNG => "png",
                ImageFormat.JPEG => "jpg",
                ImageFormat.EXR => "exr",
                ImageFormat.WebP => "webp",
                _ => "raw"
            };
        }

        private void OnConnectionStateChanged(bool isConnected, string reason)
        {
            ConnectionStateChanged?.Invoke(this, new ConnectionStateChangedEventArgs
            {
                IsConnected = isConnected,
                Reason = reason
            });
        }

        public void Dispose()
        {
            DisconnectAsync().GetAwaiter().GetResult();
        }
    }
}
