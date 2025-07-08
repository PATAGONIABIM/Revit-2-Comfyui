// ===== Communication/CommunicationManager.cs =====
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;
using WabiSabiRevitBridge.Communication.Channels;

namespace WabiSabiRevitBridge.Communication
{
    /// <summary>
    /// Manages communication channels and automatic failover
    /// </summary>
    public class CommunicationManager : IDisposable
    {
        private readonly Dictionary<CommunicationMode, ICommunicationChannel> _channels;
        private readonly IConfigurationService _configService;
        private ICommunicationChannel _activeChannel;
        private readonly SemaphoreSlim _channelLock = new SemaphoreSlim(1, 1);
        private CancellationTokenSource _heartbeatCts;
        private Task _heartbeatTask;
        
        public event EventHandler<ChannelChangedEventArgs> ChannelChanged;
        public event EventHandler<ConnectionStatusEventArgs> ConnectionStatusChanged;
        
        public ICommunicationChannel ActiveChannel => _activeChannel;
        public bool IsConnected => _activeChannel?.IsConnected ?? false;
        public string ActiveChannelName => _activeChannel?.ChannelName ?? "None";

        public CommunicationManager(IConfigurationService configService)
        {
            _configService = configService ?? throw new ArgumentNullException(nameof(configService));
            _channels = new Dictionary<CommunicationMode, ICommunicationChannel>();
            
            InitializeChannels();
        }

        private void InitializeChannels()
        {
            var config = _configService.GetConfiguration();
            
            // Initialize available channels
            _channels[CommunicationMode.File] = new FileCommunicationChannel(config.Communication.FilePath);
            _channels[CommunicationMode.NamedPipe] = new NamedPipeCommunicationChannel(config.Communication.PipeName);
            _channels[CommunicationMode.SharedMemory] = new SharedMemoryCommunicationChannel(config.Communication.SharedMemoryName);
            
            // Subscribe to connection events
            foreach (var channel in _channels.Values)
            {
                channel.ConnectionStateChanged += OnChannelConnectionStateChanged;
            }
        }

        public async Task<bool> ConnectAsync(CommunicationMode mode = CommunicationMode.Auto)
        {
            await _channelLock.WaitAsync();
            try
            {
                // Disconnect current channel if any
                if (_activeChannel != null)
                {
                    await DisconnectInternalAsync();
                }

                var config = _configService.GetConfiguration();
                
                if (mode == CommunicationMode.Auto)
                {
                    // Try channels in order of preference
                    var preferredOrder = new[] 
                    { 
                        CommunicationMode.SharedMemory, 
                        CommunicationMode.NamedPipe, 
                        CommunicationMode.File 
                    };
                    
                    foreach (var tryMode in preferredOrder)
                    {
                        if (await TryConnectChannelAsync(tryMode))
                        {
                            StartHeartbeat(config.Communication.HeartbeatIntervalMs);
                            return true;
                        }
                    }
                    
                    return false;
                }
                else
                {
                    // Connect to specific channel
                    var success = await TryConnectChannelAsync(mode);
                    if (success)
                    {
                        StartHeartbeat(config.Communication.HeartbeatIntervalMs);
                    }
                    return success;
                }
            }
            finally
            {
                _channelLock.Release();
            }
        }

        private async Task<bool> TryConnectChannelAsync(CommunicationMode mode)
        {
            if (!_channels.TryGetValue(mode, out var channel))
                return false;

            try
            {
                var config = _configService.GetConfiguration();
                var connectionString = GetConnectionString(mode, config);
                
                var connected = await channel.ConnectAsync(connectionString);
                
                if (connected)
                {
                    _activeChannel = channel;
                    OnChannelChanged(mode);
                    return true;
                }
            }
            catch (Exception ex)
            {
                // Log error
                Console.WriteLine($"Failed to connect to {mode}: {ex.Message}");
            }
            
            return false;
        }

        private string GetConnectionString(CommunicationMode mode, WabiSabiConfiguration config)
        {
            return mode switch
            {
                CommunicationMode.File => config.Communication.FilePath,
                CommunicationMode.NamedPipe => config.Communication.PipeName,
                CommunicationMode.SharedMemory => config.Communication.SharedMemoryName,
                _ => throw new ArgumentException($"Unknown communication mode: {mode}")
            };
        }

        public async Task DisconnectAsync()
        {
            await _channelLock.WaitAsync();
            try
            {
                await DisconnectInternalAsync();
            }
            finally
            {
                _channelLock.Release();
            }
        }

        private async Task DisconnectInternalAsync()
        {
            StopHeartbeat();
            
            if (_activeChannel != null)
            {
                await _activeChannel.DisconnectAsync();
                _activeChannel = null;
                OnChannelChanged(CommunicationMode.File); // Default to file when disconnected
            }
        }

        public async Task<bool> SendDataAsync(ExportData data)
        {
            if (_activeChannel == null || !_activeChannel.IsConnected)
                return false;

            try
            {
                return await _activeChannel.SendDataAsync(data);
            }
            catch (Exception ex)
            {
                // Try to reconnect with auto mode
                await HandleCommunicationError(ex);
                return false;
            }
        }

        public async Task<bool> SendMessageAsync(CommunicationMessage message)
        {
            if (_activeChannel == null || !_activeChannel.IsConnected)
                return false;

            try
            {
                return await _activeChannel.SendMessageAsync(message);
            }
            catch (Exception ex)
            {
                await HandleCommunicationError(ex);
                return false;
            }
        }

        private async Task HandleCommunicationError(Exception ex)
        {
            // Log error
            Console.WriteLine($"Communication error: {ex.Message}");
            
            // Try to reconnect
            await ConnectAsync(CommunicationMode.Auto);
        }

        private void StartHeartbeat(int intervalMs)
        {
            StopHeartbeat();
            
            _heartbeatCts = new CancellationTokenSource();
            _heartbeatTask = Task.Run(async () =>
            {
                while (!_heartbeatCts.Token.IsCancellationRequested)
                {
                    try
                    {
                        await Task.Delay(intervalMs, _heartbeatCts.Token);
                        
                        if (_activeChannel != null && _activeChannel.IsConnected)
                        {
                            var heartbeat = new CommunicationMessage
                            {
                                MessageType = "Heartbeat",
                                Headers = new Dictionary<string, object>
                                {
                                    ["Timestamp"] = DateTime.UtcNow,
                                    ["ChannelType"] = _activeChannel.ChannelName
                                }
                            };
                            
                            await _activeChannel.SendMessageAsync(heartbeat);
                        }
                    }
                    catch (Exception ex)
                    {
                        // Handle heartbeat failure
                        await HandleCommunicationError(ex);
                    }
                }
            }, _heartbeatCts.Token);
        }

        private void StopHeartbeat()
        {
            _heartbeatCts?.Cancel();
            _heartbeatTask?.Wait(TimeSpan.FromSeconds(1));
            _heartbeatCts?.Dispose();
            _heartbeatCts = null;
            _heartbeatTask = null;
        }

        private void OnChannelConnectionStateChanged(object sender, ConnectionStateChangedEventArgs e)
        {
            ConnectionStatusChanged?.Invoke(this, new ConnectionStatusEventArgs
            {
                IsConnected = e.IsConnected,
                ChannelName = (sender as ICommunicationChannel)?.ChannelName,
                Reason = e.Reason
            });
        }

        private void OnChannelChanged(CommunicationMode newMode)
        {
            ChannelChanged?.Invoke(this, new ChannelChangedEventArgs
            {
                NewMode = newMode,
                ChannelName = _activeChannel?.ChannelName ?? "None"
            });
        }

        public CommunicationStatistics GetStatistics()
        {
            if (_activeChannel == null)
                return new CommunicationStatistics();

            return new CommunicationStatistics
            {
                ChannelName = _activeChannel.ChannelName,
                IsConnected = _activeChannel.IsConnected,
                LatencyMs = _activeChannel.GetLatencyMs(),
                BytesSent = _activeChannel.GetBytesSent(),
                BytesReceived = _activeChannel.GetBytesReceived()
            };
        }

        public void Dispose()
        {
            StopHeartbeat();
            
            foreach (var channel in _channels.Values)
            {
                channel.ConnectionStateChanged -= OnChannelConnectionStateChanged;
                channel.Dispose();
            }
            
            _channels.Clear();
            _channelLock?.Dispose();
        }
    }

    public class ChannelChangedEventArgs : EventArgs
    {
        public CommunicationMode NewMode { get; set; }
        public string ChannelName { get; set; }
    }

    public class ConnectionStatusEventArgs : EventArgs
    {
        public bool IsConnected { get; set; }
        public string ChannelName { get; set; }
        public string Reason { get; set; }
    }

    public class CommunicationStatistics
    {
        public string ChannelName { get; set; }
        public bool IsConnected { get; set; }
        public double LatencyMs { get; set; }
        public long BytesSent { get; set; }
        public long BytesReceived { get; set; }
    }
}
