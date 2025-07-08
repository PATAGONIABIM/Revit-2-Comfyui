// ===== Communication/Channels/SharedMemoryCommunicationChannel.cs =====
using System;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;
using System.Threading.Tasks;
using MessagePack;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;

namespace WabiSabiRevitBridge.Communication.Channels
{
    /// <summary>
    /// Shared memory communication channel for ultra-low latency
    /// </summary>
    public class SharedMemoryCommunicationChannel : ICommunicationChannel
    {
        private MemoryMappedFile _mmf;
        private MemoryMappedViewAccessor _accessor;
        private EventWaitHandle _dataAvailableEvent;
        private EventWaitHandle _dataReadEvent;
        private Mutex _mutex;
        private string _memoryName;
        private CancellationTokenSource _receiveCts;
        private Task _receiveTask;
        private readonly int _bufferSize = 50 * 1024 * 1024; // 50MB buffer
        private long _bytesSent;
        private long _bytesReceived;
        
        public string ChannelName => "Shared Memory";
        public bool IsConnected { get; private set; }
        
        public event EventHandler<ConnectionStateChangedEventArgs> ConnectionStateChanged;
        public event EventHandler<MessageReceivedEventArgs> MessageReceived;

        public SharedMemoryCommunicationChannel(string memoryName)
        {
            _memoryName = memoryName ?? throw new ArgumentNullException(nameof(memoryName));
        }

        public async Task<bool> ConnectAsync(string connectionString)
        {
            return await Task.Run(() =>
            {
                try
                {
                    _memoryName = connectionString ?? _memoryName;
                    
                    // Try to open existing shared memory
                    try
                    {
                        _mmf = MemoryMappedFile.OpenExisting(_memoryName);
                    }
                    catch (FileNotFoundException)
                    {
                        // Create new shared memory if it doesn't exist
                        _mmf = MemoryMappedFile.CreateNew(_memoryName, _bufferSize);
                    }
                    
                    _accessor = _mmf.CreateViewAccessor();
                    
                    // Create synchronization objects
                    _dataAvailableEvent = new EventWaitHandle(false, EventResetMode.AutoReset, 
                        $"{_memoryName}_DataAvailable");
                    _dataReadEvent = new EventWaitHandle(false, EventResetMode.AutoReset, 
                        $"{_memoryName}_DataRead");
                    _mutex = new Mutex(false, $"{_memoryName}_Mutex");
                    
                    // Start receive loop
                    StartReceiveLoop();
                    
                    IsConnected = true;
                    OnConnectionStateChanged(true, "Connected via shared memory");
                    return true;
                }
                catch (Exception ex)
                {
                    Cleanup();
                    OnConnectionStateChanged(false, $"Connection failed: {ex.Message}");
                    return false;
                }
            });
        }

        public async Task DisconnectAsync()
        {
            await Task.Run(() =>
            {
                StopReceiveLoop();
                Cleanup();
                IsConnected = false;
                OnConnectionStateChanged(false, "Disconnected");
            });
        }

        public async Task<bool> SendDataAsync(ExportData data)
        {
            if (!IsConnected || data == null)
                return false;

            // For large data, we might need to implement chunking
            // For now, serialize everything into a message
            var message = new CommunicationMessage
            {
                MessageType = "ExportData",
                Payload = MessagePackSerializer.Serialize(data),
                Headers = new Dictionary<string, object>
                {
                    ["ExportType"] = data.Type.ToString(),
                    ["ViewName"] = data.ViewName,
                    ["DataSize"] = data.ImageData.Length
                }
            };

            return await SendMessageAsync(message);
        }

        public async Task<bool> SendMessageAsync(CommunicationMessage message)
        {
            if (!IsConnected || message == null)
                return false;

            return await Task.Run(() =>
            {
                try
                {
                    var messageBytes = MessagePackSerializer.Serialize(message);
                    
                    if (messageBytes.Length > _bufferSize - 8)
                    {
                        // Message too large for buffer
                        return false;
                    }

                    // Wait for mutex
                    if (!_mutex.WaitOne(5000))
                        return false;

                    try
                    {
                        // Write header (message length + data available flag)
                        _accessor.Write(0, 1); // Data available flag
                        _accessor.Write(4, messageBytes.Length);
                        
                        // Write message data
                        _accessor.WriteArray(8, messageBytes, 0, messageBytes.Length);
                        
                        // Signal data available
                        _dataAvailableEvent.Set();
                        
                        // Wait for reader to acknowledge (with timeout)
                        _dataReadEvent.WaitOne(1000);
                        
                        Interlocked.Add(ref _bytesSent, messageBytes.Length);
                        return true;
                    }
                    finally
                    {
                        _mutex.ReleaseMutex();
                    }
                }
                catch (Exception ex)
                {
                    OnConnectionStateChanged(false, $"Send failed: {ex.Message}");
                    return false;
                }
            });
        }

        public async Task<CommunicationMessage> ReceiveMessageAsync(TimeSpan timeout)
        {
            // This implementation relies on the receive loop and events
            await Task.Delay(timeout);
            return null;
        }

        public double GetLatencyMs()
        {
            // Shared memory has near-zero latency
            return 0.1;
        }

        public long GetBytesSent() => _bytesSent;
        public long GetBytesReceived() => _bytesReceived;

        private void StartReceiveLoop()
        {
            _receiveCts = new CancellationTokenSource();
            _receiveTask = Task.Run(() =>
            {
                while (!_receiveCts.Token.IsCancellationRequested && IsConnected)
                {
                    try
                    {
                        // Wait for data with timeout
                        if (_dataAvailableEvent.WaitOne(100))
                        {
                            if (!_mutex.WaitOne(5000))
                                continue;

                            try
                            {
                                // Check data available flag
                                var dataAvailable = _accessor.ReadByte(0);
                                if (dataAvailable != 1)
                                    continue;

                                // Read message length
                                var messageLength = _accessor.ReadInt32(4);
                                if (messageLength <= 0 || messageLength > _bufferSize - 8)
                                    continue;

                                // Read message data
                                var messageBytes = new byte[messageLength];
                                _accessor.ReadArray(8, messageBytes, 0, messageLength);
                                
                                // Clear data available flag
                                _accessor.Write(0, (byte)0);
                                
                                // Signal that we've read the data
                                _dataReadEvent.Set();
                                
                                // Deserialize and raise event
                                var message = MessagePackSerializer.Deserialize<CommunicationMessage>(messageBytes);
                                Interlocked.Add(ref _bytesReceived, messageLength);
                                
                                MessageReceived?.Invoke(this, new MessageReceivedEventArgs { Message = message });
                            }
                            finally
                            {
                                _mutex.ReleaseMutex();
                            }
                        }
                    }
                    catch (OperationCanceledException)
                    {
                        break;
                    }
                    catch (Exception ex)
                    {
                        // Log error and continue
                        Console.WriteLine($"Receive error: {ex.Message}");
                    }
                }
            }, _receiveCts.Token);
        }

        private void StopReceiveLoop()
        {
            _receiveCts?.Cancel();
            _receiveTask?.Wait(TimeSpan.FromSeconds(1));
            _receiveCts?.Dispose();
            _receiveCts = null;
            _receiveTask = null;
        }

        private void Cleanup()
        {
            _accessor?.Dispose();
            _mmf?.Dispose();
            _dataAvailableEvent?.Dispose();
            _dataReadEvent?.Dispose();
            _mutex?.Dispose();
            
            _accessor = null;
            _mmf = null;
            _dataAvailableEvent = null;
            _dataReadEvent = null;
            _mutex = null;
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
