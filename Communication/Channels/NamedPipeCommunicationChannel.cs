
// ===== Communication/Channels/NamedPipeCommunicationChannel.cs =====
using System;
using System.IO;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Tasks;
using MessagePack;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;

namespace WabiSabiRevitBridge.Communication.Channels
{
    /// <summary>
    /// Named pipe communication channel for low-latency communication
    /// </summary>
    public class NamedPipeCommunicationChannel : ICommunicationChannel
    {
        private NamedPipeClientStream _pipeClient;
        private StreamWriter _writer;
        private StreamReader _reader;
        private string _pipeName;
        private CancellationTokenSource _receiveCts;
        private Task _receiveTask;
        private readonly SemaphoreSlim _sendLock = new SemaphoreSlim(1, 1);
        private long _bytesSent;
        private long _bytesReceived;
        private DateTime _lastMessageTime = DateTime.UtcNow;
        
        public string ChannelName => "Named Pipe";
        public bool IsConnected => _pipeClient?.IsConnected ?? false;
        
        public event EventHandler<ConnectionStateChangedEventArgs> ConnectionStateChanged;
        public event EventHandler<MessageReceivedEventArgs> MessageReceived;

        public NamedPipeCommunicationChannel(string pipeName)
        {
            _pipeName = pipeName ?? throw new ArgumentNullException(nameof(pipeName));
        }

        public async Task<bool> ConnectAsync(string connectionString)
        {
            try
            {
                _pipeName = connectionString ?? _pipeName;
                
                _pipeClient = new NamedPipeClientStream(".", _pipeName, 
                    PipeDirection.InOut, PipeOptions.Asynchronous);
                
                // Try to connect with timeout
                var connectTask = _pipeClient.ConnectAsync();
                var timeoutTask = Task.Delay(5000);
                
                var completedTask = await Task.WhenAny(connectTask, timeoutTask);
                
                if (completedTask == timeoutTask)
                {
                    _pipeClient.Dispose();
                    _pipeClient = null;
                    OnConnectionStateChanged(false, "Connection timeout");
                    return false;
                }

                _writer = new StreamWriter(_pipeClient) { AutoFlush = true };
                _reader = new StreamReader(_pipeClient);
                
                // Start receive loop
                StartReceiveLoop();
                
                OnConnectionStateChanged(true, "Connected via named pipe");
                return true;
            }
            catch (Exception ex)
            {
                OnConnectionStateChanged(false, $"Connection failed: {ex.Message}");
                return false;
            }
        }

        public async Task DisconnectAsync()
        {
            StopReceiveLoop();
            
            _writer?.Dispose();
            _reader?.Dispose();
            
            if (_pipeClient?.IsConnected == true)
            {
                _pipeClient.Close();
            }
            _pipeClient?.Dispose();
            
            _writer = null;
            _reader = null;
            _pipeClient = null;
            
            OnConnectionStateChanged(false, "Disconnected");
        }

        public async Task<bool> SendDataAsync(ExportData data)
        {
            if (!IsConnected || data == null)
                return false;

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

            await _sendLock.WaitAsync();
            try
            {
                var messageBytes = MessagePackSerializer.Serialize(message);
                
                // Send message length first
                var lengthBytes = BitConverter.GetBytes(messageBytes.Length);
                await _pipeClient.WriteAsync(lengthBytes, 0, lengthBytes.Length);
                
                // Send message data
                await _pipeClient.WriteAsync(messageBytes, 0, messageBytes.Length);
                await _pipeClient.FlushAsync();
                
                Interlocked.Add(ref _bytesSent, messageBytes.Length + lengthBytes.Length);
                _lastMessageTime = DateTime.UtcNow;
                
                return true;
            }
            catch (Exception ex)
            {
                OnConnectionStateChanged(false, $"Send failed: {ex.Message}");
                return false;
            }
            finally
            {
                _sendLock.Release();
            }
        }

        public async Task<CommunicationMessage> ReceiveMessageAsync(TimeSpan timeout)
        {
            // This implementation relies on the receive loop and events
            // For synchronous receive, we'd need to implement a queue
            await Task.Delay(timeout);
            return null;
        }

        public double GetLatencyMs()
        {
            // Calculate approximate latency based on last message round-trip
            return (DateTime.UtcNow - _lastMessageTime).TotalMilliseconds;
        }

        public long GetBytesSent() => _bytesSent;
        public long GetBytesReceived() => _bytesReceived;

        private void StartReceiveLoop()
        {
            _receiveCts = new CancellationTokenSource();
            _receiveTask = Task.Run(async () =>
            {
                var lengthBuffer = new byte[4];
                
                while (!_receiveCts.Token.IsCancellationRequested && IsConnected)
                {
                    try
                    {
                        // Read message length
                        var bytesRead = await _pipeClient.ReadAsync(lengthBuffer, 0, 4, _receiveCts.Token);
                        if (bytesRead != 4)
                            continue;

                        var messageLength = BitConverter.ToInt32(lengthBuffer, 0);
                        if (messageLength <= 0 || messageLength > 10 * 1024 * 1024) // Max 10MB
                            continue;

                        // Read message data
                        var messageBuffer = new byte[messageLength];
                        var totalRead = 0;
                        
                        while (totalRead < messageLength)
                        {
                            var read = await _pipeClient.ReadAsync(
                                messageBuffer, totalRead, messageLength - totalRead, _receiveCts.Token);
                            
                            if (read == 0)
                                break;
                            
                            totalRead += read;
                        }

                        if (totalRead == messageLength)
                        {
                            var message = MessagePackSerializer.Deserialize<CommunicationMessage>(messageBuffer);
                            Interlocked.Add(ref _bytesReceived, messageLength + 4);
                            
                            MessageReceived?.Invoke(this, new MessageReceivedEventArgs { Message = message });
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
            _sendLock?.Dispose();
        }
    }
}