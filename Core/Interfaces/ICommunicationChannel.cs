// ===== Core/Interfaces/ICommunicationChannel.cs =====
using System;
using System.Threading.Tasks;

namespace WabiSabiRevitBridge.Core.Interfaces
{
    /// <summary>
    /// Interfaz para canales de comunicación con ComfyUI
    /// </summary>
    public interface ICommunicationChannel : IDisposable
    {
        string ChannelName { get; }
        bool IsConnected { get; }
        event EventHandler<ConnectionStateChangedEventArgs> ConnectionStateChanged;
        event EventHandler<MessageReceivedEventArgs> MessageReceived;
        
        Task<bool> ConnectAsync(string connectionString);
        Task DisconnectAsync();
        Task<bool> SendDataAsync(ExportData data);
        Task<bool> SendMessageAsync(CommunicationMessage message);
        Task<CommunicationMessage> ReceiveMessageAsync(TimeSpan timeout);
        
        // Performance metrics
        double GetLatencyMs();
        long GetBytesSent();
        long GetBytesReceived();
    }

    public class ConnectionStateChangedEventArgs : EventArgs
    {
        public bool IsConnected { get; set; }
        public string Reason { get; set; }
    }

    public class MessageReceivedEventArgs : EventArgs
    {
        public CommunicationMessage Message { get; set; }
    }
}

