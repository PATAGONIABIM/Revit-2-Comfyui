// ===== Core/Models/CommunicationMessage.cs =====
using System;
using MessagePack;

namespace WabiSabiRevitBridge.Core.Models
{
    [MessagePackObject]
    public class CommunicationMessage
    {
        [Key(0)]
        public string MessageType { get; set; }
        
        [Key(1)]
        public Guid MessageId { get; set; } = Guid.NewGuid();
        
        [Key(2)]
        public DateTime Timestamp { get; set; } = DateTime.UtcNow;
        
        [Key(3)]
        public byte[] Payload { get; set; }
        
        [Key(4)]
        public Dictionary<string, object> Headers { get; set; } = new Dictionary<string, object>();
        
        [Key(5)]
        public CompressionType Compression { get; set; } = CompressionType.None;
    }

    public enum CompressionType
    {
        None,
        LZ4,
        ZSTD,
        GZip
    }
}


