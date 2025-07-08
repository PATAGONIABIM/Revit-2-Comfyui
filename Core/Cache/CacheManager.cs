// ===== Core/Cache/CacheManager.cs =====
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Caching;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;
using WabiSabiRevitBridge.Core.Models;

namespace WabiSabiRevitBridge.Core.Cache
{
    /// <summary>
    /// Multi-level cache manager with L1 (memory) and L2 (disk) caching
    /// </summary>
    public class CacheManager : IDisposable
    {
        private readonly MemoryCache _l1Cache;
        private readonly DiskCache _l2Cache;
        private readonly CacheStatistics _statistics;
        private readonly ConcurrentDictionary<string, SemaphoreSlim> _keyLocks;
        private readonly Timer _cleanupTimer;
        private readonly CacheConfiguration _config;

        public CacheManager(CacheConfiguration config = null)
        {
            _config = config ?? new CacheConfiguration();
            _l1Cache = new MemoryCache("WabiSabiL1Cache");
            _l2Cache = new DiskCache(_config.DiskCachePath, _config.MaxDiskCacheSizeMB);
            _statistics = new CacheStatistics();
            _keyLocks = new ConcurrentDictionary<string, SemaphoreSlim>();
            
            // Start cleanup timer
            _cleanupTimer = new Timer(
                _ => Task.Run(() => PerformCleanupAsync()),
                null,
                TimeSpan.FromMinutes(5),
                TimeSpan.FromMinutes(5)
            );
        }

        public async Task<T> GetAsync<T>(string key, Func<Task<T>> factory, CacheOptions options = null) where T : class
        {
            options ??= CacheOptions.Default;
            _statistics.TotalRequests++;

            // Try L1 cache first
            if (_l1Cache.Get(key) is T l1Result)
            {
                _statistics.L1Hits++;
                return l1Result;
            }

            // Get or create lock for this key
            var keyLock = _keyLocks.GetOrAdd(key, _ => new SemaphoreSlim(1, 1));
            
            await keyLock.WaitAsync();
            try
            {
                // Double-check L1 cache
                if (_l1Cache.Get(key) is T l1ResultDoubleCheck)
                {
                    _statistics.L1Hits++;
                    return l1ResultDoubleCheck;
                }

                // Try L2 cache
                var l2Result = await _l2Cache.GetAsync<T>(key);
                if (l2Result != null)
                {
                    _statistics.L2Hits++;
                    
                    // Promote to L1
                    AddToL1Cache(key, l2Result, options);
                    
                    return l2Result;
                }

                // Cache miss - execute factory
                _statistics.Misses++;
                var result = await factory();
                
                if (result != null && options.CacheOnSuccess)
                {
                    // Add to both caches
                    AddToL1Cache(key, result, options);
                    await _l2Cache.SetAsync(key, result, options.L2Expiration);
                }

                return result;
            }
            finally
            {
                keyLock.Release();
                
                // Clean up lock if no longer needed
                if (keyLock.CurrentCount == 1)
                {
                    _keyLocks.TryRemove(key, out _);
                }
            }
        }

        public async Task<bool> InvalidateAsync(string key)
        {
            _l1Cache.Remove(key);
            return await _l2Cache.RemoveAsync(key);
        }

        public async Task<bool> InvalidateByPrefixAsync(string prefix)
        {
            // Remove from L1 cache
            var l1Keys = _l1Cache.Select(kvp => kvp.Key.ToString())
                .Where(k => k.StartsWith(prefix))
                .ToList();
            
            foreach (var key in l1Keys)
            {
                _l1Cache.Remove(key);
            }

            // Remove from L2 cache
            return await _l2Cache.RemoveByPrefixAsync(prefix);
        }

        public CacheStatistics GetStatistics()
        {
            _statistics.L1Size = _l1Cache.GetCount();
            _statistics.L2Size = _l2Cache.GetSize();
            _statistics.HitRate = _statistics.TotalRequests > 0
                ? (double)(_statistics.L1Hits + _statistics.L2Hits) / _statistics.TotalRequests
                : 0;
            
            return _statistics;
        }

        public async Task ClearAsync()
        {
            // Clear L1
            _l1Cache.Trim(100);
            
            // Clear L2
            await _l2Cache.ClearAsync();
            
            // Reset statistics
            _statistics.Reset();
        }

        private void AddToL1Cache<T>(string key, T value, CacheOptions options)
        {
            var policy = new CacheItemPolicy
            {
                AbsoluteExpiration = DateTimeOffset.UtcNow.Add(options.L1Expiration),
                Priority = options.Priority == CachePriority.High 
                    ? System.Runtime.Caching.CacheItemPriority.NotRemovable 
                    : System.Runtime.Caching.CacheItemPriority.Default
            };

            _l1Cache.Set(key, value, policy);
        }

        private async Task PerformCleanupAsync()
        {
            try
            {
                // Clean up L1 cache (handled automatically by MemoryCache)
                
                // Clean up L2 cache
                await _l2Cache.CleanupAsync();
                
                // Clean up unused locks
                var keysToRemove = _keyLocks
                    .Where(kvp => kvp.Value.CurrentCount == 1)
                    .Select(kvp => kvp.Key)
                    .ToList();
                
                foreach (var key in keysToRemove)
                {
                    _keyLocks.TryRemove(key, out _);
                }
            }
            catch
            {
                // Ignore cleanup errors
            }
        }

        public void Dispose()
        {
            _cleanupTimer?.Dispose();
            _l1Cache?.Dispose();
            _l2Cache?.Dispose();
            
            foreach (var keyLock in _keyLocks.Values)
            {
                keyLock?.Dispose();
            }
        }
    }

    /// <summary>
    /// Disk-based cache implementation
    /// </summary>
    public class DiskCache : IDisposable
    {
        private readonly string _basePath;
        private readonly long _maxSizeBytes;
        private readonly ConcurrentDictionary<string, CacheEntry> _index;
        private readonly string _indexPath;
        private readonly SemaphoreSlim _indexLock;

        public DiskCache(string basePath, int maxSizeMB)
        {
            _basePath = basePath ?? Path.Combine(Path.GetTempPath(), "WabiSabiCache");
            _maxSizeBytes = maxSizeMB * 1024L * 1024L;
            _indexPath = Path.Combine(_basePath, "cache.index");
            _index = new ConcurrentDictionary<string, CacheEntry>();
            _indexLock = new SemaphoreSlim(1, 1);
            
            Directory.CreateDirectory(_basePath);
            LoadIndex();
        }

        public async Task<T> GetAsync<T>(string key) where T : class
        {
            if (!_index.TryGetValue(key, out var entry))
                return null;

            // Check expiration
            if (entry.ExpiresAt < DateTime.UtcNow)
            {
                await RemoveAsync(key);
                return null;
            }

            var filePath = GetFilePath(key);
            if (!File.Exists(filePath))
            {
                _index.TryRemove(key, out _);
                return null;
            }

            try
            {
                var json = await File.ReadAllTextAsync(filePath);
                var data = JsonConvert.DeserializeObject<T>(json);
                
                // Update last accessed
                entry.LastAccessed = DateTime.UtcNow;
                entry.AccessCount++;
                
                return data;
            }
            catch
            {
                // Remove corrupted entry
                await RemoveAsync(key);
                return null;
            }
        }

        public async Task SetAsync<T>(string key, T value, TimeSpan expiration)
        {
            var json = JsonConvert.SerializeObject(value);
            var data = Encoding.UTF8.GetBytes(json);
            var filePath = GetFilePath(key);
            
            // Ensure directory exists
            Directory.CreateDirectory(Path.GetDirectoryName(filePath));
            
            // Write data
            await File.WriteAllBytesAsync(filePath, data);
            
            // Update index
            var entry = new CacheEntry
            {
                Key = key,
                Size = data.Length,
                CreatedAt = DateTime.UtcNow,
                LastAccessed = DateTime.UtcNow,
                ExpiresAt = DateTime.UtcNow.Add(expiration),
                AccessCount = 1
            };
            
            _index.AddOrUpdate(key, entry, (k, old) => entry);
            
            // Check size limit
            await EnforceSizeLimitAsync();
        }

        public async Task<bool> RemoveAsync(string key)
        {
            if (_index.TryRemove(key, out _))
            {
                var filePath = GetFilePath(key);
                try
                {
                    if (File.Exists(filePath))
                    {
                        File.Delete(filePath);
                    }
                    return true;
                }
                catch
                {
                    return false;
                }
            }
            
            return false;
        }

        public async Task<bool> RemoveByPrefixAsync(string prefix)
        {
            var keysToRemove = _index.Keys.Where(k => k.StartsWith(prefix)).ToList();
            var removed = 0;
            
            foreach (var key in keysToRemove)
            {
                if (await RemoveAsync(key))
                    removed++;
            }
            
            return removed > 0;
        }

        public long GetSize()
        {
            return _index.Values.Sum(e => e.Size);
        }

        public async Task CleanupAsync()
        {
            // Remove expired entries
            var expiredKeys = _index
                .Where(kvp => kvp.Value.ExpiresAt < DateTime.UtcNow)
                .Select(kvp => kvp.Key)
                .ToList();
            
            foreach (var key in expiredKeys)
            {
                await RemoveAsync(key);
            }
            
            // Save index
            await SaveIndexAsync();
        }

        public async Task ClearAsync()
        {
            _index.Clear();
            
            try
            {
                if (Directory.Exists(_basePath))
                {
                    Directory.Delete(_basePath, true);
                }
                Directory.CreateDirectory(_basePath);
            }
            catch
            {
                // Ignore errors
            }
        }

        private string GetFilePath(string key)
        {
            var hash = ComputeHash(key);
            var dir1 = hash.Substring(0, 2);
            var dir2 = hash.Substring(2, 2);
            return Path.Combine(_basePath, dir1, dir2, hash + ".cache");
        }

        private string ComputeHash(string input)
        {
            using (var sha256 = SHA256.Create())
            {
                var bytes = sha256.ComputeHash(Encoding.UTF8.GetBytes(input));
                return BitConverter.ToString(bytes).Replace("-", "").ToLower();
            }
        }

        private async Task EnforceSizeLimitAsync()
        {
            var currentSize = GetSize();
            
            if (currentSize > _maxSizeBytes)
            {
                // Remove least recently used entries
                var entriesToRemove = _index.Values
                    .OrderBy(e => e.LastAccessed)
                    .TakeWhile(e =>
                    {
                        currentSize -= e.Size;
                        return currentSize > _maxSizeBytes * 0.9; // Keep 90% full
                    })
                    .Select(e => e.Key)
                    .ToList();
                
                foreach (var key in entriesToRemove)
                {
                    await RemoveAsync(key);
                }
            }
        }

        private void LoadIndex()
        {
            try
            {
                if (File.Exists(_indexPath))
                {
                    var json = File.ReadAllText(_indexPath);
                    var entries = JsonConvert.DeserializeObject<List<CacheEntry>>(json);
                    
                    foreach (var entry in entries)
                    {
                        _index.TryAdd(entry.Key, entry);
                    }
                }
            }
            catch
            {
                // Ignore index load errors
            }
        }

        private async Task SaveIndexAsync()
        {
            await _indexLock.WaitAsync();
            try
            {
                var entries = _index.Values.ToList();
                var json = JsonConvert.SerializeObject(entries);
                await File.WriteAllTextAsync(_indexPath, json);
            }
            finally
            {
                _indexLock.Release();
            }
        }

        public void Dispose()
        {
            SaveIndexAsync().GetAwaiter().GetResult();
            _indexLock?.Dispose();
        }
    }

    /// <summary>
    /// Cache configuration
    /// </summary>
    public class CacheConfiguration
    {
        public string DiskCachePath { get; set; }
        public int MaxDiskCacheSizeMB { get; set; } = 1024; // 1GB default
        public TimeSpan DefaultL1Expiration { get; set; } = TimeSpan.FromMinutes(5);
        public TimeSpan DefaultL2Expiration { get; set; } = TimeSpan.FromHours(24);
    }

    /// <summary>
    /// Cache options for individual operations
    /// </summary>
    public class CacheOptions
    {
        public TimeSpan L1Expiration { get; set; } = TimeSpan.FromMinutes(5);
        public TimeSpan L2Expiration { get; set; } = TimeSpan.FromHours(24);
        public CachePriority Priority { get; set; } = CachePriority.Normal;
        public bool CacheOnSuccess { get; set; } = true;
        
        public static CacheOptions Default => new CacheOptions();
        
        public static CacheOptions ShortTerm => new CacheOptions
        {
            L1Expiration = TimeSpan.FromMinutes(1),
            L2Expiration = TimeSpan.FromMinutes(10)
        };
        
        public static CacheOptions LongTerm => new CacheOptions
        {
            L1Expiration = TimeSpan.FromHours(1),
            L2Expiration = TimeSpan.FromDays(7),
            Priority = CachePriority.High
        };
    }

    public enum CachePriority
    {
        Low,
        Normal,
        High
    }

    /// <summary>
    /// Cache statistics
    /// </summary>
    public class CacheStatistics
    {
        public long TotalRequests { get; set; }
        public long L1Hits { get; set; }
        public long L2Hits { get; set; }
        public long Misses { get; set; }
        public long L1Size { get; set; }
        public long L2Size { get; set; }
        public double HitRate { get; set; }
        
        public void Reset()
        {
            TotalRequests = 0;
            L1Hits = 0;
            L2Hits = 0;
            Misses = 0;
        }
    }

    /// <summary>
    /// Cache entry metadata
    /// </summary>
    public class CacheEntry
    {
        public string Key { get; set; }
        public long Size { get; set; }
        public DateTime CreatedAt { get; set; }
        public DateTime LastAccessed { get; set; }
        public DateTime ExpiresAt { get; set; }
        public int AccessCount { get; set; }
    }
}
