// ===== Extractors/Base/BaseExtractor.cs =====
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;
using WabiSabiRevitBridge.Utils;

namespace WabiSabiRevitBridge.Extractors.Base
{
    /// <summary>
    /// Base class for all data extractors
    /// </summary>
    public abstract class BaseExtractor : IDataExtractor
    {
        protected readonly PerformanceMonitor _performanceMonitor;
        protected ExtractorConfiguration _configuration;
        protected readonly object _cacheLock = new object();
        protected Dictionary<string, CachedResult> _cache = new Dictionary<string, CachedResult>();
        
        public abstract string ExtractorName { get; }
        public abstract ExportType ExportType { get; }
        public bool IsEnabled { get; set; } = true;

        protected BaseExtractor()
        {
            _performanceMonitor = new PerformanceMonitor();
            _configuration = new ExtractorConfiguration();
        }

        public virtual async Task<ExtractorResult> ExtractAsync(
            View3D view, 
            ExtractorOptions options, 
            CancellationToken cancellationToken = default)
        {
            if (!IsEnabled)
            {
                return new ExtractorResult
                {
                    Success = false,
                    ErrorMessage = $"{ExtractorName} is disabled"
                };
            }

            var stopwatch = Stopwatch.StartNew();
            
            try
            {
                // Check cache if enabled
                if (options.UseCache)
                {
                    var cacheKey = GenerateCacheKey(view, options);
                    var cachedResult = GetFromCache(cacheKey);
                    if (cachedResult != null)
                    {
                        return new ExtractorResult
                        {
                            Success = true,
                            Data = cachedResult.Data,
                            ProcessingTimeMs = 0,
                            Diagnostics = new Dictionary<string, object> 
                            { 
                                ["CacheHit"] = true,
                                ["CacheKey"] = cacheKey
                            }
                        };
                    }
                }

                // Validate view
                if (!CanExtract(view))
                {
                    return new ExtractorResult
                    {
                        Success = false,
                        ErrorMessage = $"View '{view.Name}' is not compatible with {ExtractorName}"
                    };
                }

                // Perform extraction
                var result = await PerformExtractionAsync(view, options, cancellationToken);
                
                if (result.Success && options.UseCache)
                {
                    var cacheKey = GenerateCacheKey(view, options);
                    AddToCache(cacheKey, result);
                }

                result.ProcessingTimeMs = stopwatch.ElapsedMilliseconds;
                
                // Add performance metrics
                result.Diagnostics["ExtractorName"] = ExtractorName;
                result.Diagnostics["ViewName"] = view.Name;
                result.Diagnostics["Resolution"] = $"{options.Width}x{options.Height}";
                
                return result;
            }
            catch (OperationCanceledException)
            {
                return new ExtractorResult
                {
                    Success = false,
                    ErrorMessage = "Operation was cancelled",
                    ProcessingTimeMs = stopwatch.ElapsedMilliseconds
                };
            }
            catch (Exception ex)
            {
                return new ExtractorResult
                {
                    Success = false,
                    ErrorMessage = $"Error in {ExtractorName}: {ex.Message}",
                    ProcessingTimeMs = stopwatch.ElapsedMilliseconds,
                    Diagnostics = new Dictionary<string, object> { ["Exception"] = ex.ToString() }
                };
            }
        }

        protected abstract Task<ExtractorResult> PerformExtractionAsync(
            View3D view, 
            ExtractorOptions options, 
            CancellationToken cancellationToken);

        public virtual bool CanExtract(View3D view)
        {
            return view != null && view.IsValidObject && !view.IsTemplate;
        }

        public virtual void Configure(ExtractorConfiguration config)
        {
            _configuration = config ?? throw new ArgumentNullException(nameof(config));
        }

        protected virtual string GenerateCacheKey(View3D view, ExtractorOptions options)
        {
            // Generate a unique key based on view state and options
            var viewState = GetViewState(view);
            return $"{ExtractorName}_{viewState}_{options.Width}x{options.Height}_{options.Quality}";
        }

        protected virtual string GetViewState(View3D view)
        {
            // Create a hash of the view's current state
            var origin = view.Origin;
            var forward = view.ViewDirection;
            var up = view.UpDirection;
            var scale = view.Scale;
            
            return $"{origin.X:F2},{origin.Y:F2},{origin.Z:F2}_" +
                   $"{forward.X:F2},{forward.Y:F2},{forward.Z:F2}_" +
                   $"{up.X:F2},{up.Y:F2},{up.Z:F2}_" +
                   $"{scale}";
        }

        protected CachedResult GetFromCache(string key)
        {
            lock (_cacheLock)
            {
                if (_cache.TryGetValue(key, out var cached))
                {
                    // Check if cache is still valid (e.g., not too old)
                    if (DateTime.UtcNow - cached.Timestamp < TimeSpan.FromMinutes(5))
                    {
                        return cached;
                    }
                    else
                    {
                        _cache.Remove(key);
                    }
                }
            }
            return null;
        }

        protected void AddToCache(string key, ExtractorResult result)
        {
            lock (_cacheLock)
            {
                // Implement simple LRU cache with max size
                if (_cache.Count > 100)
                {
                    // Remove oldest entry
                    var oldestKey = string.Empty;
                    var oldestTime = DateTime.MaxValue;
                    
                    foreach (var kvp in _cache)
                    {
                        if (kvp.Value.Timestamp < oldestTime)
                        {
                            oldestTime = kvp.Value.Timestamp;
                            oldestKey = kvp.Key;
                        }
                    }
                    
                    if (!string.IsNullOrEmpty(oldestKey))
                        _cache.Remove(oldestKey);
                }

                _cache[key] = new CachedResult
                {
                    Data = result.Data,
                    Timestamp = DateTime.UtcNow
                };
            }
        }

        protected byte[] ConvertBitmapToBytes(Bitmap bitmap, ImageFormat format, int quality = 90)
        {
            using (var stream = new MemoryStream())
            {
                if (format == ImageFormat.Jpeg)
                {
                    var encoderParameters = new EncoderParameters(1);
                    encoderParameters.Param[0] = new EncoderParameter(
                        System.Drawing.Imaging.Encoder.Quality, quality);
                    
                    var jpegCodec = GetEncoderInfo("image/jpeg");
                    bitmap.Save(stream, jpegCodec, encoderParameters);
                }
                else
                {
                    bitmap.Save(stream, format);
                }
                
                return stream.ToArray();
            }
        }

        private ImageCodecInfo GetEncoderInfo(string mimeType)
        {
            var codecs = ImageCodecInfo.GetImageEncoders();
            foreach (var codec in codecs)
            {
                if (codec.MimeType == mimeType)
                    return codec;
            }
            return null;
        }

        protected class CachedResult
        {
            public ExportData Data { get; set; }
            public DateTime Timestamp { get; set; }
        }
    }
}
