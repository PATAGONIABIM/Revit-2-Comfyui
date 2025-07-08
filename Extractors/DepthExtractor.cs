// ===== Extractors/DepthExtractor.cs =====
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;
using WabiSabiRevitBridge.Extractors.Base;
using WabiSabiRevitBridge.Utils;

namespace WabiSabiRevitBridge.Extractors
{
    /// <summary>
    /// Extracts depth maps from Revit views using ray casting
    /// </summary>
    public class DepthExtractor : BaseExtractor
    {
        public override string ExtractorName => "Depth Map Extractor";
        public override ExportType ExportType => ExportType.Depth;

        private readonly object _raycastLock = new object();
        private readonly int _maxParallelism = Environment.ProcessorCount;

        protected override async Task<ExtractorResult> PerformExtractionAsync(
            View3D view, 
            ExtractorOptions options, 
            CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                var doc = view.Document;
                var stopwatch = System.Diagnostics.Stopwatch.StartNew();
                
                try
                {
                    // Get view parameters
                    var viewTransform = RevitHelper.GetViewTransform(view);
                    var bbox = RevitHelper.GetViewBoundingBox(view);
                    
                    if (bbox == null)
                    {
                        throw new InvalidOperationException("Unable to determine view bounding box");
                    }

                    // Calculate depth range
                    var (nearPlane, farPlane) = CalculateDepthRange(view, bbox);
                    
                    // Generate depth map
                    var depthData = GenerateDepthMap(
                        view, 
                        options.Width, 
                        options.Height, 
                        nearPlane, 
                        farPlane,
                        options.ParallelProcessing,
                        cancellationToken
                    );

                    // Convert to image format
                    byte[] imageData;
                    if (options.Quality > 90) // High precision mode
                    {
                        imageData = ConvertToEXR(depthData, options.Width, options.Height);
                    }
                    else
                    {
                        imageData = ConvertToPNG(depthData, options.Width, options.Height);
                    }

                    var exportData = new ExportData
                    {
                        ViewName = view.Name,
                        Type = ExportType.Depth,
                        ImageData = imageData,
                        Format = options.Quality > 90 ? ImageFormat.EXR : ImageFormat.PNG,
                        Width = options.Width,
                        Height = options.Height,
                        BitDepth = options.Quality > 90 ? 32 : 16,
                        Metadata = new Dictionary<string, object>
                        {
                            ["NearPlane"] = nearPlane,
                            ["FarPlane"] = farPlane,
                            ["ViewTransform"] = SerializeTransform(viewTransform),
                            ["BoundingBox"] = SerializeBoundingBox(bbox)
                        }
                    };

                    stopwatch.Stop();

                    return new ExtractorResult
                    {
                        Success = true,
                        Data = exportData,
                        ProcessingTimeMs = stopwatch.ElapsedMilliseconds,
                        Diagnostics = new Dictionary<string, object>
                        {
                            ["RaysCast"] = options.Width * options.Height,
                            ["DepthRange"] = $"{nearPlane:F2} - {farPlane:F2}",
                            ["Parallelism"] = options.ParallelProcessing ? _maxParallelism : 1
                        }
                    };
                }
                catch (Exception ex)
                {
                    return new ExtractorResult
                    {
                        Success = false,
                        ErrorMessage = $"Depth extraction failed: {ex.Message}",
                        ProcessingTimeMs = stopwatch.ElapsedMilliseconds
                    };
                }
            }, cancellationToken);
        }

        private (double near, double far) CalculateDepthRange(View3D view, BoundingBoxXYZ bbox)
        {
            var viewDir = view.ViewDirection.Normalize();
            var origin = view.Origin;
            
            // Project bounding box corners onto view direction
            var corners = new[]
            {
                bbox.Min,
                bbox.Max,
                new XYZ(bbox.Min.X, bbox.Min.Y, bbox.Max.Z),
                new XYZ(bbox.Min.X, bbox.Max.Y, bbox.Min.Z),
                new XYZ(bbox.Max.X, bbox.Min.Y, bbox.Min.Z),
                new XYZ(bbox.Min.X, bbox.Max.Y, bbox.Max.Z),
                new XYZ(bbox.Max.X, bbox.Min.Y, bbox.Max.Z),
                new XYZ(bbox.Max.X, bbox.Max.Y, bbox.Min.Z)
            };

            double minDist = double.MaxValue;
            double maxDist = double.MinValue;

            foreach (var corner in corners)
            {
                var toCorner = corner - origin;
                var distance = toCorner.DotProduct(viewDir);
                minDist = Math.Min(minDist, distance);
                maxDist = Math.Max(maxDist, distance);
            }

            // Add some padding
            var range = maxDist - minDist;
            minDist -= range * 0.1;
            maxDist += range * 0.1;

            return (Math.Max(0.1, minDist), maxDist);
        }

        private float[,] GenerateDepthMap(
            View3D view, 
            int width, 
            int height, 
            double nearPlane, 
            double farPlane,
            bool parallel,
            CancellationToken cancellationToken)
        {
            var depthMap = new float[height, width];
            var doc = view.Document;
            
            // Get view parameters
            var origin = view.Origin;
            var forward = view.ViewDirection.Normalize();
            var up = view.UpDirection.Normalize();
            var right = forward.CrossProduct(up).Normalize();
            
            // Calculate FOV from view
            var fov = GetFieldOfView(view);
            var aspectRatio = (double)width / height;
            
            // Calculate ray directions for each pixel
            var halfFovY = fov * 0.5;
            var halfFovX = Math.Atan(Math.Tan(halfFovY) * aspectRatio);

            if (parallel)
            {
                // Parallel processing with tiles
                var tileSize = 64;
                var tiles = new List<Rectangle>();
                
                for (int y = 0; y < height; y += tileSize)
                {
                    for (int x = 0; x < width; x += tileSize)
                    {
                        tiles.Add(new Rectangle(
                            x, y,
                            Math.Min(tileSize, width - x),
                            Math.Min(tileSize, height - y)
                        ));
                    }
                }

                Parallel.ForEach(tiles, new ParallelOptions 
                { 
                    MaxDegreeOfParallelism = _maxParallelism,
                    CancellationToken = cancellationToken
                }, 
                tile =>
                {
                    ProcessTile(doc, view, depthMap, tile, origin, forward, up, right, 
                        halfFovX, halfFovY, nearPlane, farPlane, width, height);
                });
            }
            else
            {
                // Sequential processing
                var tile = new Rectangle(0, 0, width, height);
                ProcessTile(doc, view, depthMap, tile, origin, forward, up, right, 
                    halfFovX, halfFovY, nearPlane, farPlane, width, height);
            }

            return depthMap;
        }

        private void ProcessTile(
            Document doc,
            View3D view,
            float[,] depthMap,
            Rectangle tile,
            XYZ origin,
            XYZ forward,
            XYZ up,
            XYZ right,
            double halfFovX,
            double halfFovY,
            double nearPlane,
            double farPlane,
            int imageWidth,
            int imageHeight)
        {
            // Create reference intersector for this thread
            var intersector = new ReferenceIntersector(view);
            intersector.FindReferencesInRevitLinks = true;
            
            var filter = new ElementClassFilter(typeof(Element));
            intersector.SetFilter(filter);

            for (int y = tile.Y; y < tile.Y + tile.Height; y++)
            {
                for (int x = tile.X; x < tile.X + tile.Width; x++)
                {
                    // Calculate normalized screen coordinates (-1 to 1)
                    var screenX = (2.0 * x / imageWidth - 1.0);
                    var screenY = -(2.0 * y / imageHeight - 1.0); // Flip Y
                    
                    // Calculate ray direction
                    var rayX = Math.Tan(halfFovX) * screenX;
                    var rayY = Math.Tan(halfFovY) * screenY;
                    
                    var rayDir = (forward + right * rayX + up * rayY).Normalize();
                    
                    // Cast ray
                    var result = intersector.FindNearest(origin, rayDir);
                    
                    float depth;
                    if (result != null && result.Proximity > 0)
                    {
                        // Normalize depth to 0-1 range
                        depth = (float)((result.Proximity - nearPlane) / (farPlane - nearPlane));
                        depth = Math.Max(0, Math.Min(1, depth));
                    }
                    else
                    {
                        // No hit, use far plane
                        depth = 1.0f;
                    }
                    
                    lock (_raycastLock)
                    {
                        depthMap[y, x] = depth;
                    }
                }
            }
        }

        private double GetFieldOfView(View3D view)
        {
            // Try to get FOV from view parameters
            var param = view.get_Parameter(BuiltInParameter.VIEWER_BOUND_FAR_CLIPPING);
            if (param != null)
            {
                // Estimate FOV based on view properties
                // This is an approximation - Revit doesn't directly expose FOV
                return Math.PI / 3; // 60 degrees default
            }
            
            return Math.PI / 3; // 60 degrees default
        }

        private byte[] ConvertToPNG(float[,] depthData, int width, int height)
        {
            using (var bitmap = new Bitmap(width, height, PixelFormat.Format16bppGrayScale))
            {
                var bitmapData = bitmap.LockBits(
                    new Rectangle(0, 0, width, height),
                    ImageLockMode.WriteOnly,
                    PixelFormat.Format16bppGrayScale
                );

                unsafe
                {
                    ushort* ptr = (ushort*)bitmapData.Scan0;
                    
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            var depth = depthData[y, x];
                            var value = (ushort)(depth * ushort.MaxValue);
                            *ptr++ = value;
                        }
                    }
                }

                bitmap.UnlockBits(bitmapData);

                using (var stream = new MemoryStream())
                {
                    bitmap.Save(stream, System.Drawing.Imaging.ImageFormat.Png);
                    return stream.ToArray();
                }
            }
        }

        private byte[] ConvertToEXR(float[,] depthData, int width, int height)
        {
            // For now, convert to 16-bit PNG
            // In production, use OpenEXR library for true 32-bit float support
            return ConvertToPNG(depthData, width, height);
        }

        private Dictionary<string, object> SerializeTransform(Transform transform)
        {
            return new Dictionary<string, object>
            {
                ["Origin"] = new[] { transform.Origin.X, transform.Origin.Y, transform.Origin.Z },
                ["BasisX"] = new[] { transform.BasisX.X, transform.BasisX.Y, transform.BasisX.Z },
                ["BasisY"] = new[] { transform.BasisY.X, transform.BasisY.Y, transform.BasisY.Z },
                ["BasisZ"] = new[] { transform.BasisZ.X, transform.BasisZ.Y, transform.BasisZ.Z }
            };
        }

        private Dictionary<string, object> SerializeBoundingBox(BoundingBoxXYZ bbox)
        {
            return new Dictionary<string, object>
            {
                ["Min"] = new[] { bbox.Min.X, bbox.Min.Y, bbox.Min.Z },
                ["Max"] = new[] { bbox.Max.X, bbox.Max.Y, bbox.Max.Z }
            };
        }
    }
}
