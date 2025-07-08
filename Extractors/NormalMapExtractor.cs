// ===== Extractors/NormalMapExtractor.cs =====
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
    /// Extracts normal maps from 3D geometry
    /// </summary>
    public class NormalMapExtractor : BaseExtractor
    {
        public override string ExtractorName => "Normal Map Extractor";
        public override ExportType ExportType => ExportType.Normal;

        public enum NormalSpace
        {
            World,
            View,
            Tangent
        }

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
                    // Get normal space from configuration
                    var normalSpace = GetNormalSpace();
                    
                    // Get view transform
                    var viewTransform = RevitHelper.GetViewTransform(view);
                    
                    // Generate normal map
                    var normalMap = GenerateNormalMap(
                        view,
                        options.Width,
                        options.Height,
                        normalSpace,
                        viewTransform,
                        options.ParallelProcessing,
                        cancellationToken
                    );

                    // Convert to image
                    var imageData = ConvertNormalMapToImage(normalMap, options.Width, options.Height);

                    var exportData = new ExportData
                    {
                        ViewName = view.Name,
                        Type = ExportType.Normal,
                        ImageData = imageData,
                        Format = ImageFormat.PNG,
                        Width = options.Width,
                        Height = options.Height,
                        BitDepth = 24, // RGB
                        Metadata = new Dictionary<string, object>
                        {
                            ["NormalSpace"] = normalSpace.ToString(),
                            ["ViewTransform"] = SerializeTransform(viewTransform)
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
                            ["NormalSpace"] = normalSpace.ToString(),
                            ["RaysCast"] = options.Width * options.Height
                        }
                    };
                }
                catch (Exception ex)
                {
                    return new ExtractorResult
                    {
                        Success = false,
                        ErrorMessage = $"Normal map extraction failed: {ex.Message}",
                        ProcessingTimeMs = stopwatch.ElapsedMilliseconds
                    };
                }
            }, cancellationToken);
        }

        private NormalSpace GetNormalSpace()
        {
            if (_configuration?.Settings.TryGetValue("NormalSpace", out var spaceObj) == true)
            {
                if (Enum.TryParse<NormalSpace>(spaceObj.ToString(), out var space))
                {
                    return space;
                }
            }
            
            return NormalSpace.View; // Default to view space
        }

        private Vector3[,] GenerateNormalMap(
            View3D view,
            int width,
            int height,
            NormalSpace normalSpace,
            Transform viewTransform,
            bool parallel,
            CancellationToken cancellationToken)
        {
            var normalMap = new Vector3[height, width];
            var doc = view.Document;
            
            // Get view parameters
            var origin = view.Origin;
            var forward = view.ViewDirection.Normalize();
            var up = view.UpDirection.Normalize();
            var right = forward.CrossProduct(up).Normalize();
            
            // Calculate FOV
            var fov = Math.PI / 3; // 60 degrees default
            var aspectRatio = (double)width / height;
            var halfFovY = fov * 0.5;
            var halfFovX = Math.Atan(Math.Tan(halfFovY) * aspectRatio);

            if (parallel)
            {
                // Process in tiles
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
                    MaxDegreeOfParallelism = Environment.ProcessorCount,
                    CancellationToken = cancellationToken
                }, 
                tile =>
                {
                    ProcessNormalTile(doc, view, normalMap, tile, origin, forward, up, right,
                        halfFovX, halfFovY, normalSpace, viewTransform, width, height);
                });
            }
            else
            {
                var tile = new Rectangle(0, 0, width, height);
                ProcessNormalTile(doc, view, normalMap, tile, origin, forward, up, right,
                    halfFovX, halfFovY, normalSpace, viewTransform, width, height);
            }

            return normalMap;
        }

        private void ProcessNormalTile(
            Document doc,
            View3D view,
            Vector3[,] normalMap,
            Rectangle tile,
            XYZ origin,
            XYZ forward,
            XYZ up,
            XYZ right,
            double halfFovX,
            double halfFovY,
            NormalSpace normalSpace,
            Transform viewTransform,
            int imageWidth,
            int imageHeight)
        {
            var intersector = new ReferenceIntersector(view);
            intersector.FindReferencesInRevitLinks = true;

            for (int y = tile.Y; y < tile.Y + tile.Height; y++)
            {
                for (int x = tile.X; x < tile.X + tile.Width; x++)
                {
                    // Calculate ray direction
                    var screenX = (2.0 * x / imageWidth - 1.0);
                    var screenY = -(2.0 * y / imageHeight - 1.0);
                    
                    var rayX = Math.Tan(halfFovX) * screenX;
                    var rayY = Math.Tan(halfFovY) * screenY;
                    
                    var rayDir = (forward + right * rayX + up * rayY).Normalize();
                    
                    // Cast ray
                    var result = intersector.FindNearest(origin, rayDir);
                    
                    Vector3 normal;
                    if (result != null && result.Proximity > 0)
                    {
                        // Get face normal
                        var face = GetFaceFromReference(doc, result.GetReference());
                        if (face != null)
                        {
                            var faceNormal = face.ComputeNormal(new UV(0.5, 0.5));
                            
                            // Transform normal based on space
                            normal = TransformNormal(faceNormal, normalSpace, viewTransform);
                        }
                        else
                        {
                            // Default normal pointing towards camera
                            normal = new Vector3(0, 0, 1);
                        }
                    }
                    else
                    {
                        // No hit - use default normal
                        normal = new Vector3(0, 0, 1);
                    }
                    
                    normalMap[y, x] = normal;
                }
            }
        }

        private Face GetFaceFromReference(Document doc, Reference reference)
        {
            if (reference == null)
                return null;

            try
            {
                var element = doc.GetElement(reference);
                if (element == null)
                    return null;

                var geomObject = element.GetGeometryObjectFromReference(reference);
                return geomObject as Face;
            }
            catch
            {
                return null;
            }
        }

        private Vector3 TransformNormal(XYZ normal, NormalSpace space, Transform viewTransform)
        {
            var normalized = normal.Normalize();
            
            switch (space)
            {
                case NormalSpace.World:
                    // Keep in world space
                    return new Vector3(
                        (float)normalized.X,
                        (float)normalized.Y,
                        (float)normalized.Z
                    );
                    
                case NormalSpace.View:
                    // Transform to view space
                    var viewNormal = viewTransform.OfVector(normalized);
                    return new Vector3(
                        (float)viewNormal.X,
                        (float)viewNormal.Y,
                        (float)viewNormal.Z
                    );
                    
                case NormalSpace.Tangent:
                    // For tangent space, we'd need UV coordinates and tangent basis
                    // For now, use view space as approximation
                    var tangentNormal = viewTransform.OfVector(normalized);
                    return new Vector3(
                        (float)tangentNormal.X,
                        (float)tangentNormal.Y,
                        (float)tangentNormal.Z
                    );
                    
                default:
                    return new Vector3(0, 0, 1);
            }
        }

        private byte[] ConvertNormalMapToImage(Vector3[,] normalMap, int width, int height)
        {
            using (var bitmap = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            {
                var bitmapData = bitmap.LockBits(
                    new Rectangle(0, 0, width, height),
                    ImageLockMode.WriteOnly,
                    PixelFormat.Format24bppRgb
                );

                unsafe
                {
                    byte* ptr = (byte*)bitmapData.Scan0;
                    
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            var normal = normalMap[y, x];
                            
                            // Convert from [-1, 1] to [0, 255]
                            var r = (byte)((normal.X * 0.5f + 0.5f) * 255);
                            var g = (byte)((normal.Y * 0.5f + 0.5f) * 255);
                            var b = (byte)((normal.Z * 0.5f + 0.5f) * 255);
                            
                            *ptr++ = b; // BGR format
                            *ptr++ = g;
                            *ptr++ = r;
                        }
                        
                        // Handle stride
                        ptr += bitmapData.Stride - (width * 3);
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

        private struct Vector3
        {
            public float X { get; set; }
            public float Y { get; set; }
            public float Z { get; set; }

            public Vector3(float x, float y, float z)
            {
                X = x;
                Y = y;
                Z = z;
            }
        }
    }
}
