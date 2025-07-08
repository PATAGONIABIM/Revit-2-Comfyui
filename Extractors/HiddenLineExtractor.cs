
// ===== Extractors/HiddenLineExtractor.cs =====
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;
using WabiSabiRevitBridge.Extractors.Base;

namespace WabiSabiRevitBridge.Extractors
{
    /// <summary>
    /// Extracts hidden line views from Revit
    /// </summary>
    public class HiddenLineExtractor : BaseExtractor
    {
        public override string ExtractorName => "Hidden Line Extractor";
        public override ExportType ExportType => ExportType.HiddenLine;

        private readonly Dictionary<string, DisplayStyle> _styleCache = new Dictionary<string, DisplayStyle>();

        protected override async Task<ExtractorResult> PerformExtractionAsync(
            View3D view, 
            ExtractorOptions options, 
            CancellationToken cancellationToken)
        {
            return await Task.Run(() =>
            {
                var doc = view.Document;
                DisplayStyle originalStyle = view.DisplayStyle;
                
                try
                {
                    // Create a temporary view for export
                    View3D exportView = null;
                    Transaction trans = null;
                    
                    try
                    {
                        // Start transaction to modify view
                        trans = new Transaction(doc, "WabiSabi Export Hidden Lines");
                        trans.Start();

                        // Duplicate the view to avoid modifying the original
                        var viewId = view.Duplicate(ViewDuplicateOption.Duplicate);
                        exportView = doc.GetElement(viewId) as View3D;
                        
                        if (exportView == null)
                        {
                            throw new InvalidOperationException("Failed to duplicate view");
                        }

                        // Configure view for hidden line export
                        ConfigureViewForExport(exportView, options);
                        
                        trans.Commit();

                        // Export the image
                        var imageData = ExportViewImage(exportView, options, cancellationToken);
                        
                        if (imageData == null || imageData.Length == 0)
                        {
                            throw new InvalidOperationException("Failed to export view image");
                        }

                        // Create export data
                        var exportData = new ExportData
                        {
                            ViewName = view.Name,
                            Type = ExportType.HiddenLine,
                            ImageData = imageData,
                            Format = DetermineImageFormat(options),
                            Width = options.Width,
                            Height = options.Height,
                            BitDepth = 8,
                            Metadata = new Dictionary<string, object>
                            {
                                ["OriginalViewId"] = view.Id.IntegerValue,
                                ["DisplayStyle"] = originalStyle.ToString(),
                                ["ExportTimestamp"] = DateTime.UtcNow
                            }
                        };

                        return new ExtractorResult
                        {
                            Success = true,
                            Data = exportData,
                            Diagnostics = new Dictionary<string, object>
                            {
                                ["ImageSize"] = imageData.Length,
                                ["CompressionRatio"] = CalculateCompressionRatio(options, imageData.Length)
                            }
                        };
                    }
                    finally
                    {
                        // Clean up temporary view
                        if (exportView != null && trans != null)
                        {
                            using (var deleteTransaction = new Transaction(doc, "Delete temporary view"))
                            {
                                deleteTransaction.Start();
                                doc.Delete(exportView.Id);
                                deleteTransaction.Commit();
                            }
                        }
                        trans?.Dispose();
                    }
                }
                finally
                {
                    // Ensure original view style is restored
                    if (view.DisplayStyle != originalStyle)
                    {
                        using (var restoreTransaction = new Transaction(doc, "Restore view style"))
                        {
                            restoreTransaction.Start();
                            view.DisplayStyle = originalStyle;
                            restoreTransaction.Commit();
                        }
                    }
                }
            }, cancellationToken);
        }

        private void ConfigureViewForExport(View3D view, ExtractorOptions options)
        {
            // Set display style to hidden line
            view.DisplayStyle = DisplayStyle.HiddenLine;
            
            // Configure view detail level
            view.DetailLevel = ViewDetailLevel.Fine;
            
            // Disable unnecessary elements for cleaner export
            view.AreAnnotationCategoriesHidden = true;
            view.AreAnalyticalModelCategoriesHidden = true;
            view.AreImportCategoriesHidden = false;
            
            // Configure graphics options
            var graphicsOptions = view.GetRenderingSettings();
            graphicsOptions.ShowEdges = true;
            graphicsOptions.ShowSilhouettes = true;
            
            // Apply anti-aliasing if available
            ApplyAntiAliasing(view, options);
        }

        private void ApplyAntiAliasing(View3D view, ExtractorOptions options)
        {
            try
            {
                // Attempt to set anti-aliasing through view's graphics options
                // This is version-dependent and may not be available in all Revit versions
                var param = view.get_Parameter(BuiltInParameter.VIEW_GRAPHICS_MODEL_EDGES_MODE);
                if (param != null && !param.IsReadOnly)
                {
                    param.Set(1); // Enable smooth edges
                }
            }
            catch
            {
                // Anti-aliasing not available in this version
            }
        }

        private byte[] ExportViewImage(View3D view, ExtractorOptions options, CancellationToken cancellationToken)
        {
            var exportOptions = new ImageExportOptions
            {
                FilePath = Path.GetTempFileName(),
                FitDirection = FitDirectionType.Horizontal,
                HLRandWFViewsFileType = ImageFileType.PNG,
                ImageResolution = ImageResolution.DPI_300,
                PixelSize = Math.Max(options.Width, options.Height),
                ExportRange = ExportRange.SetOfViews
            };

            exportOptions.SetViewsAndSheets(new List<ElementId> { view.Id });

            // Perform export
            view.Document.ExportImage(exportOptions);

            // Read and process the exported image
            var tempFile = exportOptions.FilePath + ".png";
            
            try
            {
                if (!File.Exists(tempFile))
                {
                    throw new FileNotFoundException("Export failed - image file not created");
                }

                // Load and resize image if needed
                using (var originalBitmap = new Bitmap(tempFile))
                {
                    Bitmap processedBitmap;
                    
                    // Resize if needed
                    if (originalBitmap.Width != options.Width || originalBitmap.Height != options.Height)
                    {
                        processedBitmap = ResizeBitmap(originalBitmap, options.Width, options.Height);
                    }
                    else
                    {
                        processedBitmap = new Bitmap(originalBitmap);
                    }

                    using (processedBitmap)
                    {
                        // Apply any post-processing
                        if (options.ParallelProcessing)
                        {
                            ApplyParallelPostProcessing(processedBitmap, options);
                        }

                        // Convert to bytes
                        var format = DetermineImageFormat(options);
                        return ConvertBitmapToBytes(processedBitmap, format, options.Quality);
                    }
                }
            }
            finally
            {
                // Clean up temp file
                if (File.Exists(tempFile))
                {
                    try { File.Delete(tempFile); } catch { }
                }
            }
        }

        private Bitmap ResizeBitmap(Bitmap source, int width, int height)
        {
            var destRect = new Rectangle(0, 0, width, height);
            var destBitmap = new Bitmap(width, height);

            destBitmap.SetResolution(source.HorizontalResolution, source.VerticalResolution);

            using (var graphics = Graphics.FromImage(destBitmap))
            {
                graphics.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                graphics.CompositingQuality = System.Drawing.Drawing2D.CompositingQuality.HighQuality;
                graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                graphics.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.HighQuality;
                graphics.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.HighQuality;

                using (var wrapMode = new System.Drawing.Imaging.ImageAttributes())
                {
                    wrapMode.SetWrapMode(System.Drawing.Drawing2D.WrapMode.TileFlipXY);
                    graphics.DrawImage(source, destRect, 0, 0, source.Width, source.Height, 
                        GraphicsUnit.Pixel, wrapMode);
                }
            }

            return destBitmap;
        }

        private void ApplyParallelPostProcessing(Bitmap bitmap, ExtractorOptions options)
        {
            // Implement parallel processing for edge enhancement or other filters
            // This is a placeholder for more advanced processing
        }

        private System.Drawing.Imaging.ImageFormat DetermineImageFormat(ExtractorOptions options)
        {
            // For now, always use PNG for hidden line drawings
            // Could be extended to support other formats based on configuration
            return System.Drawing.Imaging.ImageFormat.Png;
        }

        private double CalculateCompressionRatio(ExtractorOptions options, int compressedSize)
        {
            int uncompressedSize = options.Width * options.Height * 3; // RGB
            return (double)uncompressedSize / compressedSize;
        }
    }
}