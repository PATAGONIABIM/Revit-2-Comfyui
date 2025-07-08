
// ===== Extractors/SegmentationExtractor.cs =====
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
    /// Extracts segmentation maps with unique colors per element/category
    /// </summary>
    public class SegmentationExtractor : BaseExtractor
    {
        public override string ExtractorName => "Segmentation Map Extractor";
        public override ExportType ExportType => ExportType.Segmentation;

        private readonly Dictionary<ElementId, Color> _elementColorCache = new Dictionary<ElementId, Color>();
        private readonly object _colorLock = new object();

        public enum SegmentationMode
        {
            ByElement,
            ByCategory,
            ByMaterial,
            ByPhase,
            ByParameter
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
                    // Get segmentation mode from configuration
                    var mode = GetSegmentationMode();
                    
                    // Create temporary view for segmentation
                    View3D segmentationView = null;
                    Transaction trans = null;
                    
                    try
                    {
                        trans = new Transaction(doc, "WabiSabi Segmentation Export");
                        trans.Start();

                        // Duplicate view
                        var viewId = view.Duplicate(ViewDuplicateOption.Duplicate);
                        segmentationView = doc.GetElement(viewId) as View3D;
                        
                        if (segmentationView == null)
                        {
                            throw new InvalidOperationException("Failed to duplicate view");
                        }

                        // Configure view for segmentation
                        ConfigureSegmentationView(segmentationView, mode);
                        
                        // Apply segmentation colors
                        var colorMap = ApplySegmentationColors(segmentationView, mode);
                        
                        trans.Commit();

                        // Export segmentation image
                        var segmentationImage = ExportSegmentationImage(
                            segmentationView, 
                            options.Width, 
                            options.Height,
                            cancellationToken
                        );

                        // Create ID map (optional secondary output)
                        var idMapData = GenerateIdMap(
                            view,
                            options.Width,
                            options.Height,
                            colorMap,
                            cancellationToken
                        );

                        // Prepare metadata
                        var metadata = new Dictionary<string, object>
                        {
                            ["SegmentationMode"] = mode.ToString(),
                            ["ColorMap"] = SerializeColorMap(colorMap),
                            ["ElementCount"] = colorMap.Count,
                            ["Categories"] = GetCategoryNames(view, colorMap)
                        };

                        var exportData = new ExportData
                        {
                            ViewName = view.Name,
                            Type = ExportType.Segmentation,
                            ImageData = segmentationImage,
                            Format = ImageFormat.PNG,
                            Width = options.Width,
                            Height = options.Height,
                            BitDepth = 24, // RGB
                            Metadata = metadata
                        };

                        stopwatch.Stop();

                        return new ExtractorResult
                        {
                            Success = true,
                            Data = exportData,
                            ProcessingTimeMs = stopwatch.ElapsedMilliseconds,
                            Diagnostics = new Dictionary<string, object>
                            {
                                ["UniqueSegments"] = colorMap.Count,
                                ["Mode"] = mode.ToString()
                            }
                        };
                    }
                    finally
                    {
                        // Clean up temporary view
                        if (segmentationView != null && trans != null)
                        {
                            using (var deleteTransaction = new Transaction(doc, "Delete temp segmentation view"))
                            {
                                deleteTransaction.Start();
                                doc.Delete(segmentationView.Id);
                                deleteTransaction.Commit();
                            }
                        }
                        trans?.Dispose();
                    }
                }
                catch (Exception ex)
                {
                    return new ExtractorResult
                    {
                        Success = false,
                        ErrorMessage = $"Segmentation extraction failed: {ex.Message}",
                        ProcessingTimeMs = stopwatch.ElapsedMilliseconds
                    };
                }
            }, cancellationToken);
        }

        private SegmentationMode GetSegmentationMode()
        {
            // Get from configuration
            if (_configuration?.Settings.TryGetValue("SegmentationMode", out var modeObj) == true)
            {
                if (Enum.TryParse<SegmentationMode>(modeObj.ToString(), out var mode))
                {
                    return mode;
                }
            }
            
            return SegmentationMode.ByCategory; // Default
        }

        private void ConfigureSegmentationView(View3D view, SegmentationMode mode)
        {
            // Set to shaded mode for color display
            view.DisplayStyle = DisplayStyle.Shaded;
            
            // Disable shadows and effects
            var renderSettings = view.GetRenderingSettings();
            renderSettings.ShowShadows = false;
            renderSettings.ShowSilhouettes = false;
            renderSettings.ShowEdges = false;
            
            // Hide annotations
            view.AreAnnotationCategoriesHidden = true;
            
            // Set detail level to medium for performance
            view.DetailLevel = ViewDetailLevel.Medium;
        }

        private Dictionary<object, System.Drawing.Color> ApplySegmentationColors(
            View3D view, 
            SegmentationMode mode)
        {
            var colorMap = new Dictionary<object, System.Drawing.Color>();
            var colorIndex = 0;
            var doc = view.Document;
            
            // Collect elements
            var collector = new FilteredElementCollector(doc, view.Id)
                .WhereElementIsNotElementType();

            // Group elements based on mode
            var groups = GroupElements(collector, mode, doc);
            
            // Generate unique colors
            var colors = GenerateUniqueColors(groups.Count);
            
            // Apply colors to view
            var overrideSettings = new OverrideGraphicSettings();
            
            foreach (var group in groups)
            {
                var color = colors[colorIndex++];
                var revitColor = new Autodesk.Revit.DB.Color(color.R, color.G, color.B);
                
                overrideSettings.SetProjectionFillColor(revitColor);
                overrideSettings.SetProjectionLineColor(revitColor);
                overrideSettings.SetCutFillColor(revitColor);
                overrideSettings.SetCutLineColor(revitColor);
                
                // Apply to all elements in group
                foreach (var element in group.Value)
                {
                    view.SetElementOverrides(element.Id, overrideSettings);
                    
                    lock (_colorLock)
                    {
                        _elementColorCache[element.Id] = color;
                    }
                }
                
                colorMap[group.Key] = color;
            }
            
            return colorMap;
        }

        private Dictionary<object, List<Element>> GroupElements(
            FilteredElementCollector collector, 
            SegmentationMode mode,
            Document doc)
        {
            var groups = new Dictionary<object, List<Element>>();
            
            foreach (Element elem in collector)
            {
                object key = mode switch
                {
                    SegmentationMode.ByElement => elem.Id,
                    SegmentationMode.ByCategory => elem.Category?.Id ?? ElementId.InvalidElementId,
                    SegmentationMode.ByMaterial => GetPrimaryMaterial(elem, doc),
                    SegmentationMode.ByPhase => elem.CreatedPhaseId,
                    SegmentationMode.ByParameter => GetParameterValue(elem, "Type"),
                    _ => elem.Category?.Id ?? ElementId.InvalidElementId
                };
                
                if (!groups.ContainsKey(key))
                {
                    groups[key] = new List<Element>();
                }
                
                groups[key].Add(elem);
            }
            
            return groups;
        }

        private ElementId GetPrimaryMaterial(Element element, Document doc)
        {
            var materialIds = element.GetMaterialIds(false);
            return materialIds.FirstOrDefault() ?? ElementId.InvalidElementId;
        }

        private string GetParameterValue(Element element, string parameterName)
        {
            var param = element.LookupParameter(parameterName);
            return param?.AsString() ?? "Unknown";
        }

        private List<System.Drawing.Color> GenerateUniqueColors(int count)
        {
            var colors = new List<System.Drawing.Color>();
            var hueStep = 360.0 / count;
            
            for (int i = 0; i < count; i++)
            {
                var hue = i * hueStep;
                var color = ColorFromHSV(hue, 0.8, 0.9);
                colors.Add(color);
            }
            
            // Shuffle to avoid similar colors being adjacent
            var random = new Random(42); // Fixed seed for consistency
            return colors.OrderBy(x => random.Next()).ToList();
        }

        private System.Drawing.Color ColorFromHSV(double hue, double saturation, double value)
        {
            int hi = Convert.ToInt32(Math.Floor(hue / 60)) % 6;
            double f = hue / 60 - Math.Floor(hue / 60);

            value = value * 255;
            int v = Convert.ToInt32(value);
            int p = Convert.ToInt32(value * (1 - saturation));
            int q = Convert.ToInt32(value * (1 - f * saturation));
            int t = Convert.ToInt32(value * (1 - (1 - f) * saturation));

            return hi switch
            {
                0 => System.Drawing.Color.FromArgb(255, v, t, p),
                1 => System.Drawing.Color.FromArgb(255, q, v, p),
                2 => System.Drawing.Color.FromArgb(255, p, v, t),
                3 => System.Drawing.Color.FromArgb(255, p, q, v),
                4 => System.Drawing.Color.FromArgb(255, t, p, v),
                _ => System.Drawing.Color.FromArgb(255, v, p, q),
            };
        }

        private byte[] ExportSegmentationImage(
            View3D view, 
            int width, 
            int height,
            CancellationToken cancellationToken)
        {
            var exportOptions = new ImageExportOptions
            {
                FilePath = Path.GetTempFileName(),
                FitDirection = FitDirectionType.Horizontal,
                ImageResolution = ImageResolution.DPI_300,
                PixelSize = Math.Max(width, height),
                ExportRange = ExportRange.SetOfViews
            };

            exportOptions.SetViewsAndSheets(new List<ElementId> { view.Id });

            // Export the view
            view.Document.ExportImage(exportOptions);

            var tempFile = exportOptions.FilePath + ".png";
            
            try
            {
                if (!File.Exists(tempFile))
                {
                    throw new FileNotFoundException("Segmentation export failed");
                }

                // Load and process image
                using (var bitmap = new Bitmap(tempFile))
                {
                    // Resize if needed
                    if (bitmap.Width != width || bitmap.Height != height)
                    {
                        using (var resized = new Bitmap(width, height))
                        using (var graphics = Graphics.FromImage(resized))
                        {
                            graphics.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.NearestNeighbor;
                            graphics.DrawImage(bitmap, 0, 0, width, height);
                            
                            using (var stream = new MemoryStream())
                            {
                                resized.Save(stream, System.Drawing.Imaging.ImageFormat.Png);
                                return stream.ToArray();
                            }
                        }
                    }
                    else
                    {
                        using (var stream = new MemoryStream())
                        {
                            bitmap.Save(stream, System.Drawing.Imaging.ImageFormat.Png);
                            return stream.ToArray();
                        }
                    }
                }
            }
            finally
            {
                if (File.Exists(tempFile))
                {
                    try { File.Delete(tempFile); } catch { }
                }
            }
        }

        private byte[] GenerateIdMap(
            View3D view,
            int width,
            int height,
            Dictionary<object, System.Drawing.Color> colorMap,
            CancellationToken cancellationToken)
        {
            // This would generate a secondary image where each pixel contains
            // the element ID encoded as color
            // For now, return empty array
            return new byte[0];
        }

        private Dictionary<string, object> SerializeColorMap(Dictionary<object, System.Drawing.Color> colorMap)
        {
            var serialized = new Dictionary<string, object>();
            
            foreach (var kvp in colorMap)
            {
                var key = kvp.Key.ToString();
                var color = kvp.Value;
                serialized[key] = $"#{color.R:X2}{color.G:X2}{color.B:X2}";
            }
            
            return serialized;
        }

        private List<string> GetCategoryNames(View3D view, Dictionary<object, System.Drawing.Color> colorMap)
        {
            var categories = new HashSet<string>();
            var doc = view.Document;
            
            foreach (var key in colorMap.Keys)
            {
                if (key is ElementId catId)
                {
                    var category = doc.GetElement(catId) as Category;
                    if (category != null)
                    {
                        categories.Add(category.Name);
                    }
                }
            }
            
            return categories.ToList();
        }
    }
}