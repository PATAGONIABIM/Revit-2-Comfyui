
// ===== Extractors/MetadataExtractor.cs =====
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using Autodesk.Revit.DB.Analysis;
using Newtonsoft.Json;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Models;
using WabiSabiRevitBridge.Extractors.Base;
using WabiSabiRevitBridge.Utils;

namespace WabiSabiRevitBridge.Extractors
{
    /// <summary>
    /// Extracts structured metadata from the Revit model
    /// </summary>
    public class MetadataExtractor : BaseExtractor
    {
        public override string ExtractorName => "Metadata Extractor";
        public override ExportType ExportType => ExportType.Metadata;

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
                    var metadata = new Dictionary<string, object>();
                    
                    // Extract view information
                    metadata["view"] = ExtractViewMetadata(view);
                    
                    // Extract camera information
                    metadata["camera"] = ExtractCameraMetadata(view);
                    
                    // Extract lighting information
                    metadata["lighting"] = ExtractLightingMetadata(doc, view);
                    
                    // Extract materials
                    metadata["materials"] = ExtractMaterialsMetadata(doc, view);
                    
                    // Extract room information if available
                    metadata["rooms"] = ExtractRoomMetadata(doc, view);
                    
                    // Extract element statistics
                    metadata["statistics"] = ExtractStatistics(doc, view);
                    
                    // Extract project information
                    metadata["project"] = ExtractProjectMetadata(doc);
                    
                    // Serialize metadata
                    var jsonMetadata = JsonConvert.SerializeObject(metadata, Formatting.Indented);
                    var metadataBytes = Encoding.UTF8.GetBytes(jsonMetadata);

                    var exportData = new ExportData
                    {
                        ViewName = view.Name,
                        Type = ExportType.Metadata,
                        ImageData = metadataBytes, // Using ImageData field for metadata JSON
                        Format = ImageFormat.Raw,
                        Width = 0,
                        Height = 0,
                        BitDepth = 0,
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
                            ["MetadataSize"] = metadataBytes.Length,
                            ["Keys"] = metadata.Keys.ToList()
                        }
                    };
                }
                catch (Exception ex)
                {
                    return new ExtractorResult
                    {
                        Success = false,
                        ErrorMessage = $"Metadata extraction failed: {ex.Message}",
                        ProcessingTimeMs = stopwatch.ElapsedMilliseconds
                    };
                }
            }, cancellationToken);
        }

        private Dictionary<string, object> ExtractViewMetadata(View3D view)
        {
            var viewData = new Dictionary<string, object>
            {
                ["name"] = view.Name,
                ["id"] = view.Id.IntegerValue,
                ["displayStyle"] = view.DisplayStyle.ToString(),
                ["detailLevel"] = view.DetailLevel.ToString(),
                ["scale"] = view.Scale,
                ["cropBoxActive"] = view.CropBoxActive,
                ["sectionBoxActive"] = view.IsSectionBoxActive
            };

            // Get view orientation
            var orientation = view.GetOrientation();
            viewData["orientation"] = new Dictionary<string, object>
            {
                ["eyePosition"] = SerializeXYZ(orientation.EyePosition),
                ["upDirection"] = SerializeXYZ(orientation.UpDirection),
                ["forwardDirection"] = SerializeXYZ(orientation.ForwardDirection)
            };

            return viewData;
        }

        private Dictionary<string, object> ExtractCameraMetadata(View3D view)
        {
            var camera = new Dictionary<string, object>();
            
            // Camera position and orientation
            camera["position"] = SerializeXYZ(view.Origin);
            camera["forward"] = SerializeXYZ(view.ViewDirection);
            camera["up"] = SerializeXYZ(view.UpDirection);
            camera["right"] = SerializeXYZ(view.RightDirection);
            
            // Field of view (approximation)
            camera["fov"] = 60.0; // Default, as Revit doesn't directly expose FOV
            
            // Projection type
            var isPerspective = view.IsPerspective;
            camera["projection"] = isPerspective ? "perspective" : "orthographic";
            
            if (!isPerspective)
            {
                // For orthographic views, include view scale
                camera["scale"] = view.Scale;
            }
            
            // Clipping planes
            var param = view.get_Parameter(BuiltInParameter.VIEWER_BOUND_FAR_CLIPPING);
            if (param != null)
            {
                camera["farClip"] = param.AsDouble();
            }
            
            return camera;
        }

        private Dictionary<string, object> ExtractLightingMetadata(Document doc, View3D view)
        {
            var lighting = new Dictionary<string, object>();
            
            // Sun settings
            var sunSettings = view.SunAndShadowSettings;
            if (sunSettings != null)
            {
                lighting["sunEnabled"] = sunSettings.SunAndShadowType != SunAndShadowType.StillImage;
                
                if (sunSettings.SunAndShadowType == SunAndShadowType.OneDayStudy)
                {
                    lighting["dateTime"] = sunSettings.GetFrameTime().ToDateTimeString();
                }
                
                // Get sun direction
                var sunDirection = CalculateSunDirection(doc, sunSettings);
                if (sunDirection != null)
                {
                    lighting["sunDirection"] = SerializeXYZ(sunDirection);
                }
            }
            
            // Rendering settings
            var renderSettings = view.GetRenderingSettings();
            lighting["useSilhouettes"] = renderSettings.UseSilhouetteEdges;
            lighting["showShadows"] = renderSettings.ShowShadows;
            
            return lighting;
        }

        private XYZ CalculateSunDirection(Document doc, SunAndShadowSettings sunSettings)
        {
            try
            {
                // This is a simplified calculation
                // In production, use proper sun path calculations
                var projectLocation = doc.ActiveProjectLocation;
                var siteLocation = projectLocation.GetSiteLocation();
                
                if (siteLocation != null)
                {
                    var latitude = siteLocation.Latitude * (180.0 / Math.PI);
                    var longitude = siteLocation.Longitude * (180.0 / Math.PI);
                    
                    // Simple approximation - should use proper solar calculations
                    var altitude = 45.0 * (Math.PI / 180.0); // 45 degrees elevation
                    var azimuth = 180.0 * (Math.PI / 180.0); // South
                    
                    var x = Math.Cos(altitude) * Math.Sin(azimuth);
                    var y = Math.Cos(altitude) * Math.Cos(azimuth);
                    var z = Math.Sin(altitude);
                    
                    return new XYZ(x, y, z);
                }
            }
            catch
            {
                // Ignore calculation errors
            }
            
            return null;
        }

        private List<Dictionary<string, object>> ExtractMaterialsMetadata(Document doc, View3D view)
        {
            var materials = new List<Dictionary<string, object>>();
            var materialIds = new HashSet<ElementId>();
            
            // Collect all materials used in view
            var collector = new FilteredElementCollector(doc, view.Id)
                .WhereElementIsNotElementType();
            
            foreach (Element elem in collector)
            {
                var matIds = elem.GetMaterialIds(false);
                foreach (var matId in matIds)
                {
                    materialIds.Add(matId);
                }
            }
            
            // Extract material information
            foreach (var matId in materialIds)
            {
                var material = doc.GetElement(matId) as Material;
                if (material != null)
                {
                    var matData = new Dictionary<string, object>
                    {
                        ["id"] = matId.IntegerValue,
                        ["name"] = material.Name,
                        ["color"] = SerializeColor(material.Color),
                        ["transparency"] = material.Transparency,
                        ["shininess"] = material.Shininess,
                        ["smoothness"] = material.Smoothness
                    };
                    
                    // Material class for prompting
                    var materialClass = GetMaterialClass(material);
                    if (!string.IsNullOrEmpty(materialClass))
                    {
                        matData["class"] = materialClass;
                    }
                    
                    materials.Add(matData);
                }
            }
            
            return materials;
        }

        private string GetMaterialClass(Material material)
        {
            var name = material.Name.ToLower();
            
            if (name.Contains("concrete") || name.Contains("cement"))
                return "concrete";
            if (name.Contains("glass") || name.Contains("glazing"))
                return "glass";
            if (name.Contains("steel") || name.Contains("metal") || name.Contains("aluminum"))
                return "metal";
            if (name.Contains("wood") || name.Contains("timber"))
                return "wood";
            if (name.Contains("brick") || name.Contains("masonry"))
                return "masonry";
            if (name.Contains("gypsum") || name.Contains("plaster") || name.Contains("drywall"))
                return "plaster";
            if (name.Contains("fabric") || name.Contains("carpet"))
                return "fabric";
            
            return "generic";
        }

        private List<Dictionary<string, object>> ExtractRoomMetadata(Document doc, View3D view)
        {
            var rooms = new List<Dictionary<string, object>>();
            
            // Get rooms visible in view
            var roomCollector = new FilteredElementCollector(doc, view.Id)
                .OfClass(typeof(SpatialElement))
                .Cast<SpatialElement>()
                .Where(r => r is Room);
            
            foreach (Room room in roomCollector)
            {
                var roomData = new Dictionary<string, object>
                {
                    ["id"] = room.Id.IntegerValue,
                    ["name"] = room.Name,
                    ["number"] = room.Number,
                    ["area"] = room.Area,
                    ["volume"] = room.Volume,
                    ["perimeter"] = room.Perimeter
                };
                
                // Get room type/function
                var department = room.get_Parameter(BuiltInParameter.ROOM_DEPARTMENT);
                if (department != null)
                {
                    roomData["department"] = department.AsString();
                }
                
                rooms.Add(roomData);
            }
            
            return rooms;
        }

        private Dictionary<string, object> ExtractStatistics(Document doc, View3D view)
        {
            var stats = new Dictionary<string, object>();
            
            // Count elements by category
            var categoryCounts = new Dictionary<string, int>();
            var collector = new FilteredElementCollector(doc, view.Id)
                .WhereElementIsNotElementType();
            
            foreach (Element elem in collector)
            {
                var category = elem.Category;
                if (category != null)
                {
                    var catName = category.Name;
                    if (!categoryCounts.ContainsKey(catName))
                        categoryCounts[catName] = 0;
                    categoryCounts[catName]++;
                }
            }
            
            stats["elementCounts"] = categoryCounts;
            stats["totalElements"] = categoryCounts.Values.Sum();
            stats["uniqueCategories"] = categoryCounts.Count;
            
            return stats;
        }

        private Dictionary<string, object> ExtractProjectMetadata(Document doc)
        {
            var project = new Dictionary<string, object>();
            
            var projectInfo = doc.ProjectInformation;
            if (projectInfo != null)
            {
                project["name"] = projectInfo.Name;
                project["number"] = projectInfo.Number;
                project["address"] = projectInfo.Address;
                project["author"] = projectInfo.Author;
                project["status"] = projectInfo.Status;
                
                // Get custom parameters
                var clientName = projectInfo.get_Parameter(BuiltInParameter.CLIENT_NAME);
                if (clientName != null)
                {
                    project["client"] = clientName.AsString();
                }
            }
            
            // Get project location
            var location = doc.ActiveProjectLocation;
            if (location != null)
            {
                var siteLocation = location.GetSiteLocation();
                if (siteLocation != null)
                {
                    project["location"] = new Dictionary<string, object>
                    {
                        ["latitude"] = siteLocation.Latitude * (180.0 / Math.PI),
                        ["longitude"] = siteLocation.Longitude * (180.0 / Math.PI),
                        ["timeZone"] = siteLocation.TimeZone
                    };
                }
            }
            
            return project;
        }

        private double[] SerializeXYZ(XYZ point)
        {
            return new[] { point.X, point.Y, point.Z };
        }

        private string SerializeColor(Color color)
        {
            if (color != null && color.IsValid)
            {
                return $"#{color.Red:X2}{color.Green:X2}{color.Blue:X2}";
            }
            return "#808080"; // Default gray
        }
    }
}