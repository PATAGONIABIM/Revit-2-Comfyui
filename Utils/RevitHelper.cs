// ===== Utils/RevitHelper.cs =====
using System;
using System.Collections.Generic;
using System.Linq;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;

namespace WabiSabiRevitBridge.Utils
{
    /// <summary>
    /// Helper methods for Revit-specific operations
    /// </summary>
    public static class RevitHelper
    {
        public static List<View3D> GetAll3DViews(Document doc)
        {
            var collector = new FilteredElementCollector(doc)
                .OfClass(typeof(View3D))
                .Cast<View3D>()
                .Where(v => !v.IsTemplate && v.CanBePrinted)
                .ToList();
            
            return collector;
        }

        public static BoundingBoxXYZ GetViewBoundingBox(View3D view)
        {
            var bbox = view.GetSectionBox();
            if (!bbox.Enabled)
            {
                // Get bounding box of all visible elements
                var collector = new FilteredElementCollector(view.Document, view.Id)
                    .WhereElementIsNotElementType();
                
                BoundingBoxXYZ combinedBox = null;
                
                foreach (Element elem in collector)
                {
                    var elemBox = elem.get_BoundingBox(view);
                    if (elemBox != null)
                    {
                        if (combinedBox == null)
                        {
                            combinedBox = elemBox;
                        }
                        else
                        {
                            combinedBox.Min = new XYZ(
                                Math.Min(combinedBox.Min.X, elemBox.Min.X),
                                Math.Min(combinedBox.Min.Y, elemBox.Min.Y),
                                Math.Min(combinedBox.Min.Z, elemBox.Min.Z)
                            );
                            combinedBox.Max = new XYZ(
                                Math.Max(combinedBox.Max.X, elemBox.Max.X),
                                Math.Max(combinedBox.Max.Y, elemBox.Max.Y),
                                Math.Max(combinedBox.Max.Z, elemBox.Max.Z)
                            );
                        }
                    }
                }
                
                return combinedBox;
            }
            
            return bbox;
        }

        public static Dictionary<ElementId, Color> GetElementColors(View3D view)
        {
            var colors = new Dictionary<ElementId, Color>();
            var doc = view.Document;
            
            var collector = new FilteredElementCollector(doc, view.Id)
                .WhereElementIsNotElementType();
            
            foreach (Element elem in collector)
            {
                var overrideGraphics = view.GetElementOverrides(elem.Id);
                Color color = null;
                
                // Check for override color
                if (overrideGraphics.ProjectionLineColor.IsValid)
                {
                    color = overrideGraphics.ProjectionLineColor;
                }
                else
                {
                    // Get material color
                    var materialId = elem.GetMaterialIds(false).FirstOrDefault();
                    if (materialId != null)
                    {
                        var material = doc.GetElement(materialId) as Material;
                        if (material != null)
                        {
                            color = material.Color;
                        }
                    }
                }
                
                if (color != null && color.IsValid)
                {
                    colors[elem.Id] = color;
                }
            }
            
            return colors;
        }

        public static List<Category> GetVisibleCategories(View3D view)
        {
            var categories = new List<Category>();
            var doc = view.Document;
            
            foreach (Category cat in doc.Settings.Categories)
            {
                if (cat.get_Visible(view))
                {
                    categories.Add(cat);
                }
            }
            
            return categories;
        }

        public static Transform GetViewTransform(View3D view)
        {
            var origin = view.Origin;
            var forward = view.ViewDirection;
            var up = view.UpDirection;
            var right = forward.CrossProduct(up);
            
            var transform = Transform.Identity;
            transform.Origin = origin;
            transform.BasisX = right.Normalize();
            transform.BasisY = up.Normalize();
            transform.BasisZ = forward.Normalize();
            
            return transform;
        }

        public static bool IsElementVisible(Element element, View3D view)
        {
            if (element == null || view == null)
                return false;
            
            // Check if element is hidden in view
            if (element.IsHidden(view))
                return false;
            
            // Check category visibility
            var category = element.Category;
            if (category != null && !category.get_Visible(view))
                return false;
            
            // Check if element has geometry in view
            var options = new Options
            {
                View = view,
                ComputeReferences = false
            };
            
            var geom = element.get_Geometry(options);
            return geom != null && geom.Any();
        }
    }
}