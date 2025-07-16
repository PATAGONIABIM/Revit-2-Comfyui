// DepthExtractorFast.cs - Extractor de profundidad optimizado con submuestreo
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Drawing = System.Drawing;

namespace WabiSabiBridge.Extractors
{
    public class DepthExtractorFast
    {
        private readonly UIApplication _uiApp;
        private readonly int _resolution;
        private readonly int _subsampleFactor;
        
        public DepthExtractorFast(UIApplication uiApp, int resolution = 512, int subsampleFactor = 4)
        {
            _uiApp = uiApp;
            _resolution = resolution;
            _subsampleFactor = subsampleFactor;
        }
        
        // <-- CAMBIO: La firma del método ahora acepta las esquinas de la vista.
        public void ExtractDepthMap(View3D view3D, string outputPath, string timestamp, int width, int height, IList<XYZ> viewCorners)
        {
            Document doc = _uiApp.ActiveUIDocument.Document;
            
            var viewOrientation = view3D.GetOrientation();
            var eyePosition = viewOrientation.EyePosition;
            var forwardDirection = viewOrientation.ForwardDirection.Normalize();
            var upDirection = viewOrientation.UpDirection.Normalize();
            var rightDirection = forwardDirection.CrossProduct(upDirection).Normalize();
            
            var (minDepth, maxDepth) = CalculateDepthRange(doc, view3D, eyePosition, forwardDirection);
            
            int sampleWidth = width / _subsampleFactor;
            int sampleHeight = height / _subsampleFactor;
            
            double[,] depthSamples = new double[sampleHeight, sampleWidth];
            
            ICollection<ElementId> elementIds = GetIntersectableElementIds(doc, view3D);
            ReferenceIntersector intersector = new ReferenceIntersector(elementIds, FindReferenceTarget.Element, view3D);

            // <-- CAMBIO: Reconstruir las cuatro esquinas del plano de la vista para una interpolación precisa.
            XYZ bottomLeft = viewCorners[0];
            XYZ topRight = viewCorners[1];
            XYZ viewX_vec = topRight - bottomLeft;
            XYZ viewWidthVector = viewX_vec.DotProduct(rightDirection) * rightDirection;
            XYZ viewHeightVector = viewX_vec.DotProduct(upDirection) * upDirection;
            XYZ bottomRight = bottomLeft + viewWidthVector;
            XYZ topLeft = bottomLeft + viewHeightVector;
            
            // Fase 1: Submuestreo rápido
            for (int sy = 0; sy < sampleHeight; sy++)
            {
                for (int sx = 0; sx < sampleWidth; sx++)
                {
                    int x = sx * _subsampleFactor + _subsampleFactor / 2;
                    int y = sy * _subsampleFactor + _subsampleFactor / 2;
                    
                    // <-- CAMBIO: Calcular la dirección del rayo interpolando desde las esquinas de la vista.
                    double u_param = (double)x / (width - 1);
                    double v_param = 1.0 - ((double)y / (height - 1)); // y=0 es la parte superior

                    XYZ point_bottom = bottomLeft.Add(u_param * (bottomRight - bottomLeft));
                    XYZ point_top = topLeft.Add(u_param * (topRight - topLeft));
                    XYZ targetPoint = point_bottom.Add(v_param * (point_top - point_bottom));
                    
                    XYZ rayDirection = (targetPoint - eyePosition).Normalize();
                    
                    double distance = GetRayDistance(intersector, eyePosition, rayDirection);
                    
                    if (distance < 0)
                    {
                        depthSamples[sy, sx] = 0;
                    }
                    else
                    {
                        double normalized = 1.0 - ((distance - minDepth) / (maxDepth - minDepth));
                        depthSamples[sy, sx] = Math.Max(0.0, Math.Min(1.0, normalized));
                    }
                }
            }
            
            // Fase 2: Interpolar para obtener imagen completa
            using (Bitmap depthMap = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        float fx = (float)x / _subsampleFactor;
                        float fy = (float)y / _subsampleFactor;
                        
                        int x0 = (int)Math.Floor(fx);
                        int y0 = (int)Math.Floor(fy);
                        int x1 = Math.Min(x0 + 1, sampleWidth - 1);
                        int y1 = Math.Min(y0 + 1, sampleHeight - 1);
                        
                        float dx = fx - x0;
                        float dy = fy - y0;
                        
                        double d00 = depthSamples[Math.Max(0, Math.Min(y0, sampleHeight - 1)), Math.Max(0, Math.Min(x0, sampleWidth - 1))];
                        double d10 = depthSamples[Math.Max(0, Math.Min(y0, sampleHeight - 1)), x1];
                        double d01 = depthSamples[y1, Math.Max(0, Math.Min(x0, sampleWidth - 1))];
                        double d11 = depthSamples[y1, x1];
                        
                        double depth = (1 - dx) * (1 - dy) * d00 +
                                      dx * (1 - dy) * d10 +
                                      (1 - dx) * dy * d01 +
                                      dx * dy * d11;
                        
                        byte depthValue = (byte)(depth * 255);
                        depthMap.SetPixel(x, y, Drawing.Color.FromArgb(depthValue, depthValue, depthValue));
                    }
                }
                
                string depthPath = System.IO.Path.Combine(outputPath, $"depth_{timestamp}.png");
                depthMap.Save(depthPath, ImageFormat.Png);
                
                string currentDepthPath = System.IO.Path.Combine(outputPath, "current_depth.png");
                depthMap.Save(currentDepthPath, ImageFormat.Png);
            }
        }
        
        private (double min, double max) CalculateDepthRange(Document doc, View3D view3D, XYZ eyePosition, XYZ forwardDirection)
        {
            // Obtener bounding box
            BoundingBoxXYZ viewBounds = view3D.GetSectionBox();
            if (!viewBounds.Enabled)
            {
                viewBounds = GetVisibleElementsBounds(doc, view3D);
            }
            
            // Calcular profundidades de las esquinas
            var corners = GetBoundingBoxCorners(viewBounds);
            double minDepth = double.MaxValue;
            double maxDepth = double.MinValue;
            
            foreach (var corner in corners)
            {
                XYZ vectorToCorner = corner.Subtract(eyePosition);
                double depth = vectorToCorner.DotProduct(forwardDirection);
                if (depth > 0)
                {
                    minDepth = Math.Min(minDepth, depth);
                    maxDepth = Math.Max(maxDepth, depth);
                }
            }
            
            // Asegurar rango válido
            if (minDepth >= maxDepth || minDepth == double.MaxValue)
            {
                minDepth = 1.0;
                maxDepth = 100.0;
            }
            
            // Agregar margen
            double range = maxDepth - minDepth;
            minDepth = Math.Max(0.1, minDepth - range * 0.1);
            maxDepth = maxDepth + range * 0.1;
            
            return (minDepth, maxDepth);
        }
        
        private double GetRayDistance(ReferenceIntersector intersector, XYZ origin, XYZ direction)
        {
            try
            {
                ReferenceWithContext refContext = intersector.FindNearest(origin, direction);
                if (refContext != null)
                {
                    return refContext.Proximity;
                }
                return -1;
            }
            catch
            {
                return -1;
            }
        }
        
        private ICollection<ElementId> GetIntersectableElementIds(Document doc, View3D view3D)
        {
            var allowedCategories = new List<BuiltInCategory>
            {
                BuiltInCategory.OST_Walls,
                BuiltInCategory.OST_Floors,
                BuiltInCategory.OST_Roofs,
                BuiltInCategory.OST_Ceilings,
                BuiltInCategory.OST_GenericModel,
                BuiltInCategory.OST_Furniture,
                BuiltInCategory.OST_StructuralColumns,
                BuiltInCategory.OST_Doors,
                BuiltInCategory.OST_Windows,
                BuiltInCategory.OST_Stairs
            };
            
            var categoryFilter = new ElementMulticategoryFilter(allowedCategories);
            
            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WherePasses(categoryFilter)
                .WhereElementIsNotElementType();
            
            return collector.ToElementIds();
        }
        
        private BoundingBoxXYZ GetVisibleElementsBounds(Document doc, View3D view3D)
        {
            BoundingBoxXYZ bounds = new BoundingBoxXYZ();
            bounds.Min = new XYZ(double.MaxValue, double.MaxValue, double.MaxValue);
            bounds.Max = new XYZ(double.MinValue, double.MinValue, double.MinValue);
            
            var elementIds = GetIntersectableElementIds(doc, view3D);
            
            bool foundAny = false;
            foreach (ElementId id in elementIds)
            {
                Element elem = doc.GetElement(id);
                if (elem != null)
                {
                    BoundingBoxXYZ elemBounds = elem.get_BoundingBox(view3D);
                    if (elemBounds != null)
                    {
                        bounds.Min = new XYZ(
                            Math.Min(bounds.Min.X, elemBounds.Min.X),
                            Math.Min(bounds.Min.Y, elemBounds.Min.Y),
                            Math.Min(bounds.Min.Z, elemBounds.Min.Z));
                        
                        bounds.Max = new XYZ(
                            Math.Max(bounds.Max.X, elemBounds.Max.X),
                            Math.Max(bounds.Max.Y, elemBounds.Max.Y),
                            Math.Max(bounds.Max.Z, elemBounds.Max.Z));
                        
                        foundAny = true;
                    }
                }
            }
            
            if (!foundAny)
            {
                bounds.Min = new XYZ(-100, -100, -100);
                bounds.Max = new XYZ(100, 100, 100);
            }
            
            return bounds;
        }
        
        private XYZ[] GetBoundingBoxCorners(BoundingBoxXYZ bbox)
        {
            return new XYZ[]
            {
                new XYZ(bbox.Min.X, bbox.Min.Y, bbox.Min.Z),
                new XYZ(bbox.Max.X, bbox.Min.Y, bbox.Min.Z),
                new XYZ(bbox.Min.X, bbox.Max.Y, bbox.Min.Z),
                new XYZ(bbox.Max.X, bbox.Max.Y, bbox.Min.Z),
                new XYZ(bbox.Min.X, bbox.Min.Y, bbox.Max.Z),
                new XYZ(bbox.Max.X, bbox.Min.Y, bbox.Max.Z),
                new XYZ(bbox.Min.X, bbox.Max.Y, bbox.Max.Z),
                new XYZ(bbox.Max.X, bbox.Max.Y, bbox.Max.Z)
            };
        }
    }
}