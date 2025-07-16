// DepthExtractor.cs - Extractor de mapa de profundidad para WabiSabi Bridge (Versión Optimizada)
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Drawing = System.Drawing;

namespace WabiSabiBridge.Extractors
{
    public class DepthExtractor
    {
        private readonly UIApplication _uiApp;
        private readonly int _resolution;
        
        public DepthExtractor(UIApplication uiApp, int resolution = 512)
        {
            _uiApp = uiApp;
            _resolution = resolution;
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
            
            var viewBounds = view3D.GetSectionBox();
            if (!viewBounds.Enabled)
            {
                viewBounds = GetVisibleElementsBounds(doc, view3D);
            }
            
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
            
            if (minDepth >= maxDepth || minDepth == double.MaxValue)
            {
                minDepth = 1.0;
                maxDepth = 100.0;
            }
            
            double range = maxDepth - minDepth;
            minDepth = Math.Max(0.1, minDepth - range * 0.1);
            maxDepth = maxDepth + range * 0.1;

            // <-- CAMBIO: Reconstruir las cuatro esquinas del plano de la vista para una interpolación precisa.
            XYZ bottomLeft = viewCorners[0];
            XYZ topRight = viewCorners[1];
            XYZ viewX_vec = topRight - bottomLeft;
            XYZ viewWidthVector = viewX_vec.DotProduct(rightDirection) * rightDirection;
            XYZ viewHeightVector = viewX_vec.DotProduct(upDirection) * upDirection;
            XYZ bottomRight = bottomLeft + viewWidthVector;
            XYZ topLeft = bottomLeft + viewHeightVector;
            
            using (Bitmap depthMap = new Bitmap(width, height, PixelFormat.Format24bppRgb))
            {
                BitmapData bmpData = depthMap.LockBits(
                    new Drawing.Rectangle(0, 0, width, height),
                    ImageLockMode.WriteOnly,
                    depthMap.PixelFormat);
                
                unsafe
                {
                    byte* ptr = (byte*)bmpData.Scan0;
                    int stride = bmpData.Stride;
                    
                    ICollection<ElementId> elementIds = GetIntersectableElementIds(doc, view3D);                    
                    ReferenceIntersector intersector = new ReferenceIntersector(
                        elementIds,
                        FindReferenceTarget.Element,
                        view3D);
                    
                    for (int y = 0; y < height; y++)
                    {
                        for (int x = 0; x < width; x++)
                        {
                            // <-- CAMBIO: Calcular la dirección del rayo interpolando desde las esquinas de la vista,
                            // en lugar de usar un FOV fijo. Esto garantiza un encuadre perfecto.
                            double u_param = (double)x / (width - 1);
                            double v_param = 1.0 - ((double)y / (height - 1)); // y=0 es la parte superior

                            XYZ point_bottom = bottomLeft.Add(u_param * (bottomRight - bottomLeft));
                            XYZ point_top = topLeft.Add(u_param * (topRight - topLeft));
                            XYZ targetPoint = point_bottom.Add(v_param * (point_top - point_bottom));
                            
                            XYZ rayDirection = (targetPoint - eyePosition).Normalize();
                            
                            double distance = GetRayDistance(intersector, eyePosition, rayDirection);
                            
                            byte depthValue;
                            if (distance < 0)
                            {
                                depthValue = 0;
                            }
                            else
                            {
                                double normalized = 1.0 - ((distance - minDepth) / (maxDepth - minDepth));
                                normalized = Math.Max(0.0, Math.Min(1.0, normalized));
                                depthValue = (byte)(normalized * 255);
                            }
                            
                            int pixelOffset = y * stride + x * 3;
                            ptr[pixelOffset] = depthValue;
                            ptr[pixelOffset + 1] = depthValue;
                            ptr[pixelOffset + 2] = depthValue;
                        }
                    }
                }
                
                depthMap.UnlockBits(bmpData);
                
                string depthPath = System.IO.Path.Combine(outputPath, $"depth_{timestamp}.png");
                depthMap.Save(depthPath, ImageFormat.Png);
                
                string currentDepthPath = System.IO.Path.Combine(outputPath, "current_depth.png");
                depthMap.Save(currentDepthPath, ImageFormat.Png);
            }
        }
        
        /// <summary>
        /// Lanza un rayo y obtiene la distancia al primer objeto
        /// </summary>
        private double GetRayDistance(ReferenceIntersector intersector, XYZ origin, XYZ direction)
        {
            try
            {
                // Encontrar la intersección más cercana
                ReferenceWithContext refContext = intersector.FindNearest(origin, direction);
                
                if (refContext != null && refContext.GetReference() != null)
                {
                    // Calcular distancia
                    double distance = refContext.Proximity;
                    return distance;
                }
                else
                {
                    // No hay intersección
                    return -1;
                }
            }
            catch
            {
                // En caso de error, retornar sin intersección
                return -1;
            }
        }
        
        /// <summary>
        /// Obtiene los IDs de elementos que pueden ser intersectados
        /// </summary>
        private ICollection<ElementId> GetIntersectableElementIds(Document doc, View3D view3D)
        {
            // Categorías de elementos sólidos que queremos incluir
            var allowedCategories = new List<BuiltInCategory>
            {
                BuiltInCategory.OST_Walls,
                BuiltInCategory.OST_Floors,
                BuiltInCategory.OST_Roofs,
                BuiltInCategory.OST_Ceilings,
                BuiltInCategory.OST_GenericModel,
                BuiltInCategory.OST_Furniture,
                BuiltInCategory.OST_StructuralColumns,
                BuiltInCategory.OST_StructuralFraming,
                BuiltInCategory.OST_Doors,
                BuiltInCategory.OST_Windows,
                BuiltInCategory.OST_CurtainWallPanels,
                BuiltInCategory.OST_CurtainWallMullions,
                BuiltInCategory.OST_Stairs,
                BuiltInCategory.OST_StairsRailing,
                BuiltInCategory.OST_Ramps,
                BuiltInCategory.OST_Topography,
                BuiltInCategory.OST_Site,
                BuiltInCategory.OST_Parking,
                BuiltInCategory.OST_Planting,
                BuiltInCategory.OST_Entourage,
                BuiltInCategory.OST_Casework,
                BuiltInCategory.OST_Columns,
                BuiltInCategory.OST_MechanicalEquipment,
                BuiltInCategory.OST_ElectricalEquipment,
                BuiltInCategory.OST_ElectricalFixtures,
                BuiltInCategory.OST_LightingFixtures,
                BuiltInCategory.OST_PlumbingFixtures
            };
            
            // Crear filtro de categorías
            var categoryFilter = new ElementMulticategoryFilter(allowedCategories);
            
            // Colectar elementos visibles en la vista
            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WherePasses(categoryFilter)
                .WhereElementIsNotElementType();
            
            return collector.ToElementIds();
        }
        
        /// <summary>
        /// Obtiene los límites de todos los elementos visibles
        /// </summary>
        private BoundingBoxXYZ GetVisibleElementsBounds(Document doc, View3D view3D)
        {
            BoundingBoxXYZ bounds = new BoundingBoxXYZ();
            bounds.Min = new XYZ(double.MaxValue, double.MaxValue, double.MaxValue);
            bounds.Max = new XYZ(double.MinValue, double.MinValue, double.MinValue);
            
            // Obtener elementos visibles
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
            
            // Si no se encontraron elementos, usar límites por defecto
            if (!foundAny)
            {
                bounds.Min = new XYZ(-100, -100, -100);
                bounds.Max = new XYZ(100, 100, 100);
            }
            
            return bounds;
        }
        
        /// <summary>
        /// Obtiene las 8 esquinas de un bounding box
        /// </summary>
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