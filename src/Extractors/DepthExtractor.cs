// DepthExtractor.cs - Extractor de mapa de profundidad optimizado con soporte GPU
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Threading.Tasks;
using System.Buffers;
using System.IO.MemoryMappedFiles;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Drawing = System.Drawing;
using WabiSabiBridge.Extractors.Gpu;
using ComputeSharp;

namespace WabiSabiBridge.Extractors
{
    public class DepthExtractor : IDisposable
    {
        private readonly UIApplication _uiApp;
        private readonly int _resolution;
        private int _debugRayCount = 0;
        // CORREGIDO: Declarado como nullable para resolver la advertencia CS8618.
        private IGpuAccelerationManager? _gpuManager; 
        private readonly ArrayPool<float> _floatPool = ArrayPool<float>.Shared;
        private readonly ArrayPool<int> _intPool = ArrayPool<int>.Shared;

        public bool AutoDepthRange { get; set; } = true;
        public double ManualDepthDistance { get; set; } = 50.0;
        public bool UseGpuAcceleration { get; set; } = true;
        public bool UseGeometryExtraction { get; set; } = false; // Experimental

        // Estructura para datos extraídos
        private class ExtractedSceneData
        {
            public List<float> Vertices { get; set; } = new List<float>();
            public List<int> Triangles { get; set; } = new List<int>();
            public List<float> Normals { get; set; } = new List<float>();
            public Dictionary<XYZ, int> VertexMap { get; set; } = new Dictionary<XYZ, int>();
            public int VertexCount => VertexMap.Count;
            public int TriangleCount => Triangles.Count / 3;
        }

        public DepthExtractor(UIApplication uiApp, int resolution = 512)
        {
            _uiApp = uiApp;
            _resolution = resolution;
            
            // Intentar inicializar GPU si está habilitado
            if (UseGpuAcceleration)
            {
                try
                {
                    // Intentar primero con ComputeSharp
                    try
                    {
                        _gpuManager = new GpuAccelerationManager();
                    }
                    catch
                    {
                        // Si falla, usar versión simplificada
                        _gpuManager = new GpuAccelerationSimple();
                    }
                    
                    System.Diagnostics.Debug.WriteLine($"GPU Acceleration: {(_gpuManager.IsGpuAvailable ? "Disponible" : "No disponible (usando CPU paralela)")}");
                }
                catch (Exception ex)
                {
                    System.Diagnostics.Debug.WriteLine($"Error inicializando GPU: {ex.Message}");
                    UseGpuAcceleration = false;
                }
            }
        }
        
        public void ExtractDepthMap(View3D view3D, string outputPath, string timestamp, int width, int height, IList<XYZ> viewCorners)
        {
            Document doc = _uiApp.ActiveUIDocument.Document;
            
            var viewOrientation = view3D.GetOrientation();
            var eyePosition = viewOrientation.EyePosition;
            var forwardDirection = viewOrientation.ForwardDirection.Normalize();
            var upDirection = viewOrientation.UpDirection.Normalize();
            var rightDirection = forwardDirection.CrossProduct(upDirection).Normalize();
            
            // Reconstruir las cuatro esquinas del plano de la vista
            XYZ bottomLeft = viewCorners[0];
            XYZ topRight = viewCorners[1];
            XYZ viewX_vec = topRight - bottomLeft;
            XYZ viewWidthVector = viewX_vec.DotProduct(rightDirection) * rightDirection;
            XYZ viewHeightVector = viewX_vec.DotProduct(upDirection) * upDirection;
            XYZ bottomRight = bottomLeft + viewWidthVector;
            XYZ topLeft = bottomLeft + viewHeightVector;
            
            // Calcular el centro de la vista
            XYZ viewCenter = new XYZ(
                (bottomLeft.X + topRight.X) / 2.0,
                (bottomLeft.Y + topRight.Y) / 2.0,
                (bottomLeft.Z + topRight.Z) / 2.0
            );

            // Configurar el rango de profundidad
            double minDepth = 0.1;
            double maxDepth = CalculateMaxDepth(view3D, eyePosition, forwardDirection, viewCenter);

            System.Diagnostics.Debug.WriteLine($"=== WabiSabi Depth Debug (GPU Optimized) ===");
            System.Diagnostics.Debug.WriteLine($"View Type: {(view3D.IsPerspective ? "Perspective" : "Orthographic")}");
            System.Diagnostics.Debug.WriteLine($"Resolution: {width}x{height}");
            System.Diagnostics.Debug.WriteLine($"Depth Range: {minDepth:F2} to {maxDepth:F2}");
            System.Diagnostics.Debug.WriteLine($"GPU Acceleration: {(UseGpuAcceleration && _gpuManager?.IsGpuAvailable == true ? "Enabled" : "Disabled")}");
            
            _debugRayCount = 0;
            
            double[,] depthData;
            
            if (UseGeometryExtraction && UseGpuAcceleration && _gpuManager?.IsGpuAvailable == true)
            {
                // MODO EXPERIMENTAL: Extracción de geometría completa + GPU
                depthData = ProcessWithGeometryExtractionGpu(
                    doc, view3D, eyePosition, forwardDirection, upDirection, rightDirection,
                    bottomLeft, bottomRight, topLeft, topRight,
                    width, height, minDepth, maxDepth
                );
            }
            else
            {
                // MODO ESTÁNDAR: ReferenceIntersector con optimizaciones
                depthData = ProcessWithReferenceIntersector(
                    doc, view3D, eyePosition,
                    bottomLeft, bottomRight, topLeft, topRight,
                    width, height, minDepth, maxDepth
                );
            }
            
            // Generar imagen (paralelo)
            GenerateDepthImage(depthData, width, height, outputPath, timestamp);
            
            System.Diagnostics.Debug.WriteLine("=== End WabiSabi Depth Debug ===");
        }
        
        /// <summary>
        /// Modo experimental: Extrae toda la geometría y procesa en GPU
        /// </summary>
        private double[,] ProcessWithGeometryExtractionGpu(
            Document doc, View3D view3D, XYZ eyePosition,
            XYZ forwardDirection, XYZ upDirection, XYZ rightDirection,
            XYZ bottomLeft, XYZ bottomRight, XYZ topLeft, XYZ topRight,
            int width, int height, double minDepth, double maxDepth)
        {
            if (_gpuManager == null) // Guardia por si la GPU no se inicializó
            {
                System.Diagnostics.Debug.WriteLine("Se intentó usar el modo de extracción de geometría sin un gestor de GPU válido. Volviendo al modo estándar.");
                return ProcessWithReferenceIntersector(doc, view3D, eyePosition, bottomLeft, bottomRight, topLeft, topRight, width, height, minDepth, maxDepth);
            }
            
            System.Diagnostics.Debug.WriteLine("=== MODO GPU: Extracción de geometría ===");
            
            // FASE 1: Extraer geometría (Main Thread)
            var sceneData = ExtractSceneGeometry(doc, view3D);
            System.Diagnostics.Debug.WriteLine($"Geometría extraída: {sceneData.VertexCount} vértices, {sceneData.TriangleCount} triángulos");
            
            if (sceneData.TriangleCount == 0)
            {
                System.Diagnostics.Debug.WriteLine("No se encontró geometría para procesar");
                return new double[height, width];
            }
            
            // FASE 2: Procesar en GPU (Async)
            var depthTask = Task.Run(async () =>
            {
                // Preparar configuración de ray tracing
                var config = new RayTracingConfig
                {
                    EyePosition = new Float3((float)eyePosition.X, (float)eyePosition.Y, (float)eyePosition.Z),
                    ViewDirection = new Float3((float)forwardDirection.X, (float)forwardDirection.Y, (float)forwardDirection.Z),
                    UpDirection = new Float3((float)upDirection.X, (float)upDirection.Y, (float)upDirection.Z),
                    RightDirection = new Float3((float)rightDirection.X, (float)rightDirection.Y, (float)rightDirection.Z),
                    Width = width,
                    Height = height,
                    MinDepth = (float)minDepth,
                    MaxDepth = (float)maxDepth
                };
                
                // Crear memory-mapped file para geometría grande
                string mmfName = $"WabiSabi_Geometry_{Guid.NewGuid():N}";
                long dataSize = (sceneData.Vertices.Count + sceneData.Triangles.Count) * 4 + sceneData.Normals.Count * 4;
                
                _gpuManager.CreateGeometrySharedMemory(mmfName, dataSize);
                _gpuManager.WriteGeometryData(
                    sceneData.Vertices.ToArray(),
                    sceneData.Triangles.ToArray(),
                    sceneData.Normals.ToArray()
                );
                
                var geometry = new ExtractedGeometry
                {
                    VertexCount = sceneData.VertexCount,
                    TriangleCount = sceneData.TriangleCount,
                    VerticesOffset = 0,
                    IndicesOffset = sceneData.Vertices.Count * sizeof(float),
                    NormalsOffset = sceneData.Normals.Count > 0 ? 
                        sceneData.Vertices.Count * sizeof(float) + sceneData.Triangles.Count * sizeof(int) : 0
                };
                
                // Ejecutar en GPU
                float[] gpuDepthBuffer = await _gpuManager.ExecuteDepthRayTracingAsync(geometry, config);
                
                // Convertir a double[,]
                double[,] result = new double[height, width];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        result[y, x] = gpuDepthBuffer[y * width + x];
                    }
                }
                
                return result;
            });
            
            return depthTask.Result;
        }
        
        /// <summary>
        /// Extrae la geometría de todos los elementos visibles
        /// </summary>
        private ExtractedSceneData ExtractSceneGeometry(Document doc, View3D view3D)
        {
            var sceneData = new ExtractedSceneData();
            var options = new Options
            {
                ComputeReferences = true,
                DetailLevel = ViewDetailLevel.Coarse,
                IncludeNonVisibleObjects = false
            };
            
            // Obtener elementos visibles
            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WhereElementIsNotElementType()
                .Where(e => e.Category != null && e.Category.CategoryType == CategoryType.Model);
            
            int processedElements = 0;
            foreach (var element in collector)
            {
                try
                {
                    var geometry = element.get_Geometry(options);
                    if (geometry != null)
                    {
                        ExtractGeometryFromElement(geometry, sceneData);
                        processedElements++;
                        
                        // Permitir que Revit responda cada cierto número de elementos
                        if (processedElements % 100 == 0)
                        {
                            System.Windows.Forms.Application.DoEvents();
                        }
                    }
                }
                catch (Exception ex)
                {
                    // Ignorar elementos problemáticos
                    System.Diagnostics.Debug.WriteLine($"Error extrayendo geometría: {ex.Message}");
                }
            }
            
            return sceneData;
        }
        
        /// <summary>
        /// Extrae triángulos de un GeometryElement
        /// </summary>
        private void ExtractGeometryFromElement(GeometryElement geometryElement, ExtractedSceneData sceneData)
        {
            foreach (GeometryObject geomObj in geometryElement)
            {
                if (geomObj is Solid solid && solid.Volume > 0)
                {
                    foreach (Face face in solid.Faces)
                    {
                        if (face is PlanarFace || face is CylindricalFace || face is RuledFace)
                        {
                            var mesh = face.Triangulate();
                            if (mesh != null)
                            {
                                AddMeshToSceneData(mesh, sceneData);
                            }
                        }
                    }
                }
                else if (geomObj is GeometryInstance instance)
                {
                    var instanceGeometry = instance.GetInstanceGeometry();
                    if (instanceGeometry != null)
                    {
                        ExtractGeometryFromElement(instanceGeometry, sceneData);
                    }
                }
                else if (geomObj is Mesh mesh)
                {
                    AddMeshToSceneData(mesh, sceneData);
                }
            }
        }
        
        /// <summary>
        /// Agrega una malla a los datos de la escena
        /// </summary>
        private void AddMeshToSceneData(Mesh mesh, ExtractedSceneData sceneData)
        {
            int baseIndex = sceneData.VertexCount;
            
            // Agregar vértices
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                XYZ vertex = mesh.Vertices[i];
                
                if (!sceneData.VertexMap.ContainsKey(vertex))
                {
                    sceneData.VertexMap[vertex] = sceneData.VertexMap.Count;
                    sceneData.Vertices.Add((float)vertex.X);
                    sceneData.Vertices.Add((float)vertex.Y);
                    sceneData.Vertices.Add((float)vertex.Z);
                    
                    // Normal placeholder (podríamos calcularla)
                    sceneData.Normals.Add(0);
                    sceneData.Normals.Add(0);
                    sceneData.Normals.Add(1);
                }
            }
            
            // Agregar triángulos
            for (int i = 0; i < mesh.NumTriangles; i++)
            {
                var triangle = mesh.get_Triangle(i);
                
                sceneData.Triangles.Add(sceneData.VertexMap[mesh.Vertices[(int)triangle.get_Index(0)]]);
                sceneData.Triangles.Add(sceneData.VertexMap[mesh.Vertices[(int)triangle.get_Index(1)]]);
                sceneData.Triangles.Add(sceneData.VertexMap[mesh.Vertices[(int)triangle.get_Index(2)]]);
            }
        }
        
        /// <summary>
        /// Modo estándar: Usa ReferenceIntersector con optimizaciones
        /// </summary>
        private double[,] ProcessWithReferenceIntersector(
            Document doc, View3D view3D, XYZ eyePosition,
            XYZ bottomLeft, XYZ bottomRight, XYZ topLeft, XYZ topRight,
            int width, int height, double minDepth, double maxDepth)
        {
            // Pre-calcular todos los rayos
            var allRays = PreCalculateRays(
                eyePosition, bottomLeft, bottomRight, topLeft, topRight,
                width, height
            );
            
            // Procesar rayos en el thread principal de Revit
            var depthData = new double[height, width];
            ProcessRaysInBatches(doc, view3D, eyePosition, allRays, depthData, minDepth, maxDepth);
            
            return depthData;
        }
        
        /// <summary>
        /// Pre-calcula las direcciones de todos los rayos
        /// </summary>
        private List<RayData> PreCalculateRays(
            XYZ eyePosition, XYZ bottomLeft, XYZ bottomRight, XYZ topLeft, XYZ topRight,
            int width, int height)
        {
            var allRays = new List<RayData>(width * height);
            
            // Usar chunks para evitar Large Object Heap
            const int chunkSize = 10000;
            
            Parallel.For(0, (height * width + chunkSize - 1) / chunkSize, chunkIndex =>
            {
                int startIdx = chunkIndex * chunkSize;
                int endIdx = Math.Min(startIdx + chunkSize, height * width);
                var localRays = new List<RayData>(endIdx - startIdx);
                
                for (int idx = startIdx; idx < endIdx; idx++)
                {
                    int x = idx % width;
                    int y = idx / width;
                    
                    double u_param = (double)x / (width - 1);
                    double v_param = 1.0 - ((double)y / (height - 1));

                    XYZ point_bottom = bottomLeft.Add(u_param * (bottomRight - bottomLeft));
                    XYZ point_top = topLeft.Add(u_param * (topRight - topLeft));
                    XYZ targetPoint = point_bottom.Add(v_param * (point_top - point_bottom));
                    
                    localRays.Add(new RayData
                    {
                        Direction = (targetPoint - eyePosition).Normalize(),
                        X = x,
                        Y = y
                    });
                }
                
                lock (allRays)
                {
                    allRays.AddRange(localRays);
                }
            });
            
            return allRays;
        }
        
        private void ProcessRaysInBatches(Document doc, View3D view3D, XYZ eyePosition, 
            List<RayData> allRays, double[,] depthData, double minDepth, double maxDepth)
        {
            ICollection<ElementId> elementIds = GetIntersectableElementIds(doc, view3D);
            System.Diagnostics.Debug.WriteLine($"Intersectable elements found: {elementIds.Count}");
            
            var intersector = new ReferenceIntersector(elementIds, FindReferenceTarget.Element, view3D)
            {
                FindReferencesInRevitLinks = false,
                TargetType = FindReferenceTarget.Face
            };
            
            int hitCount = 0;
            int totalRays = allRays.Count;
            double minHitDistance = double.MaxValue;
            double maxHitDistance = double.MinValue;
            
            // Procesar en lotes pequeños para mejor responsividad
            int batchSize = UseGpuAcceleration ? 5000 : 1000; // Lotes más grandes si tenemos GPU
            int processedRays = 0;
            
            for (int i = 0; i < allRays.Count; i += batchSize)
            {
                int currentBatchSize = Math.Min(batchSize, allRays.Count - i);
                
                // Procesar lote actual
                for (int j = 0; j < currentBatchSize; j++)
                {
                    var ray = allRays[i + j];
                    double distance = GetRayDistance(intersector, eyePosition, ray.Direction);
                    
                    if (distance >= 0)
                    {
                        hitCount++;
                        minHitDistance = Math.Min(minHitDistance, distance);
                        maxHitDistance = Math.Max(maxHitDistance, distance);
                        
                        double normalized = 1.0 - ((distance - minDepth) / (maxDepth - minDepth));
                        depthData[ray.Y, ray.X] = Math.Max(0.0, Math.Min(1.0, normalized));
                    }
                    else
                    {
                        depthData[ray.Y, ray.X] = 0;
                    }
                }
                
                processedRays += currentBatchSize;
                
                // Permitir que Revit procese eventos
                System.Windows.Forms.Application.DoEvents();
                
                // Mostrar progreso
                if (processedRays % 10000 == 0)
                {
                    double progress = (double)processedRays / totalRays * 100;
                    System.Diagnostics.Debug.WriteLine($"Progress: {progress:F1}%");
                }
            }
            
            System.Diagnostics.Debug.WriteLine($"Ray hits: {hitCount}/{totalRays} ({(hitCount * 100.0 / totalRays):F1}%)");
            if (hitCount > 0)
            {
                System.Diagnostics.Debug.WriteLine($"Hit distance range: {minHitDistance:F2} to {maxHitDistance:F2}");
            }
        }
        
        private void GenerateDepthImage(double[,] depthData, int width, int height, 
            string outputPath, string timestamp)
        {
            // Usar ArrayPool para evitar allocaciones grandes
            int pixelDataSize = width * height * 3;
            byte[] pixelData = ArrayPool<byte>.Shared.Rent(pixelDataSize);
            
            try
            {
                // Procesamiento paralelo optimizado
                Parallel.For(0, height, new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount }, y =>
                {
                    int rowOffset = y * width * 3;
                    for (int x = 0; x < width; x++)
                    {
                        byte depthValue = (byte)(depthData[y, x] * 255);
                        int pixelOffset = rowOffset + x * 3;
                        pixelData[pixelOffset] = depthValue;
                        pixelData[pixelOffset + 1] = depthValue;
                        pixelData[pixelOffset + 2] = depthValue;
                    }
                });
                
                // Crear bitmap desde array
                using (Bitmap depthMap = new Bitmap(width, height, PixelFormat.Format24bppRgb))
                {
                    BitmapData bmpData = depthMap.LockBits(
                        new Drawing.Rectangle(0, 0, width, height),
                        ImageLockMode.WriteOnly,
                        depthMap.PixelFormat);
                    
                    System.Runtime.InteropServices.Marshal.Copy(pixelData, 0, bmpData.Scan0, width * height * 3);
                    depthMap.UnlockBits(bmpData);
                    
                    // Guardar con compresión óptima
                    var encoderParams = new EncoderParameters(1);
                    encoderParams.Param[0] = new EncoderParameter(Encoder.Quality, 90L);
                    
                    var pngCodec = ImageCodecInfo.GetImageEncoders().First(codec => codec.FormatID == Drawing.Imaging.ImageFormat.Png.Guid);
                    
                    string depthPath = System.IO.Path.Combine(outputPath, $"depth_{timestamp}.png");
                    depthMap.Save(depthPath, pngCodec, encoderParams);
                    
                    string currentDepthPath = System.IO.Path.Combine(outputPath, "current_depth.png");
                    depthMap.Save(currentDepthPath, pngCodec, encoderParams);
                    
                    System.Diagnostics.Debug.WriteLine($"Depth map saved to: {currentDepthPath}");
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(pixelData);
            }
        }
        
        private double CalculateMaxDepth(View3D view3D, XYZ eyePosition, XYZ forwardDirection, XYZ viewCenter)
        {
            if (!AutoDepthRange)
            {
                System.Diagnostics.Debug.WriteLine($"Using manual depth range: {ManualDepthDistance:F2} feet");
                return ManualDepthDistance;
            }

            double targetDistance = 10.0;
            
            if (view3D.IsPerspective)
            {
                double distanceToTarget = -1.0;

                BoundingBoxXYZ cropBox = view3D.CropBox;
                if (cropBox.Enabled)
                {
                    XYZ cropCenter = (cropBox.Min + cropBox.Max) * 0.5;
                    XYZ eyeToCenter = cropCenter.Subtract(eyePosition);
                    double distanceAlongView = eyeToCenter.DotProduct(forwardDirection);
                    
                    if (distanceAlongView > 0)
                    {
                        distanceToTarget = distanceAlongView;
                    }
                }

                if (distanceToTarget < 0)
                {
                    XYZ eyeToCenter = viewCenter.Subtract(eyePosition);
                    double distanceToViewPlane = eyeToCenter.DotProduct(forwardDirection);
                    if (distanceToViewPlane > 0)
                    {
                        distanceToTarget = distanceToViewPlane;
                    }
                }

                if (distanceToTarget > 0)
                {
                    targetDistance = distanceToTarget;
                    return targetDistance * 1.2;
                }
                else
                {
                    BoundingBoxUV outline = view3D.Outline;
                    double viewWidth = outline.Max.U - outline.Min.U;
                    targetDistance = viewWidth / (2.0 * Math.Tan(Math.PI / 6.0));
                    return targetDistance * 1.5;
                }
            }
            else
            {
                return 50.0;
            }
        }
        
        private double GetRayDistance(ReferenceIntersector intersector, XYZ origin, XYZ direction)
        {
            try
            {
                ReferenceWithContext refContext = intersector.FindNearest(origin, direction);
                
                if (refContext != null && refContext.GetReference() != null)
                {
                    double distance = refContext.Proximity;
                    
                    if (_debugRayCount < 10)
                    {
                        _debugRayCount++;
                        Element hitElement = _uiApp.ActiveUIDocument.Document.GetElement(refContext.GetReference());
                        string elementInfo = hitElement != null ? 
                            $"{hitElement.Category?.Name ?? "NoCategory"}: {hitElement.Name}" : 
                            "Unknown";
                        System.Diagnostics.Debug.WriteLine($"Ray hit #{_debugRayCount}: Distance={distance:F2}, Element={elementInfo}");
                    }
                    
                    return distance;
                }
                else
                {
                    return -1;
                }
            }
            catch (Exception ex)
            {
                if (_debugRayCount == 0)
                {
                    _debugRayCount++;
                    System.Diagnostics.Debug.WriteLine($"RayDistance Error: {ex.Message}");
                }
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
            
            var categoryFilter = new ElementMulticategoryFilter(allowedCategories);
            
            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WherePasses(categoryFilter)
                .WhereElementIsNotElementType();
            
            return collector.ToElementIds();
        }
        
        // Estructura para pre-calcular datos de rayos
        private struct RayData
        {
            public XYZ Direction;
            public int X;
            public int Y;
        }
        
        // Cleanup
        public void Dispose()
        {
            _gpuManager?.Dispose();
        }
    }
}