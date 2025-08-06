// GeometryCacheManager.cs --- VERSIÓN ACTUALIZADA CON PROCESAMIENTO SECUENCIAL Y MEJOR FEEDBACK ---
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Runtime.InteropServices; // Añadido para StructLayout y Marshal
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using Drawing = System.Drawing;
using WabiSabiBridge.Extractors.Gpu; // Para GpuAccelerationManager, RayTracingConfig
using ComputeSharp;                   // Para Float3
using WabiSabiBridge;                 // Para WabiSabiLogger
using Newtonsoft.Json;                // Añadido para la serialización en WriteStateFile
using System.Collections.Concurrent;  // Para ConcurrentDictionary

namespace WabiSabiBridge.Extractors.Cache
{
    // --- CORRECCIÓN DEFINITIVA: Clase de extensión estática y de nivel superior ---
    // Se coloca aquí, directamente dentro del namespace, para que sea visible en todo el archivo.
    public static class MeshExtensions
    {
        public static IList<MeshTriangle> GetTriangles(this Mesh mesh)
        {
            var triangles = new List<MeshTriangle>(mesh.NumTriangles);
            for (int i = 0; i < mesh.NumTriangles; i++)
            {
                triangles.Add(mesh.get_Triangle(i));
            }
            return triangles;
        }
    }
    // --- INICIO DE CORRECCIÓN 1: AÑADIR XYZCOMPARER ---
    // Comparador para la clase XYZ que permite agrupar vértices muy cercanos.
    public class XyzComparer : IEqualityComparer<XYZ>
    {           
        private readonly double _tolerance;

        public XyzComparer(double tolerance = 1e-9)
        {
            _tolerance = tolerance;
        }

        public bool Equals(XYZ? p1, XYZ? p2)
        {
            if (p1 == null && p2 == null) return true;
            if (p1 == null || p2 == null) return false;

            return Math.Abs(p1.X - p2.X) < _tolerance &&
                   Math.Abs(p1.Y - p2.Y) < _tolerance &&
                   Math.Abs(p1.Z - p2.Z) < _tolerance;
        }

        public int GetHashCode(XYZ p)
        {
            // Multiplicar por un número grande y redondear para agrupar vértices cercanos
            int hashX = (int)(Math.Round(p.X / _tolerance) * _tolerance * 1000);
            int hashY = (int)(Math.Round(p.Y / _tolerance) * _tolerance * 1000);
            int hashZ = (int)(Math.Round(p.Z / _tolerance) * _tolerance * 1000);
            return hashX ^ (hashY << 12) ^ (hashZ << 24);
        }
    }
    // --- FIN DE CORRECCIÓN 1 ---

    public sealed class GeometryCacheManager : IDisposable
    {
        [StructLayout(LayoutKind.Sequential, Pack = 1)]
        private struct GeometryHeader
        {
            public int VertexCount;
            public int TriangleCount;
            public int CategoryCount;
            public long VerticesOffset;
            public long IndicesOffset;
            public long NormalsOffset;
            public long ElementIdsOffset;
            public long CategoryIdsOffset;
            public long CategoryMappingOffset;
        }

        private class RawMeshData
        {
            public required IList<XYZ> Vertices { get; init; }
            public required IList<MeshTriangle> Triangles { get; init; }
            public required Element Element { get; init; }
        }

        private static readonly Lazy<GeometryCacheManager> _instance = 
            new Lazy<GeometryCacheManager>(() => new GeometryCacheManager());
        
        public static GeometryCacheManager Instance => _instance.Value;
        
        private MemoryMappedFile? _geometryMmf;
        private MemoryMappedViewAccessor? _geometryAccessor;
        private bool _isCacheValid = false;
        private string _lastModelStateHash = string.Empty;
        private string _currentMmfName = string.Empty;
        private readonly string _persistentCacheDirectory;
        private readonly object _cacheLock = new object();
        
        public int VertexCount { get; private set; }
        public int TriangleCount { get; private set; }
        public int CategoryCount { get; private set; }
        public long CacheSizeBytes { get; private set; }
        public DateTime LastCacheTime { get; private set; }
        public bool IsCacheValid => _isCacheValid && _geometryMmf != null;
        
        private int _cacheHits = 0;
        private int _cacheMisses = 0;
        private TimeSpan _totalExtractionTime = TimeSpan.Zero;
        
        private GeometryCacheManager() 
        {
            _persistentCacheDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "WabiSabiBridge",
                "GeometryCache"
            );
            Directory.CreateDirectory(_persistentCacheDirectory);
            System.Diagnostics.Debug.WriteLine("GeometryCacheManager: Inicializado");
        }

        public async Task<CachedGeometryData> EnsureCacheIsValidAsync(
            Document doc,
            View3D view3D,
            Action<string, Drawing.Color>? updateStatusCallback = null,
            CancellationToken cancellationToken = default)
        {
            return await Task.Run(() =>
            {
                return EnsureCacheIsValid(doc, view3D, updateStatusCallback);
            }, cancellationToken);
        }
        
        public CachedGeometryData EnsureCacheIsValid(
            Document doc, 
            View3D view3D, 
            Action<string, Drawing.Color>? updateStatusCallback = null)
        {
            lock (_cacheLock)
            {
                string currentModelHash = ComputeModelStateHash(doc);
                
                bool needsRebuild = !_isCacheValid || 
                                _lastModelStateHash != currentModelHash ||
                                _geometryMmf == null;
                
                if (needsRebuild)
                {
                    if (TryLoadFromPersistentCache(currentModelHash, updateStatusCallback))
                    {
                        _cacheHits++;
                        updateStatusCallback?.Invoke($"Caché cargado desde disco (Hits: {_cacheHits})", Drawing.Color.Green);
                        return CreateCachedGeometryData();
                    }

                    _cacheMisses++;
                    updateStatusCallback?.Invoke($"Reconstruyendo caché de geometría (Misses: {_cacheMisses})...", Drawing.Color.Orange);
                        
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    
                    RebuildCache(doc, view3D, updateStatusCallback);
                    
                    sw.Stop();
                    _totalExtractionTime = _totalExtractionTime.Add(sw.Elapsed);
                    updateStatusCallback?.Invoke($"Caché reconstruido en {sw.ElapsedMilliseconds}ms", Drawing.Color.Blue);
                    
                    _lastModelStateHash = currentModelHash;
                    
                    SaveToPersistentCache(currentModelHash, updateStatusCallback);
                }
                else
                {
                    _cacheHits++;
                    updateStatusCallback?.Invoke($"Usando caché en memoria existente (Hit Rate: {GetHitRate():P1})", Drawing.Color.Green);
                }
                
                return CreateCachedGeometryData();
            }
        }       


        private bool TryLoadFromPersistentCache(string modelHash, Action<string, Drawing.Color>? updateStatusCallback)
        {
            string cacheFilePath = Path.Combine(_persistentCacheDirectory, $"{modelHash}.wabi_geom");
            if (!File.Exists(cacheFilePath)) return false;
        
            try
            {
                updateStatusCallback?.Invoke("Cargando caché desde disco...", Drawing.Color.Blue);
                DisposeCurrentCache();
        
                _currentMmfName = "WabiSabi_Geometry_Cache";
                
                using (var fileStream = new FileStream(cacheFilePath, FileMode.Open, FileAccess.Read))
                {
                    try
                    {
                        var existingMmf = MemoryMappedFile.OpenExisting(_currentMmfName);
                        existingMmf.Dispose();
                    }
                    catch { /* No existe, está bien */ }
                    
                    _geometryMmf = MemoryMappedFile.CreateFromFile(
                        fileStream, 
                        _currentMmfName, 
                        fileStream.Length, 
                        MemoryMappedFileAccess.Read, 
                        HandleInheritability.None, 
                        false
                    );
                }
                
                CacheSizeBytes = new FileInfo(cacheFilePath).Length;
                
                _isCacheValid = true;
                _lastModelStateHash = modelHash;
                
                WriteStateFile();
                return true;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error cargando caché persistente", ex);
                return false;
            }
        }

        private void SaveToPersistentCache(string modelHash, Action<string, Drawing.Color>? updateStatusCallback)
        {
            if (!_isCacheValid || _geometryAccessor == null) return;
            
            string cacheFilePath = Path.Combine(_persistentCacheDirectory, $"{modelHash}.wabi_geom");
            updateStatusCallback?.Invoke($"Guardando caché en disco: {cacheFilePath}", Drawing.Color.CornflowerBlue);

            try
            {
                using (var fileStream = new FileStream(cacheFilePath, FileMode.Create, FileAccess.Write))
                {
                    using (var writer = new BinaryWriter(fileStream))
                    {
                        byte[] buffer = new byte[CacheSizeBytes];
                        _geometryAccessor.ReadArray(0, buffer, 0, (int)CacheSizeBytes);
                        writer.Write(buffer);
                    }
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error guardando caché persistente", ex);
            }
        }

        private CachedGeometryData CreateCachedGeometryData()
        {
            long vertexDataSize = (long)this.VertexCount * 3 * sizeof(float);
            long indexDataSize = (long)this.TriangleCount * 3 * sizeof(int);
            long normalDataSize = (long)this.VertexCount * 3 * sizeof(float);
            long elementIdDataSize = (long)this.VertexCount * sizeof(int);
            long categoryIdDataSize = (long)this.VertexCount * sizeof(int);
        
            return new CachedGeometryData
            {
                MmfName = this._currentMmfName,
                VertexCount = this.VertexCount,
                TriangleCount = this.TriangleCount,
                CategoryCount = this.CategoryCount,
                VerticesOffset = 0,
                IndicesOffset = vertexDataSize,
                NormalsOffset = vertexDataSize + indexDataSize,
                ElementIdsOffset = vertexDataSize + indexDataSize + normalDataSize,
                CategoryIdsOffset = vertexDataSize + indexDataSize + normalDataSize + elementIdDataSize,
                CategoryMappingOffset = vertexDataSize + indexDataSize + normalDataSize + elementIdDataSize + categoryIdDataSize,
            };
        }

        public async Task<byte[]> RenderWithCachedGeometry(
            CachedGeometryData cache, 
            ViewOrientation3D orientation, 
            int resolution)
        {
            var gpuManager = new GpuAccelerationManager(null);

            var eyePosition = orientation.EyePosition;
            var forwardDirection = orientation.ForwardDirection.Normalize();
            var upDirection = orientation.UpDirection.Normalize();
            var rightDirection = forwardDirection.CrossProduct(upDirection).Normalize();

            var config = new RayTracingConfig
            {
                EyePosition = new Float3((float)eyePosition.X, (float)eyePosition.Y, (float)eyePosition.Z),
                ViewDirection = new Float3((float)forwardDirection.X, (float)forwardDirection.Y, (float)forwardDirection.Z),
                UpDirection = new Float3((float)upDirection.Y, (float)upDirection.Y, (float)upDirection.Z),
                RightDirection = new Float3((float)rightDirection.X, (float)rightDirection.Y, (float)rightDirection.Z),
                Width = resolution,
                Height = resolution,
                MinDepth = 0.1f,
                MaxDepth = 1000.0f
            };

            float[] gpuBuffer = await gpuManager.ExecuteDepthRayTracingFromCacheAsync(
                cache.MmfName, cache.VertexCount, cache.TriangleCount, config);

            byte[] imageBytes = ConvertFloatBufferToPngBytes(gpuBuffer, resolution, resolution);
            
            return imageBytes;
        }

        private byte[] ConvertFloatBufferToPngBytes(float[] buffer, int width, int height)
        {
            using (var bitmap = new System.Drawing.Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb))
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        float depthValue = buffer[y * width + x];
                        byte grayScale = (byte)(Math.Min(Math.Max(depthValue * 255.0f, 0), 255));
                        bitmap.SetPixel(x, y, System.Drawing.Color.FromArgb(grayScale, grayScale, grayScale));
                    }
                }

                using (var ms = new MemoryStream())
                {
                    bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                    return ms.ToArray();
                }
            }
        }

        private static string ComputeModelStateHash(Document doc)
        {
            var sb = new StringBuilder();
            
            sb.Append(doc.Title);
            sb.Append(doc.PathName);
            
            var collector = new FilteredElementCollector(doc)
                .WhereElementIsNotElementType()
                .Where(e => e.Category != null && e.Category.CategoryType == CategoryType.Model);

            var elements = collector.OrderBy(e => e.UniqueId);
            
            sb.Append(elements.Count());
            
            foreach (var elem in elements.Take(2000))
            {
                sb.Append(elem.UniqueId);
                try
                {
                    var bbox = elem.get_BoundingBox(null); 
                    if (bbox != null && bbox.Enabled)
                    {
                        sb.Append(bbox.Min.X.ToString("F3"));
                        sb.Append(bbox.Min.Y.ToString("F3"));
                        sb.Append(bbox.Min.Z.ToString("F3"));
                        sb.Append(bbox.Max.X.ToString("F3"));
                        sb.Append(bbox.Max.Y.ToString("F3"));
                        sb.Append(bbox.Max.Z.ToString("F3"));
                    }
                }
                catch { /* Ignorar elementos sin bounding box */ }
            }
            
            using (var md5 = MD5.Create())
            {
                byte[] inputBytes = Encoding.UTF8.GetBytes(sb.ToString());
                byte[] hashBytes = md5.ComputeHash(inputBytes);
                return BitConverter.ToString(hashBytes).Replace("-", "");
            }
        }

        private void WriteStateFile()
        {
            try
            {
                string stateFilePath = Path.Combine(_persistentCacheDirectory, "wabisabi_state.json");
                
                var state = new
                {
                    MmfName = _currentMmfName,
                    VertexCount = VertexCount,
                    TriangleCount = TriangleCount,
                    CategoryCount = CategoryCount,
                    LastUpdate = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                    ProcessId = System.Diagnostics.Process.GetCurrentProcess().Id
                };
                
                string json = JsonConvert.SerializeObject(state, Formatting.Indented);
                File.WriteAllText(stateFilePath, json);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error escribiendo archivo de estado", ex);
            }
        }

        private void RebuildCache(Document doc, View3D view3D, Action<string, Drawing.Color>? updateStatusCallback)
        {
            try
            {
                DisposeCurrentCache();
                updateStatusCallback?.Invoke("Extrayendo geometría del modelo...", Drawing.Color.Blue);
                var sceneData = ExtractSceneGeometry(doc, view3D, updateStatusCallback);
                
                if (sceneData.VertexCount == 0)
                {
                    throw new InvalidOperationException("No se encontró geometría visible para cachear");
                }
                
                updateStatusCallback?.Invoke("Optimizando datos de geometría...", Drawing.Color.Blue);
                var vertexArray = sceneData.Vertices.ToArray();
                var indexArray = sceneData.Triangles.ToArray();
                var normalArray = sceneData.Normals.ToArray();
                var elementIdArray = sceneData.ElementIds.ToArray();
                var categoryIdArray = sceneData.CategoryIds.ToArray();
                var categoryMapBytes = SerializeCategoryMap(sceneData.CategoryMap);

                int headerSize = Marshal.SizeOf<GeometryHeader>();
                long vertexDataSize = (long)vertexArray.Length * sizeof(float);
                long indexDataSize = (long)indexArray.Length * sizeof(int);
                long normalDataSize = (long)normalArray.Length * sizeof(float);
                long elementIdDataSize = (long)elementIdArray.Length * sizeof(int);
                long categoryIdDataSize = (long)categoryIdArray.Length * sizeof(int);
                long categoryMapDataSize = categoryMapBytes.Length;
                
                var header = new GeometryHeader
                {
                    VertexCount = sceneData.VertexCount,
                    TriangleCount = sceneData.TriangleCount,
                    CategoryCount = sceneData.CategoryMap.Count,
                    VerticesOffset = headerSize,
                    IndicesOffset = headerSize + vertexDataSize,
                    NormalsOffset = headerSize + vertexDataSize + indexDataSize,
                    ElementIdsOffset = headerSize + vertexDataSize + indexDataSize + normalDataSize,
                    CategoryIdsOffset = headerSize + vertexDataSize + indexDataSize + normalDataSize + elementIdDataSize,
                    CategoryMappingOffset = headerSize + vertexDataSize + indexDataSize + normalDataSize + elementIdDataSize + categoryIdDataSize
                };
                
                long totalSize = headerSize + vertexDataSize + indexDataSize + normalDataSize + 
                                elementIdDataSize + categoryIdDataSize + categoryMapDataSize;
                
                CacheSizeBytes = totalSize;
                
                updateStatusCallback?.Invoke("Creando caché en memoria...", Drawing.Color.Blue);
                
                string mmfName = "WabiSabi_Geometry_Cache";
                _currentMmfName = mmfName;
                
                try
                {
                    var existingMmf = MemoryMappedFile.OpenExisting(mmfName);
                    existingMmf.Dispose();
                }
                catch (FileNotFoundException)
                { 
                    /* No existe, está bien, continuamos para crearlo */ 
                }
                catch (Exception ex)
                {
                    WabiSabiLogger.Log($"No se pudo disponer del MMF existente '{mmfName}'. Error: {ex.Message}", LogLevel.Warning);
                }
                
                _geometryMmf = MemoryMappedFile.CreateNew(_currentMmfName, totalSize, MemoryMappedFileAccess.ReadWrite);
                _geometryAccessor = _geometryMmf.CreateViewAccessor();
                
                updateStatusCallback?.Invoke("Escribiendo datos al caché...", Drawing.Color.Blue);
                
                _geometryAccessor.Write(0, ref header);
                
                _geometryAccessor.WriteArray(header.VerticesOffset, vertexArray, 0, vertexArray.Length);
                _geometryAccessor.WriteArray(header.IndicesOffset, indexArray, 0, indexArray.Length);
                _geometryAccessor.WriteArray(header.NormalsOffset, normalArray, 0, normalArray.Length);
                _geometryAccessor.WriteArray(header.ElementIdsOffset, elementIdArray, 0, elementIdArray.Length);
                _geometryAccessor.WriteArray(header.CategoryIdsOffset, categoryIdArray, 0, categoryIdArray.Length);
                _geometryAccessor.WriteArray(header.CategoryMappingOffset, categoryMapBytes, 0, categoryMapBytes.Length);
                
                VertexCount = sceneData.VertexCount;
                TriangleCount = sceneData.TriangleCount;
                CategoryCount = sceneData.CategoryMap.Count;
                LastCacheTime = DateTime.Now;
                _isCacheValid = true;

                string sizeInfo = CacheSizeBytes > 1048576 ? $"{CacheSizeBytes / 1048576.0:F2} MB" : $"{CacheSizeBytes / 1024.0:F2} KB";
                updateStatusCallback?.Invoke($"Caché generado: {VertexCount:N0} vértices, {TriangleCount:N0} triángulos ({sizeInfo})", Drawing.Color.Green);
                
                WriteStateFile();
            }
            catch (Exception ex)
            {
                updateStatusCallback?.Invoke($"Error fatal al generar caché: {ex.Message}", Drawing.Color.Red);
                WabiSabiLogger.LogError("Error reconstruyendo caché", ex);
                DisposeCurrentCache();
            }
        }
       
        // --- INICIO DE MODIFICACIÓN: MÉTODO ExtractSceneGeometry SECUENCIAL ---
        private ExtractedSceneData ExtractSceneGeometry(
            Document doc, 
            View3D view3D, 
            Action<string, Drawing.Color>? updateStatusCallback)
        {
            var options = new Options 
            { 
                DetailLevel = ViewDetailLevel.Medium, // Usar Medium para un buen balance
                IncludeNonVisibleObjects = false,
                ComputeReferences = true 
            };

            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WhereElementIsNotElementType()
                .Where(e => e.Category != null && e.Category.CategoryType == CategoryType.Model);
            
            var elements = collector.ToList();
            int processedCount = 0;
            int totalElements = elements.Count;

            // Objeto de datos único que se llenará de forma secuencial.
            var finalData = new ExtractedSceneData();
            
            updateStatusCallback?.Invoke($"Extrayendo geometría de {totalElements:N0} elementos de forma secuencial...", Drawing.Color.Blue);

            // Bucle foreach secuencial, como en la versión de referencia.
            foreach (var element in elements)
            {
                try
                {
                    var geometry = element.get_Geometry(options);
                    if (geometry != null)
                    {
                        // Se llama al método auxiliar para que popule directamente el objeto 'finalData'.
                        // La lógica interna de 'ExtractGeometryFromElement' y 'AddMeshToSceneData'
                        // ya es compatible con este enfoque.
                        ExtractGeometryFromElement(geometry, finalData, element);
                    }
                }
                catch { /* Ignorar elementos problemáticos */ }
                
                // Actualizar el progreso
                processedCount++;
                if (processedCount % 200 == 0)
                {
                    updateStatusCallback?.Invoke($"Extrayendo... {processedCount}/{totalElements}", Drawing.Color.Blue);
                }
            }

            updateStatusCallback?.Invoke($"Extracción completada: {finalData.VertexCount:N0} vértices", Drawing.Color.Green);
            return finalData;
        }
        // --- FIN DE MODIFICACIÓN ---

        // --- Nuevo método auxiliar para extraer y copiar la geometría de forma recursiva ---
        private void ExtractMeshesFromGeometry(GeometryElement geometryElement, Element element, List<RawMeshData> targetList)
        {
            foreach (GeometryObject geomObj in geometryElement)
            {
                if (geomObj is Solid solid && solid.Volume > 1e-6)
                {
                    foreach (Face face in solid.Faces)
                    {
                        try
                        {
                            var mesh = face.Triangulate();
                            if (mesh != null && mesh.NumTriangles > 0)
                            {
                                // Copiar los datos usando el método de extensión
                                targetList.Add(new RawMeshData { 
                                    Vertices = mesh.Vertices, 
                                    Triangles = mesh.GetTriangles(), // <-- Llamada al método de extensión
                                    Element = element 
                                });
                            }
                        }
                        catch { /* Ignorar caras que no se pueden triangular */ }
                    }
                }
                else if (geomObj is GeometryInstance instance)
                {
                    ExtractMeshesFromGeometry(instance.GetInstanceGeometry(), element, targetList);
                }
                else if (geomObj is Mesh mesh && mesh.NumTriangles > 0)
                {
                     targetList.Add(new RawMeshData { 
                        Vertices = mesh.Vertices, 
                        Triangles = mesh.GetTriangles(), // <-- Llamada al método de extensión
                        Element = element 
                    });
                }
            }
        }

        

        private void ExtractGeometryFromElement(GeometryElement geometryElement, ExtractedSceneData sceneData, Element element)
        {
            foreach (GeometryObject geomObj in geometryElement)
            {
                if (geomObj is Solid solid && solid.Volume > 1e-6)
                {
                    foreach (Face face in solid.Faces)
                    {
                        try
                        {
                            var mesh = face.Triangulate();
                            if (mesh != null && mesh.NumTriangles > 0) 
                                AddMeshToSceneData(mesh, sceneData, element);
                        }
                        catch { }
                    }
                }
                else if (geomObj is GeometryInstance instance)
                {
                    // La recursión ahora pasa los mismos parámetros.
                    ExtractGeometryFromElement(instance.GetInstanceGeometry(), sceneData, element);
                }
                else if (geomObj is Mesh mesh && mesh.NumTriangles > 0)
                {
                    AddMeshToSceneData(mesh, sceneData, element);
                }
            }
        }

        private void AddMeshToSceneData(Mesh mesh, ExtractedSceneData sceneData, Element element)
        {
            // 1. Obtener el índice base actual antes de añadir nuevos vértices.
            //    Esto funciona perfectamente en modo secuencial.
            int baseIndex = sceneData.VertexCount;
            
            // Obtener IDs de forma segura
            int elementId = (int)element.Id.Value;
            int categoryId = (int)(element.Category?.Id.Value ?? -1);
            string categoryName = element.Category?.Name ?? "Unknown";
            
            // Guardar mapeo de categorías.
            if (!sceneData.CategoryMap.ContainsKey(categoryId))
            {
                sceneData.CategoryMap[categoryId] = categoryName;
            }

            // 2. Añadir los vértices del mesh a la lista de vértices de la escena.
            foreach (XYZ vertex in mesh.Vertices)
            {
                sceneData.Vertices.Add((float)vertex.X);
                sceneData.Vertices.Add((float)vertex.Y);
                sceneData.Vertices.Add((float)vertex.Z);

                // Añadir IDs por vértice
                sceneData.ElementIds.Add(elementId);
                sceneData.CategoryIds.Add(categoryId);
            }

            // 3. Añadir los triángulos del mesh, ajustando sus índices con el baseIndex.
            for (int i = 0; i < mesh.NumTriangles; i++)
            {
                var triangle = mesh.get_Triangle(i);
                sceneData.Triangles.Add(baseIndex + (int)triangle.get_Index(0));
                sceneData.Triangles.Add(baseIndex + (int)triangle.get_Index(1));
                sceneData.Triangles.Add(baseIndex + (int)triangle.get_Index(2));
            }
            
            // --- Lógica de cálculo de normales ---
            var vertexNormals = new XYZ[mesh.Vertices.Count];
            for (int i = 0; i < vertexNormals.Length; i++) { vertexNormals[i] = XYZ.Zero; }
            
            for (int i = 0; i < mesh.NumTriangles; i++)
            {
                var triangle = mesh.get_Triangle(i);
                int index0 = (int)triangle.get_Index(0);
                int index1 = (int)triangle.get_Index(1);
                int index2 = (int)triangle.get_Index(2);
                
                var v0 = mesh.Vertices[index0];
                var v1 = mesh.Vertices[index1];
                var v2 = mesh.Vertices[index2];
                
                var faceNormal = (v1 - v0).CrossProduct(v2 - v0).Normalize();
                
                vertexNormals[index0] += faceNormal;
                vertexNormals[index1] += faceNormal;
                vertexNormals[index2] += faceNormal;
            }
            
            foreach (var normal in vertexNormals)
            {
                var finalNormal = normal.IsZeroLength() ? XYZ.BasisZ : normal.Normalize();
                sceneData.Normals.Add((float)finalNormal.X);
                sceneData.Normals.Add((float)finalNormal.Y);
                sceneData.Normals.Add((float)finalNormal.Z);
            }
        }

        // Método para calcular las normales suavizadas después de consolidar toda la geometría
        private void CalculateSmoothNormals(ExtractedSceneData sceneData)
        {
            if (sceneData.VertexCount == 0) return;

            // 1. Crear un array para almacenar la suma de normales para cada vértice.
            var vertexNormals = new XYZ[sceneData.VertexCount];
            
            // 2. Recorrer cada triángulo para calcular la normal de su cara.
            for (int i = 0; i < sceneData.TriangleCount; i++)
            {
                int index0 = sceneData.Triangles[i * 3 + 0];
                int index1 = sceneData.Triangles[i * 3 + 1];
                int index2 = sceneData.Triangles[i * 3 + 2];
                
                var v0 = new XYZ(sceneData.Vertices[index0 * 3], sceneData.Vertices[index0 * 3 + 1], sceneData.Vertices[index0 * 3 + 2]);
                var v1 = new XYZ(sceneData.Vertices[index1 * 3], sceneData.Vertices[index1 * 3 + 1], sceneData.Vertices[index1 * 3 + 2]);
                var v2 = new XYZ(sceneData.Vertices[index2 * 3], sceneData.Vertices[index2 * 3 + 1], sceneData.Vertices[index2 * 3 + 2]);
                
                var faceNormal = (v1 - v0).CrossProduct(v2 - v0);
                
                // Sumar la normal de esta cara a cada uno de los 3 vértices que la componen.
                if (vertexNormals[index0] == null) vertexNormals[index0] = XYZ.Zero;
                if (vertexNormals[index1] == null) vertexNormals[index1] = XYZ.Zero;
                if (vertexNormals[index2] == null) vertexNormals[index2] = XYZ.Zero;
                
                vertexNormals[index0] += faceNormal;
                vertexNormals[index1] += faceNormal;
                vertexNormals[index2] += faceNormal;
            }
            
            // 3. Finalmente, normalizar los vectores acumulados y añadirlos a la lista de normales de la escena.
            sceneData.Normals.Clear();
            sceneData.Normals.Capacity = sceneData.VertexCount * 3;
            foreach (var normal in vertexNormals)
            {
                var finalNormal = (normal == null || normal.IsZeroLength()) ? XYZ.BasisZ : normal.Normalize();
                sceneData.Normals.Add((float)finalNormal.X);
                sceneData.Normals.Add((float)finalNormal.Y);
                sceneData.Normals.Add((float)finalNormal.Z);
            }
        }

        private byte[] SerializeCategoryMap(Dictionary<int, string> categoryMap)
        {
            using (var ms = new MemoryStream())
            using (var writer = new BinaryWriter(ms))
            {
                writer.Write(categoryMap.Count);
                foreach (var kvp in categoryMap)
                {
                    writer.Write(kvp.Key);
                    writer.Write(kvp.Value);
                }
                return ms.ToArray();
            }
        }
        
        public void InvalidateCache()
        {
            System.Diagnostics.Debug.WriteLine("GeometryCacheManager: Caché invalidado manualmente");
            _isCacheValid = false;
        }

        public double GetHitRate()
        {
            int total = _cacheHits + _cacheMisses;
            return total > 0 ? (double)_cacheHits / total : 0;
        }
        
        public async Task WarmUpCacheAsync(Document doc, View3D view3D, 
            IProgress<string>? progress = null,
            CancellationToken cancellationToken = default)
        {
            await Task.Run(() =>
            {
                Action<string, Drawing.Color>? progressCallback = null;
                if (progress != null)
                {
                    progressCallback = (msg, color) => progress.Report(msg);
                }

                try
                {
                    progress?.Report("Iniciando pre-calentamiento del caché...");
                    EnsureCacheIsValid(doc, view3D, progressCallback);
                    progress?.Report("Caché pre-calentado exitosamente");
                }
                catch (Exception ex)
                {
                    progress?.Report($"Error en pre-calentamiento: {ex.Message}");
                }
            }, cancellationToken);
        }

        public CacheStatistics GetStatistics()
        {
            return new CacheStatistics
            {
                IsValid = this.IsCacheValid,
                VertexCount = this.VertexCount,
                TriangleCount = this.TriangleCount,
                SizeInBytes = this.CacheSizeBytes,
                LastCacheTime = this.LastCacheTime,
                CacheHits = _cacheHits,
                CacheMisses = _cacheMisses,
                HitRate = GetHitRate(),
                TotalExtractionTime = _totalExtractionTime,
                AverageExtractionTime = _cacheMisses > 0 ? 
                    TimeSpan.FromMilliseconds(_totalExtractionTime.TotalMilliseconds / _cacheMisses) : 
                    TimeSpan.Zero
            };
        }

        public void ClearPersistentCache()
        {
            try
            {
                DisposeCurrentCache();

                if (Directory.Exists(_persistentCacheDirectory))
                {
                    var files = Directory.GetFiles(_persistentCacheDirectory);
                    foreach (var file in files)
                    {
                        File.Delete(file);
                    }
                    WabiSabiLogger.Log("Caché persistente en disco ha sido limpiado.", LogLevel.Info);
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error al limpiar el caché persistente.", ex);
            }
        }

        private void DisposeCurrentCache()
        {
            _geometryAccessor?.Dispose();
            _geometryAccessor = null;
            _geometryMmf?.Dispose();
            _geometryMmf = null;
            _isCacheValid = false;
        }

        public void Dispose()
        {
            System.Diagnostics.Debug.WriteLine($"GeometryCacheManager: Disposing. {GetStatistics()}");
            DisposeCurrentCache();
        }
    }
    
    public class CacheStatistics
    {
        public bool IsValid { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public long SizeInBytes { get; set; }
        public DateTime LastCacheTime { get; set; }
        public int CacheHits { get; set; }
        public int CacheMisses { get; set; }
        public double HitRate { get; set; }
        public TimeSpan TotalExtractionTime { get; set; }
        public TimeSpan AverageExtractionTime { get; set; }

        public string GetFormattedSize()
        {
            if (SizeInBytes < 1024) return $"{SizeInBytes} B";
            if (SizeInBytes < 1024 * 1024) return $"{SizeInBytes / 1024.0:F1} KB";
            return $"{SizeInBytes / (1024.0 * 1024.0):F1} MB";
        }

        public override string ToString()
        {
            return $"Caché {(IsValid ? "Válido" : "Inválido")} - " +
                   $"{VertexCount:N0} V, {TriangleCount:N0} T ({GetFormattedSize()}) - " +
                   $"Hit Rate: {HitRate:P1} ({CacheHits} Hits / {CacheMisses} Misses)";
        }
    }

    public struct CachedGeometryData
    {
        public string MmfName { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public long VerticesOffset { get; set; }
        public long IndicesOffset { get; set; }
        public long NormalsOffset { get; set; }
        public long ElementIdsOffset { get; set; }
        public long CategoryIdsOffset { get; set; }
        public long CategoryMappingOffset { get; set; }
        public int CategoryCount { get; set; }
        public bool IsValid => !string.IsNullOrEmpty(MmfName) && VertexCount > 0 && TriangleCount > 0;
    }

    public class ExtractedSceneData
    {
        // Las propiedades permanecen igual
        public List<float> Vertices { get; } = new List<float>();
        public List<int> Triangles { get; } = new List<int>();
        public List<float> Normals { get; } = new List<float>();
        public List<int> ElementIds { get; } = new List<int>();
        public List<int> CategoryIds { get; } = new List<int>();
        public Dictionary<int, string> CategoryMap { get; } = new Dictionary<int, string>();
        
        public int VertexCount => Vertices.Count / 3;
        public int TriangleCount => Triangles.Count / 3;
        
        public void Clear()
        {
            Vertices.Clear();
            Triangles.Clear();
            Normals.Clear();
            ElementIds.Clear();
            CategoryIds.Clear();
            CategoryMap.Clear();
        }

        // --- INICIO DE LA CORRECCIÓN: MÉTODO DE FUSIÓN SEGURO ---
        // Este método está diseñado para ser llamado en un bucle SECUENCIAL
        // después de que todo el procesamiento paralelo haya terminado.
        // Aunque ya no se usa en ExtractSceneGeometry, se mantiene por si se reutiliza en otro lugar.
        public void MergeFrom(ExtractedSceneData other)
        {
            // 1. Obtener el índice base actual. Es seguro porque estamos en un bucle secuencial.
            int baseIndex = this.VertexCount; 
            
            // 2. Añadir todos los datos por vértice del otro chunk.
            this.Vertices.AddRange(other.Vertices);
            this.Normals.AddRange(other.Normals);
            this.ElementIds.AddRange(other.ElementIds);
            this.CategoryIds.AddRange(other.CategoryIds);

            // 3. Añadir los triángulos, AHORA SÍ re-indexando correctamente.
            //    Cada índice del otro chunk se desplaza por el baseIndex.
            foreach (var index in other.Triangles)
            {
                this.Triangles.Add(baseIndex + index);
            }
            
            // 4. Fusionar el mapa de categorías.
            foreach (var kvp in other.CategoryMap)
            {
                // Add or overwrite. Could use TryAdd in newer .NET versions.
                if (!this.CategoryMap.ContainsKey(kvp.Key))
                {
                    this.CategoryMap[kvp.Key] = kvp.Value;
                }
            }
        }
    }
}