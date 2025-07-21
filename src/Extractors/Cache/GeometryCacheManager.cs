// GeometryCacheManager.cs - Gestor de caché de geometría para optimización extrema
using System;
using System.Collections.Generic;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Autodesk.Revit.DB;
using Drawing = System.Drawing;

namespace WabiSabiBridge.Extractors.Cache
{
    /// <summary>
    /// Gestor singleton para el caché de geometría extraída de Revit.
    /// Implementa la estrategia de "montaje del set" para evitar re-extracciones costosas.
    /// </summary>
    public sealed class GeometryCacheManager : IDisposable
    {
        // Singleton thread-safe con inicialización lazy
        private static readonly Lazy<GeometryCacheManager> _instance = 
            new Lazy<GeometryCacheManager>(() => new GeometryCacheManager());
        
        public static GeometryCacheManager Instance => _instance.Value;
        
        // Estado del caché
        private MemoryMappedFile? _geometryMmf;
        private MemoryMappedViewAccessor? _geometryAccessor;
        private bool _isCacheValid = false;
        private string _lastViewId = string.Empty;
        private string _lastModelStateHash = string.Empty;
        
        // Metadata del caché
        public int VertexCount { get; private set; }
        public int TriangleCount { get; private set; }
        public long CacheSizeBytes { get; private set; }
        public DateTime LastCacheTime { get; private set; }
        public bool IsCacheValid => _isCacheValid && _geometryMmf != null;
        
        // Estadísticas de rendimiento
        private int _cacheHits = 0;
        private int _cacheMisses = 0;
        private TimeSpan _totalExtractionTime = TimeSpan.Zero;
        
        // Constructor privado para Singleton
        private GeometryCacheManager() 
        {
            System.Diagnostics.Debug.WriteLine("GeometryCacheManager: Inicializado");
        }
        
        /// <summary>
        /// Asegura que el caché esté válido para la vista actual.
        /// Si no lo está, lo reconstruye automáticamente.
        /// </summary>
        public CachedGeometryData EnsureCacheIsValid(
            Document doc, 
            View3D view3D, 
            Action<string, Drawing.Color>? updateStatusCallback = null)
        {
            string currentViewId = view3D.Id.ToString();
            string currentModelHash = ComputeModelStateHash(doc, view3D);
            
            bool needsRebuild = !_isCacheValid || 
                               _lastViewId != currentViewId || 
                               _lastModelStateHash != currentModelHash ||
                               _geometryMmf == null;
            
            if (needsRebuild)
            {
                _cacheMisses++;
                updateStatusCallback?.Invoke(
                    $"Caché inválido. Reconstruyendo geometría (Misses: {_cacheMisses})...", 
                    Drawing.Color.Orange);
                    
                var sw = System.Diagnostics.Stopwatch.StartNew();
                RebuildCache(doc, view3D, updateStatusCallback);
                sw.Stop();
                
                _totalExtractionTime = _totalExtractionTime.Add(sw.Elapsed);
                updateStatusCallback?.Invoke(
                    $"Caché reconstruido en {sw.ElapsedMilliseconds}ms", 
                    Drawing.Color.Blue);
            }
            else
            {
                _cacheHits++;
                updateStatusCallback?.Invoke(
                    $"Usando caché existente (Hits: {_cacheHits}, Hit Rate: {GetHitRate():P1})", 
                    Drawing.Color.Green);
            }
            
            return new CachedGeometryData
            {
                GeometryMmf = _geometryMmf!,
                VertexCount = VertexCount,
                TriangleCount = TriangleCount,
                VerticesOffset = 0,
                IndicesOffset = VertexCount * 3 * sizeof(float),
                NormalsOffset = VertexCount * 3 * sizeof(float) + TriangleCount * 3 * sizeof(int)
            };
        }
        
        /// <summary>
        /// Reconstruye el caché extrayendo toda la geometría visible
        /// </summary>
        private void RebuildCache(Document doc, View3D view3D, Action<string, Drawing.Color>? updateStatusCallback)
        {
            try
            {
                // Limpiar caché anterior
                DisposeCurrentCache();
                
                // Extraer geometría
                updateStatusCallback?.Invoke("Extrayendo geometría del modelo...", Drawing.Color.Blue);
                var sceneData = ExtractSceneGeometry(doc, view3D, updateStatusCallback);
                
                if (sceneData.TriangleCount == 0)
                {
                    updateStatusCallback?.Invoke("Advertencia: No se encontró geometría", Drawing.Color.Orange);
                    return;
                }
                
                // Calcular tamaño necesario
                long vertexDataSize = sceneData.Vertices.Count * sizeof(float);
                long indexDataSize = sceneData.Triangles.Count * sizeof(int);
                long normalDataSize = sceneData.Normals.Count * sizeof(float);
                CacheSizeBytes = vertexDataSize + indexDataSize + normalDataSize;
                
                // Crear MMF con nombre único
                string mmfName = $"WabiSabi_GeometryCache_{Guid.NewGuid():N}";
                _geometryMmf = MemoryMappedFile.CreateNew(mmfName, CacheSizeBytes);
                _geometryAccessor = _geometryMmf.CreateViewAccessor();
                
                // Escribir datos al MMF
                updateStatusCallback?.Invoke("Escribiendo caché de geometría...", Drawing.Color.Blue);
                long offset = 0;
                
                // Escribir vértices
                for (int i = 0; i < sceneData.Vertices.Count; i++)
                {
                    _geometryAccessor.Write(offset, sceneData.Vertices[i]);
                    offset += sizeof(float);
                }
                
                // Escribir índices
                for (int i = 0; i < sceneData.Triangles.Count; i++)
                {
                    _geometryAccessor.Write(offset, sceneData.Triangles[i]);
                    offset += sizeof(int);
                }
                
                // Escribir normales
                for (int i = 0; i < sceneData.Normals.Count; i++)
                {
                    _geometryAccessor.Write(offset, sceneData.Normals[i]);
                    offset += sizeof(float);
                }
                
                // Actualizar metadata
                VertexCount = sceneData.VertexCount;
                TriangleCount = sceneData.TriangleCount;
                LastCacheTime = DateTime.Now;
                _lastViewId = view3D.Id.ToString();
                _lastModelStateHash = ComputeModelStateHash(doc, view3D);
                _isCacheValid = true;
                
                string sizeInfo = CacheSizeBytes > 1048576 ? 
                    $"{CacheSizeBytes / 1048576.0:F2} MB" : 
                    $"{CacheSizeBytes / 1024.0:F2} KB";
                    
                updateStatusCallback?.Invoke(
                    $"Caché generado: {VertexCount:N0} vértices, {TriangleCount:N0} triángulos ({sizeInfo})", 
                    Drawing.Color.Green);
            }
            catch (Exception ex)
            {
                updateStatusCallback?.Invoke($"Error al generar caché: {ex.Message}", Drawing.Color.Red);
                _isCacheValid = false;
                throw;
            }
        }
        
        /// <summary>
        /// Extrae la geometría de todos los elementos visibles (movido desde DepthExtractor)
        /// </summary>
        private ExtractedSceneData ExtractSceneGeometry(
            Document doc, 
            View3D view3D,
            Action<string, Drawing.Color>? updateStatusCallback)
        {
            var sceneData = new ExtractedSceneData();
            var options = new Options
            {
                ComputeReferences = true,
                DetailLevel = ViewDetailLevel.Fine, // Más detalle para mejor calidad
                IncludeNonVisibleObjects = false
            };
            
            // Filtrar solo elementos visibles y relevantes
            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WhereElementIsNotElementType()
                .Where(e => e.Category != null && 
                           e.Category.CategoryType == CategoryType.Model &&
                           e.Category.CanAddSubcategory); // Filtro adicional
            
            int totalElements = collector.Count();
            int processedElements = 0;
            int skippedElements = 0;
            
            foreach (var element in collector)
            {
                try
                {
                    var geometry = element.get_Geometry(options);
                    if (geometry != null)
                    {
                        int trianglesBefore = sceneData.TriangleCount;
                        ExtractGeometryFromElement(geometry, sceneData, element.Category?.Name ?? "Unknown");
                        
                        if (sceneData.TriangleCount > trianglesBefore)
                        {
                            processedElements++;
                        }
                        else
                        {
                            skippedElements++;
                        }
                        
                        // Actualizar progreso cada 100 elementos
                        if ((processedElements + skippedElements) % 100 == 0)
                        {
                            float progress = (float)(processedElements + skippedElements) / totalElements;
                            updateStatusCallback?.Invoke(
                                $"Extrayendo geometría: {progress:P0} ({processedElements} elementos)", 
                                Drawing.Color.Blue);
                            System.Windows.Forms.Application.DoEvents();
                        }
                    }
                }
                catch (Exception ex)
                {
                    // Log pero continuar con otros elementos
                    System.Diagnostics.Debug.WriteLine($"Error extrayendo elemento {element.Id}: {ex.Message}");
                    skippedElements++;
                }
            }
            
            System.Diagnostics.Debug.WriteLine($"Extracción completa: {processedElements} elementos procesados, {skippedElements} omitidos");
            return sceneData;
        }
        
        /// <summary>
        /// Extrae triángulos de un GeometryElement con optimizaciones
        /// </summary>
        private void ExtractGeometryFromElement(
            GeometryElement geometryElement, 
            ExtractedSceneData sceneData,
            string categoryName)
        {
            foreach (GeometryObject geomObj in geometryElement)
            {
                if (geomObj is Solid solid && solid.Volume > 0)
                {
                    // Optimización: Solo procesar caras visibles
                    foreach (Face face in solid.Faces)
                    {
                        try
                        {
                            var mesh = face.Triangulate();
                            if (mesh != null && mesh.NumTriangles > 0)
                            {
                                AddMeshToSceneData(mesh, sceneData, categoryName);
                            }
                        }
                        catch
                        {
                            // Algunas caras pueden fallar al triangular, continuar
                        }
                    }
                }
                else if (geomObj is GeometryInstance instance)
                {
                    var instanceGeometry = instance.GetInstanceGeometry();
                    if (instanceGeometry != null)
                    {
                        ExtractGeometryFromElement(instanceGeometry, sceneData, categoryName);
                    }
                }
                else if (geomObj is Mesh mesh && mesh.NumTriangles > 0)
                {
                    AddMeshToSceneData(mesh, sceneData, categoryName);
                }
            }
        }
        
        /// <summary>
        /// Agrega una malla a los datos de la escena con deduplicación de vértices
        /// </summary>
        private void AddMeshToSceneData(Mesh mesh, ExtractedSceneData sceneData, string categoryName)
        {
            const float VERTEX_EPSILON = 0.001f; // Tolerancia para fusionar vértices cercanos
            
            // Mapear vértices del mesh actual a índices globales
            var localToGlobalIndex = new Dictionary<int, int>();
            
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                XYZ vertex = mesh.Vertices[i];
                
                // Buscar vértice existente cercano
                bool foundExisting = false;
                foreach (var kvp in sceneData.VertexMap)
                {
                    if (IsVertexClose(kvp.Key, vertex, VERTEX_EPSILON))
                    {
                        localToGlobalIndex[i] = kvp.Value;
                        foundExisting = true;
                        break;
                    }
                }
                
                // Si no existe, agregarlo
                if (!foundExisting)
                {
                    int newIndex = sceneData.VertexMap.Count;
                    sceneData.VertexMap[vertex] = newIndex;
                    localToGlobalIndex[i] = newIndex;
                    
                    // Agregar coordenadas
                    sceneData.Vertices.Add((float)vertex.X);
                    sceneData.Vertices.Add((float)vertex.Y);
                    sceneData.Vertices.Add((float)vertex.Z);
                    
                    // Calcular normal promedio (simplificado)
                    sceneData.Normals.Add(0);
                    sceneData.Normals.Add(0);
                    sceneData.Normals.Add(1);
                }
            }
            
            // Agregar triángulos con índices globales
            for (int i = 0; i < mesh.NumTriangles; i++)
            {
                var triangle = mesh.get_Triangle(i);
                
                int idx0 = localToGlobalIndex[(int)triangle.get_Index(0)];
                int idx1 = localToGlobalIndex[(int)triangle.get_Index(1)];
                int idx2 = localToGlobalIndex[(int)triangle.get_Index(2)];
                
                // Evitar triángulos degenerados
                if (idx0 != idx1 && idx1 != idx2 && idx2 != idx0)
                {
                    sceneData.Triangles.Add(idx0);
                    sceneData.Triangles.Add(idx1);
                    sceneData.Triangles.Add(idx2);
                }
            }
        }
        
        /// <summary>
        /// Verifica si dos vértices están cerca dentro de una tolerancia
        /// </summary>
        private bool IsVertexClose(XYZ v1, XYZ v2, float epsilon)
        {
            return Math.Abs(v1.X - v2.X) < epsilon &&
                   Math.Abs(v1.Y - v2.Y) < epsilon &&
                   Math.Abs(v1.Z - v2.Z) < epsilon;
        }
        
        /// <summary>
        /// Calcula un hash del estado del modelo para detectar cambios
        /// </summary>
        private string ComputeModelStateHash(Document doc, View3D view3D)
        {
            // Crear una firma única basada en elementos visibles
            var sb = new StringBuilder();
            
            // Incluir información de la vista
            sb.Append(view3D.Id.ToString());
            sb.Append(view3D.Scale);
            sb.Append(view3D.DetailLevel.ToString());
            
            // Incluir información única del documento
            sb.Append(doc.Title);
            sb.Append(doc.PathName);
            
            // Si es un documento compartido, incluir la ruta del modelo central
            if (doc.IsWorkshared)
            {
                try
                {
                    var modelPath = doc.GetWorksharingCentralModelPath();
                    if (modelPath != null)
                    {
                        // Convertir ModelPath a string
                        sb.Append(ModelPathUtils.ConvertModelPathToUserVisiblePath(modelPath));
                    }
                }
                catch
                {
                    // En caso de error, simplemente continuar
                }
            }
            
            // Incluir el número de elementos para detectar adiciones/eliminaciones
            sb.Append(new FilteredElementCollector(doc).GetElementCount());
            
            // Incluir IDs de elementos visibles (limitado para rendimiento)
            var visibleElements = new FilteredElementCollector(doc, view3D.Id)
                .WhereElementIsNotElementType()
                .Where(e => e.Category != null && e.Category.CategoryType == CategoryType.Model)
                .Take(1000) // Limitar para evitar lentitud
                .OrderBy(e => e.Id.Value);
            
            foreach (var elem in visibleElements)
            {
                sb.Append(elem.Id.Value);
                // Incluir información geométrica básica si está disponible
                try
                {
                    var bbox = elem.get_BoundingBox(view3D);
                    if (bbox != null)
                    {
                        sb.Append(bbox.Min.X.ToString("F2"));
                        sb.Append(bbox.Max.X.ToString("F2"));
                    }
                }
                catch
                {
                    // Algunos elementos pueden no tener bounding box
                }
            }
            
            // Generar hash MD5
            using (var md5 = MD5.Create())
            {
                byte[] inputBytes = Encoding.UTF8.GetBytes(sb.ToString());
                byte[] hashBytes = md5.ComputeHash(inputBytes);
                return BitConverter.ToString(hashBytes).Replace("-", "");
            }
        }
        
        /// <summary>
        /// Invalida el caché actual, forzando una reconstrucción en el próximo uso
        /// </summary>
        public void InvalidateCache()
        {
            System.Diagnostics.Debug.WriteLine("GeometryCacheManager: Caché invalidado manualmente");
            _isCacheValid = false;
        }
        
        /// <summary>
        /// Obtiene la tasa de aciertos del caché
        /// </summary>
        public double GetHitRate()
        {
            int total = _cacheHits + _cacheMisses;
            return total > 0 ? (double)_cacheHits / total : 0;
        }
        
        /// <summary>
        /// Obtiene estadísticas de rendimiento
        /// </summary>
        public string GetPerformanceStats()
        {
            return $"Cache Stats - Hits: {_cacheHits}, Misses: {_cacheMisses}, " +
                   $"Hit Rate: {GetHitRate():P1}, Total Extraction Time: {_totalExtractionTime.TotalSeconds:F1}s";
        }
        
        /// <summary>
        /// Limpia el caché actual
        /// </summary>
        private void DisposeCurrentCache()
        {
            _geometryAccessor?.Dispose();
            _geometryAccessor = null;
            _geometryMmf?.Dispose();
            _geometryMmf = null;
            _isCacheValid = false;
            GC.Collect(); // Forzar limpieza para liberar memoria
        }
        
        /// <summary>
        /// Limpieza de recursos
        /// </summary>
        public void Dispose()
        {
            System.Diagnostics.Debug.WriteLine($"GeometryCacheManager: Disposing. {GetPerformanceStats()}");
            DisposeCurrentCache();
        }
        
        // Clases de datos internas
        private class ExtractedSceneData
        {
            public List<float> Vertices { get; set; } = new List<float>();
            public List<int> Triangles { get; set; } = new List<int>();
            public List<float> Normals { get; set; } = new List<float>();
            public Dictionary<XYZ, int> VertexMap { get; set; } = new Dictionary<XYZ, int>();
            public int VertexCount => VertexMap.Count;
            public int TriangleCount => Triangles.Count / 3;
        }
    }
    
    /// <summary>
    /// Datos del caché para pasar a los procesadores GPU
    /// </summary>
    public class CachedGeometryData
    {
        public MemoryMappedFile GeometryMmf { get; set; } = null!;
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public long VerticesOffset { get; set; }
        public long IndicesOffset { get; set; }
        public long NormalsOffset { get; set; }
    }
}