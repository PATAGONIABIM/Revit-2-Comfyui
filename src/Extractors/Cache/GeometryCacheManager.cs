// GeometryCacheManager.cs - Gestor de caché de geometría con lógica de invalidación corregida
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using Autodesk.Revit.DB;
using Drawing = System.Drawing;

namespace WabiSabiBridge.Extractors.Cache
{
    public sealed class GeometryCacheManager : IDisposable
    {
        private static readonly Lazy<GeometryCacheManager> _instance = 
            new Lazy<GeometryCacheManager>(() => new GeometryCacheManager());
        
        public static GeometryCacheManager Instance => _instance.Value;
        
        private MemoryMappedFile? _geometryMmf;
        private MemoryMappedViewAccessor? _geometryAccessor;
        private bool _isCacheValid = false;
        private string _lastModelStateHash = string.Empty;
        
        public int VertexCount { get; private set; }
        public int TriangleCount { get; private set; }
        public long CacheSizeBytes { get; private set; }
        public DateTime LastCacheTime { get; private set; }
        public bool IsCacheValid => _isCacheValid && _geometryMmf != null;
        
        private int _cacheHits = 0;
        private int _cacheMisses = 0;
        private TimeSpan _totalExtractionTime = TimeSpan.Zero;
        
        private GeometryCacheManager() 
        {
            System.Diagnostics.Debug.WriteLine("GeometryCacheManager: Inicializado");
        }
        
        // --- MÉTODO CORREGIDO: LÓGICA DE INVALIDACIÓN REFINADA ---
        public CachedGeometryData EnsureCacheIsValid(
            Document doc, 
            View3D view3D, 
            Action<string, Drawing.Color>? updateStatusCallback = null)
        {
            // El hash ahora es independiente de la vista, solo depende del modelo.
            string currentModelHash = ComputeModelStateHash(doc);
            
            // La reconstrucción del caché solo es necesaria si el MODELO ha cambiado.
            // El movimiento de la cámara NO invalida el caché de geometría.
            bool needsRebuild = !_isCacheValid || 
                               _lastModelStateHash != currentModelHash ||
                               _geometryMmf == null;
            
            if (needsRebuild)
            {
                _cacheMisses++;
                updateStatusCallback?.Invoke(
                    $"Cambio en el modelo detectado. Reconstruyendo caché (Misses: {_cacheMisses})...", 
                    Drawing.Color.Orange);
                    
                var sw = System.Diagnostics.Stopwatch.StartNew();
                // Pasamos view3D solo para saber qué elementos están visibles en ESE momento de la extracción
                RebuildCache(doc, view3D, updateStatusCallback);
                sw.Stop();
                
                _totalExtractionTime = _totalExtractionTime.Add(sw.Elapsed);
                updateStatusCallback?.Invoke(
                    $"Caché reconstruido en {sw.ElapsedMilliseconds}ms", 
                    Drawing.Color.Blue);
                
                // Guardamos el nuevo hash del modelo
                _lastModelStateHash = currentModelHash;
            }
            else
            {
                _cacheHits++;
                updateStatusCallback?.Invoke(
                    $"Usando caché de geometría existente (Hits: {_cacheHits}, Hit Rate: {GetHitRate():P1})", 
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
        
        // --- MÉTODO CORREGIDO: HASH 100% INDEPENDIENTE DE LA VISTA ---
        private static string ComputeModelStateHash(Document doc)
        {
            var sb = new StringBuilder();
            
            // 1. Incluir información del documento que indica su estado general.
            sb.Append(doc.Title);
            sb.Append(doc.PathName);
            
            // 2. Usar un colector de elementos de TODO EL DOCUMENTO.
            // Esto asegura que el hash solo cambie si se añaden, eliminan o modifican elementos.
            var collector = new FilteredElementCollector(doc)
                .WhereElementIsNotElementType()
                .Where(e => e.Category != null && e.Category.CategoryType == CategoryType.Model);

            // 3. Ordenar por UniqueId para garantizar un orden consistente.
            var elements = collector.OrderBy(e => e.UniqueId);
            
            sb.Append(elements.Count()); // Un cambio en el número de elementos es un indicador clave.
            
            // 4. Crear una firma a partir de los elementos. Usamos el BoundingBox del elemento
            // en coordenadas del MODELO (pasando null a get_BoundingBox), que es independiente de la vista.
            foreach (var elem in elements.Take(2000)) // Limitar para buen rendimiento en modelos grandes
            {
                sb.Append(elem.UniqueId);
                try
                {
                    // get_BoundingBox(null) devuelve el cuadro delimitador en coordenadas del modelo.
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

        // El resto del archivo no necesita cambios, pero lo incluyo para que sea un reemplazo completo.
        private void RebuildCache(Document doc, View3D view3D, Action<string, Drawing.Color>? updateStatusCallback)
        {
            try
            {
                DisposeCurrentCache();
                updateStatusCallback?.Invoke("Extrayendo geometría del modelo...", Drawing.Color.Blue);
                var sceneData = ExtractSceneGeometry(doc, view3D, updateStatusCallback);
                
                if (sceneData.TriangleCount == 0)
                {
                    updateStatusCallback?.Invoke("Advertencia: No se encontró geometría visible para cachear.", Drawing.Color.Orange);
                    _isCacheValid = false;
                    return;
                }
                
                long vertexDataSize = (long)sceneData.Vertices.Count * sizeof(float);
                long indexDataSize = (long)sceneData.Triangles.Count * sizeof(int);
                long normalDataSize = (long)sceneData.Normals.Count * sizeof(float);
                CacheSizeBytes = vertexDataSize + indexDataSize + normalDataSize;
                
                string mmfName = $"WabiSabi_GeometryCache_{Guid.NewGuid():N}";
                _geometryMmf = MemoryMappedFile.CreateNew(mmfName, CacheSizeBytes, MemoryMappedFileAccess.ReadWrite);
                _geometryAccessor = _geometryMmf.CreateViewAccessor();
                
                updateStatusCallback?.Invoke("Escribiendo datos al caché de geometría...", Drawing.Color.Blue);
                
                // Escribir datos en buffers antes de pasarlos al MMF para eficiencia
                var vertexArray = sceneData.Vertices.ToArray();
                var indexArray = sceneData.Triangles.ToArray();
                var normalArray = sceneData.Normals.ToArray();
                
                long offset = 0;
                _geometryAccessor.WriteArray(offset, vertexArray, 0, vertexArray.Length);
                offset += vertexDataSize;
                _geometryAccessor.WriteArray(offset, indexArray, 0, indexArray.Length);
                offset += indexDataSize;
                _geometryAccessor.WriteArray(offset, normalArray, 0, normalArray.Length);

                VertexCount = sceneData.VertexCount;
                TriangleCount = sceneData.TriangleCount;
                LastCacheTime = DateTime.Now;
                _isCacheValid = true;
                
                string sizeInfo = CacheSizeBytes > 1048576 ? $"{CacheSizeBytes / 1048576.0:F2} MB" : $"{CacheSizeBytes / 1024.0:F2} KB";
                updateStatusCallback?.Invoke($"Caché generado: {VertexCount:N0} vértices, {TriangleCount:N0} triángulos ({sizeInfo})", Drawing.Color.Green);
            }
            catch (Exception ex)
            {
                updateStatusCallback?.Invoke($"Error fatal al generar caché: {ex.Message}", Drawing.Color.Red);
                _isCacheValid = false;
                DisposeCurrentCache(); // Limpiar en caso de error
            }
        }

        private ExtractedSceneData ExtractSceneGeometry(Document doc, View3D view3D, Action<string, Drawing.Color>? updateStatusCallback)
        {
            var sceneData = new ExtractedSceneData();
            var options = new Options { ComputeReferences = false, DetailLevel = ViewDetailLevel.Fine, IncludeNonVisibleObjects = false };
            var collector = new FilteredElementCollector(doc, view3D.Id).WhereElementIsNotElementType().Where(e => e.Category != null && e.Category.CategoryType == CategoryType.Model && e.get_Geometry(options) != null);
            
            int totalElements = collector.Count();
            int processedElements = 0;

            foreach (var element in collector)
            {
                try
                {
                    var geometry = element.get_Geometry(options);
                    ExtractGeometryFromElement(geometry, sceneData);
                    processedElements++;
                    if (processedElements % 100 == 0)
                    {
                        updateStatusCallback?.Invoke($"Extrayendo geometría: {((float)processedElements / totalElements):P0}", Drawing.Color.Blue);
                        System.Windows.Forms.Application.DoEvents();
                    }
                }
                catch { /* Ignorar elementos problemáticos */ }
            }
            return sceneData;
        }

        private void ExtractGeometryFromElement(GeometryElement geometryElement, ExtractedSceneData sceneData)
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
                            if (mesh != null && mesh.NumTriangles > 0) AddMeshToSceneData(mesh, sceneData);
                        }
                        catch { /* Ignorar caras que no se pueden triangular */ }
                    }
                }
                else if (geomObj is GeometryInstance instance)
                {
                    ExtractGeometryFromElement(instance.GetInstanceGeometry(), sceneData);
                }
                else if (geomObj is Mesh mesh && mesh.NumTriangles > 0)
                {
                    AddMeshToSceneData(mesh, sceneData);
                }
            }
        }

        private void AddMeshToSceneData(Mesh mesh, ExtractedSceneData sceneData)
        {
            int baseIndex = sceneData.VertexCount;
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                XYZ v = mesh.Vertices[i];
                sceneData.Vertices.Add((float)v.X);
                sceneData.Vertices.Add((float)v.Y);
                sceneData.Vertices.Add((float)v.Z);
            }
            for (int i = 0; i < mesh.NumTriangles; i++)
            {
                var t = mesh.get_Triangle(i);
                sceneData.Triangles.Add(baseIndex + (int)t.get_Index(0));
                sceneData.Triangles.Add(baseIndex + (int)t.get_Index(1));
                sceneData.Triangles.Add(baseIndex + (int)t.get_Index(2));
            }
            // Añadir normales placeholder
            for (int i = 0; i < mesh.Vertices.Count; i++)
            {
                sceneData.Normals.Add(0);
                sceneData.Normals.Add(0);
                sceneData.Normals.Add(1);
            }
            sceneData.VertexCount += mesh.Vertices.Count;
            sceneData.TriangleCount += mesh.NumTriangles;
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

        public string GetPerformanceStats()
        {
            return $"Stats - Hits: {_cacheHits}, Misses: {_cacheMisses}, Rate: {GetHitRate():P1}, Avg.Ext.Time: {(_cacheMisses > 0 ? _totalExtractionTime.TotalSeconds / _cacheMisses : 0):F1}s";
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
            System.Diagnostics.Debug.WriteLine($"GeometryCacheManager: Disposing. {GetPerformanceStats()}");
            DisposeCurrentCache();
        }
        
        private class ExtractedSceneData
        {
            public List<float> Vertices { get; set; } = new List<float>();
            public List<int> Triangles { get; set; } = new List<int>();
            public List<float> Normals { get; set; } = new List<float>();
            public int VertexCount { get; set; } = 0;
            public int TriangleCount { get; set; } = 0;
        }
    }
    
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