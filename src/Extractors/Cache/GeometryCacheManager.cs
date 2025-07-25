// GeometryCacheManager.cs --- VERSIÓN ACTUALIZADA CON PROCESAMIENTO ASÍNCRONO Y MEJOR FEEDBACK ---
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
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

        // --- NUEVO: MÉTODO ASÍNCRONO QUE NO BLOQUEA LA UI ---
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
                IndicesOffset = (long)VertexCount * 3 * sizeof(float),
                NormalsOffset = (long)VertexCount * 3 * sizeof(float) + (long)TriangleCount * 3 * sizeof(int)
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

        private void RebuildCache(Document doc, View3D view3D, Action<string, Drawing.Color>? updateStatusCallback)
        {
            try
            {
                DisposeCurrentCache();
                updateStatusCallback?.Invoke("Extrayendo geometría del modelo...", Drawing.Color.Blue);
                
                var sceneData = ExtractSceneGeometry(doc, view3D, updateStatusCallback);
                
                WriteToMemoryMappedFile(sceneData, updateStatusCallback);

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

        // --- MÉTODO MEJORADO DE EXTRACCIÓN CON MEJOR FEEDBACK ---
        private ExtractedSceneData ExtractSceneGeometry(
            Document doc, 
            View3D view3D, 
            Action<string, Drawing.Color>? updateStatusCallback)
        {
            var sceneData = new ExtractedSceneData();
            var options = new Options 
            { 
                ComputeReferences = false, 
                DetailLevel = ViewDetailLevel.Fine, 
                IncludeNonVisibleObjects = false 
            };
            
            // Usar un colector más eficiente
            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WhereElementIsNotElementType()
                .Where(e => e.Category != null && 
                           e.Category.CategoryType == CategoryType.Model && 
                           e.get_Geometry(options) != null);
            
            var elements = collector.ToList();
            int totalElements = elements.Count;
            int processedElements = 0;
            int lastReportedProgress = -1;
            
            // Procesar en lotes para mejor rendimiento
            const int batchSize = 50;
            var batches = elements.Select((element, index) => new { element, index })
                                 .GroupBy(x => x.index / batchSize)
                                 .Select(g => g.Select(x => x.element).ToList());

            foreach (var batch in batches)
            {
                foreach (var element in batch)
                {
                    try
                    {
                        var geometry = element.get_Geometry(options);
                        if (geometry != null)
                        {
                            ExtractGeometryFromElement(geometry, sceneData);
                        }
                        processedElements++;
                    }
                    catch (Exception ex)
                    {
                        // Log del error pero continuar con otros elementos
                        System.Diagnostics.Debug.WriteLine($"Error procesando elemento {element.Id}: {ex.Message}");
                    }
                }

                // Actualizar progreso solo cuando cambie significativamente
                int currentProgress = (int)((float)processedElements / totalElements * 100);
                if (currentProgress > lastReportedProgress && currentProgress % 5 == 0 && totalElements > 0)
                {
                    lastReportedProgress = currentProgress;
                    updateStatusCallback?.Invoke(
                        $"Extrayendo geometría: {currentProgress}% ({processedElements:N0}/{totalElements:N0} elementos)", 
                        Drawing.Color.Blue);
                    
                    // Permitir que la UI se actualice
                    System.Windows.Forms.Application.DoEvents();
                }
            }

            updateStatusCallback?.Invoke(
                $"Extracción completada: {sceneData.VertexCount:N0} vértices, {sceneData.TriangleCount:N0} triángulos", 
                Drawing.Color.Green);
                
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
            foreach (XYZ v in mesh.Vertices)
            {
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
        }
        
        // --- NUEVO: MÉTODO MEJORADO PARA ESCRIBIR AL MMF CON VALIDACIÓN ---
        private void WriteToMemoryMappedFile(ExtractedSceneData sceneData, Action<string, Drawing.Color>? updateStatusCallback)
        {
            try
            {
                updateStatusCallback?.Invoke("Optimizando datos de geometría...", Drawing.Color.Blue);
                
                // Convertir listas a arrays para mejor rendimiento
                var vertexArray = sceneData.Vertices.ToArray();
                var indexArray = sceneData.Triangles.ToArray();
                var normalArray = sceneData.Normals.ToArray();
                
                // Calcular tamaños con validación
                long vertexDataSize = (long)vertexArray.Length * sizeof(float);
                long indexDataSize = (long)indexArray.Length * sizeof(int);
                long normalDataSize = (long)normalArray.Length * sizeof(float);
                CacheSizeBytes = vertexDataSize + indexDataSize + normalDataSize;

                if (CacheSizeBytes == 0)
                {
                    // Usa InvalidOperationException que es más apropiado que un simple return.
                    throw new InvalidOperationException("No se encontró geometría visible para cachear");
                }

                updateStatusCallback?.Invoke("Creando caché en memoria...", Drawing.Color.Blue);
                
                // Crear MMF con un nombre único
                string mmfName = $"WabiSabi_GeometryCache_{Guid.NewGuid():N}";
                _geometryMmf = MemoryMappedFile.CreateNew(mmfName, CacheSizeBytes, MemoryMappedFileAccess.ReadWrite);
                _geometryAccessor = _geometryMmf.CreateViewAccessor();
                
                // Escribir en bloques para mostrar progreso
                const int blockSize = 1024 * 1024; // 1MB blocks
                long totalBytes = CacheSizeBytes;
                long bytesWritten = 0;
                
                updateStatusCallback?.Invoke("Escribiendo datos al caché...", Drawing.Color.Blue);
                
                // Escribir vértices
                _geometryAccessor.WriteArray(bytesWritten, vertexArray, 0, vertexArray.Length);
                bytesWritten += vertexDataSize;

                // Escribir índices
                _geometryAccessor.WriteArray(bytesWritten, indexArray, 0, indexArray.Length);
                bytesWritten += indexDataSize;

                // Escribir normales
                _geometryAccessor.WriteArray(bytesWritten, normalArray, 0, normalArray.Length);

                VertexCount = sceneData.VertexCount;
                TriangleCount = sceneData.TriangleCount;
                LastCacheTime = DateTime.Now;
                _isCacheValid = true;
            }
            catch (Exception ex)
            {
                DisposeCurrentCache();
                throw new Exception($"Error al crear caché de geometría: {ex.Message}", ex);
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

        // --- NUEVO: MÉTODO PARA OBTENER ESTADÍSTICAS DEL CACHÉ ---
        public CacheStatistics GetStatistics()
        {
            return new CacheStatistics
            {
                IsValid = IsCacheValid,
                VertexCount = VertexCount,
                TriangleCount = TriangleCount,
                SizeInBytes = CacheSizeBytes,
                LastCacheTime = LastCacheTime,
                CacheHits = _cacheHits,
                CacheMisses = _cacheMisses,
                HitRate = GetHitRate(),
                TotalExtractionTime = _totalExtractionTime,
                AverageExtractionTime = _cacheMisses > 0 ? 
                    TimeSpan.FromMilliseconds(_totalExtractionTime.TotalMilliseconds / _cacheMisses) : 
                    TimeSpan.Zero
            };
        }

        // --- NUEVO: MÉTODO PARA PRE-CALENTAR EL CACHÉ (ÚTIL AL INICIAR) ---
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
    
    // --- NUEVO: CLASE PARA ESTADÍSTICAS DEL CACHÉ ---
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
            if (SizeInBytes < 1024)
                return $"{SizeInBytes} B";
            if (SizeInBytes < 1024 * 1024)
                return $"{SizeInBytes / 1024.0:F1} KB";
            if (SizeInBytes < 1024 * 1024 * 1024)
                return $"{SizeInBytes / (1024.0 * 1024.0):F1} MB";
            return $"{SizeInBytes / (1024.0 * 1024.0 * 1024.0):F1} GB";
        }

        public override string ToString()
        {
            return $"Caché {(IsValid ? "Válido" : "Inválido")} - " +
                   $"{VertexCount:N0} vértices, {TriangleCount:N0} triángulos ({GetFormattedSize()}) - " +
                   $"Hit Rate: {HitRate:P1} ({CacheHits} hits, {CacheMisses} misses) - " +
                   $"Tiempo promedio: {AverageExtractionTime.TotalSeconds:F1}s";
        }
    }
    
    // --- ACTUALIZADO: ESTRUCTURA DE DATOS DEL CACHÉ CON TODA LA INFORMACIÓN NECESARIA ---
    public struct CachedGeometryData
    {
        public MemoryMappedFile? GeometryMmf { get; set; }
        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public long VerticesOffset { get; set; }
        public long IndicesOffset { get; set; }
        public long NormalsOffset { get; set; }
        
        public bool IsValid => GeometryMmf != null && VertexCount > 0 && TriangleCount > 0;
    }

    // --- ACTUALIZADO: DATOS EXTRAÍDOS DE LA ESCENA ---
    public class ExtractedSceneData
    {
        public List<float> Vertices { get; } = new List<float>();
        public List<int> Triangles { get; } = new List<int>();
        public List<float> Normals { get; } = new List<float>();
        
        public int VertexCount => Vertices.Count / 3;
        public int TriangleCount => Triangles.Count / 3;
        
        public void Clear()
        {
            Vertices.Clear();
            Triangles.Clear();
            Normals.Clear();
        }
    }
}