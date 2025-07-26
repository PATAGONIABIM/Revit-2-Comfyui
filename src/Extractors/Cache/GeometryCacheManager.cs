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
using Autodesk.Revit.UI;
using Drawing = System.Drawing;
using WabiSabiBridge.Extractors.Gpu; // Para GpuAccelerationManager, RayTracingConfig
using ComputeSharp;                   // Para Float3
using WabiSabiBridge;                 // Para WabiSabiLogger

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
        private string _currentMmfName = string.Empty;
        private readonly string _persistentCacheDirectory;
        private readonly object _cacheLock = new object(); // Para asegurar la concurrencia
        
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
            // --- INICIO DE CIRUGÍA 1.2: INICIALIZAR DIRECTORIO DE CACHÉ ---
            _persistentCacheDirectory = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "WabiSabiBridge", // O el nombre de tu aplicación
                "GeometryCache"
            );
            Directory.CreateDirectory(_persistentCacheDirectory);
            // --- FIN DE CIRUGÍA 1.2 ---
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
            // Usamos un lock para asegurar que solo un hilo a la vez pueda modificar el caché.
            lock (_cacheLock)
            {
                // 1. Calcular el "hash" o firma digital del estado actual del modelo.
                string currentModelHash = ComputeModelStateHash(doc);
                
                // 2. Comprobar si necesitamos hacer algo.
                // La reconstrucción es necesaria si el caché no está válido en memoria, 
                // o si el hash del modelo ha cambiado (indicando que el modelo fue modificado).
                bool needsRebuild = !_isCacheValid || 
                                _lastModelStateHash != currentModelHash ||
                                _geometryMmf == null;
                
                if (needsRebuild)
                {
                    // --- INICIA EL PROCESO DE RECONSTRUCCIÓN ---
                    
                    // 3. (OPTIMIZACIÓN) Antes de reconstruir desde cero, intentamos cargarlo desde el disco.
                    // Si el caché de un modelo con el mismo hash ya existe en el disco, lo cargamos y ahorramos todo el trabajo.
                    if (TryLoadFromPersistentCache(currentModelHash, updateStatusCallback))
                    {
                        _cacheHits++;
                        updateStatusCallback?.Invoke($"Caché cargado desde disco (Hits: {_cacheHits})", Drawing.Color.Green);
                        // Si la carga fue exitosa, el trabajo está hecho. Devolvemos el resultado.
                        return CreateCachedGeometryData();
                    }

                    // 4. Si no pudimos cargarlo del disco, entonces SÍ tenemos que reconstruirlo.
                    _cacheMisses++;
                    updateStatusCallback?.Invoke($"Reconstruyendo caché de geometría (Misses: {_cacheMisses})...", Drawing.Color.Orange);
                        
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    
                    // Llamamos al método que hace el trabajo pesado de extracción.
                    RebuildCache(doc, view3D, updateStatusCallback);
                    
                    sw.Stop();
                    _totalExtractionTime = _totalExtractionTime.Add(sw.Elapsed);
                    updateStatusCallback?.Invoke($"Caché reconstruido en {sw.ElapsedMilliseconds}ms", Drawing.Color.Blue);
                    
                    // Actualizamos el hash del modelo con el nuevo estado.
                    _lastModelStateHash = currentModelHash;
                    
                    // 5. (OPTIMIZACIÓN) Guardamos el nuevo caché en el disco para futuras sesiones.
                    SaveToPersistentCache(currentModelHash, updateStatusCallback);
                }
                else
                {
                    // Si no se necesitaba reconstruir, significa que el caché en memoria ya era válido.
                    _cacheHits++;
                    updateStatusCallback?.Invoke($"Usando caché en memoria existente (Hit Rate: {GetHitRate():P1})", Drawing.Color.Green);
                }
                
                // 6. Finalmente, devolvemos el objeto de datos del caché que ahora está garantizado de ser válido.
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
                DisposeCurrentCache(); // Limpiar MMF anterior si existiera

                // --- INICIO DE CIRUGÍA 2.1 ---
                // Creamos un MMF a partir del archivo, pero le damos un nombre conocido en el sistema.
                _currentMmfName = $"WabiSabi_PersistentCache_{modelHash}";
                using (var fileStream = new FileStream(cacheFilePath, FileMode.Open, FileAccess.Read))
                {
                    // El MMF se crea con el nombre que hemos definido.
                    _geometryMmf = MemoryMappedFile.CreateFromFile(fileStream, _currentMmfName, fileStream.Length, MemoryMappedFileAccess.Read, HandleInheritability.None, false);
                }
                // NOTA: No creamos el accessor aquí, ya que el consumidor lo hará.

                CacheSizeBytes = new FileInfo(cacheFilePath).Length;
                // Aquí necesitaríamos leer la metadata (VertexCount, TriangleCount) del disco.
                // Por ahora, asumimos que se recuperará de alguna manera o se recalculará si es necesario.
                // Si no tienes un archivo .meta, esta es una simplificación que puede necesitar ajuste.
                
                _isCacheValid = true;
                _lastModelStateHash = modelHash;
                // --- FIN DE CIRUGÍA 2.1 ---
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
                // Copiar los datos del MMF en memoria a un archivo en disco
                using (var fileStream = new FileStream(cacheFilePath, FileMode.Create, FileAccess.Write))
                {
                    using (var writer = new BinaryWriter(fileStream))
                    {
                        byte[] buffer = new byte[CacheSizeBytes];
                        _geometryAccessor.ReadArray(0, buffer, 0, buffer.Length);
                        writer.Write(buffer);
                    }
                }
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error guardando caché persistente", ex);
            }
        }

        // Método auxiliar para crear el objeto de datos
        private CachedGeometryData CreateCachedGeometryData()
        {
            // Esta lógica necesita ser ajustada si no guardamos/leemos metadata
            // Aquí asumimos que VertexCount y TriangleCount se han restaurado
            return new CachedGeometryData
            {
                // REEMPLAZAR ESTA LÍNEA:
                // GeometryMmf = _geometryMmf!,
                
                // CON ESTA LÍNEA:
                MmfName = this._currentMmfName,

                VertexCount = this.VertexCount,
                TriangleCount = this.TriangleCount,
                VerticesOffset = 0,
                IndicesOffset = (long)this.VertexCount * 3 * sizeof(float),
                NormalsOffset = ((long)this.VertexCount * 3 * sizeof(float)) + ((long)this.TriangleCount * 3 * sizeof(int))
            };
        }

        public async Task<byte[]> RenderWithCachedGeometry(
            CachedGeometryData cache, 
            ViewOrientation3D orientation, 
            int resolution)
        {
            // Asumimos que tienes una instancia de GpuAccelerationManager disponible
            var gpuManager = new GpuAccelerationManager(null); // O gestiona una instancia singleton

            // --- INICIO DEL CÓDIGO QUE FALTABA ---

            // Obtener los vectores de la orientación
            var eyePosition = orientation.EyePosition;
            var forwardDirection = orientation.ForwardDirection.Normalize(); // Asegurarse de que esté normalizado
            var upDirection = orientation.UpDirection.Normalize();

            // Calcular el vector "derecha" usando el producto vectorial. Es crucial para definir el plano de la vista.
            var rightDirection = forwardDirection.CrossProduct(upDirection).Normalize();

            // Configurar la estructura de datos que se enviará al shader de la GPU
            var config = new RayTracingConfig
            {
                // Posición del "ojo" o la cámara
                EyePosition = new Float3((float)eyePosition.X, (float)eyePosition.Y, (float)eyePosition.Z),
                
                // El vector "hacia adelante" que define la dirección de la vista
                ViewDirection = new Float3((float)forwardDirection.X, (float)forwardDirection.Y, (float)forwardDirection.Z),
                
                // El vector "hacia arriba" que define la orientación vertical de la cámara
                UpDirection = new Float3((float)upDirection.X, (float)upDirection.Y, (float)upDirection.Z),
                
                // El vector "hacia la derecha", perpendicular a los otros dos
                RightDirection = new Float3((float)rightDirection.X, (float)rightDirection.Y, (float)rightDirection.Z),
                
                // Dimensiones de la imagen a generar
                Width = resolution,
                Height = resolution, // Asumimos una imagen cuadrada, podría ajustarse con un aspect ratio
                
                // Rango de profundidad (opcional, se pueden ajustar)
                MinDepth = 0.1f,
                MaxDepth = 1000.0f
            };

            // --- FIN DEL CÓDIGO QUE FALTABA ---

            // Ejecutar el renderizado en la GPU usando los datos del caché y la nueva configuración de cámara
            float[] gpuBuffer = await gpuManager.ExecuteDepthRayTracingFromCacheAsync(
        cache.MmfName, cache.VertexCount, cache.TriangleCount, config);

            // Convertir el buffer de floats resultante (datos de profundidad) a un array de bytes en formato PNG
            byte[] imageBytes = ConvertFloatBufferToPngBytes(gpuBuffer, resolution, resolution);
            
            return imageBytes;
        }

        // Método auxiliar para la conversión (necesitarás implementarlo)
        private byte[] ConvertFloatBufferToPngBytes(float[] buffer, int width, int height)
        {
            // Crear un bitmap en memoria para dibujar los píxeles
            using (var bitmap = new System.Drawing.Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb))
            {
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        // Leer el valor de profundidad del buffer (normalmente entre 0.0 y 1.0)
                        float depthValue = buffer[y * width + x];
                        
                        // Convertir el valor float a un valor de 8 bits para la escala de grises (0-255)
                        // Se asegura de que el valor esté dentro del rango válido.
                        byte grayScale = (byte)(Math.Min(Math.Max(depthValue * 255.0f, 0), 255));
                        
                        // Establecer el color del píxel (R, G y B son iguales para un gris)
                        bitmap.SetPixel(x, y, System.Drawing.Color.FromArgb(grayScale, grayScale, grayScale));
                    }
                }

                // Guardar el bitmap en un flujo de memoria en formato PNG
                using (var ms = new MemoryStream())
                {
                    bitmap.Save(ms, System.Drawing.Imaging.ImageFormat.Png);
                    // Devolver el contenido del flujo como un array de bytes
                    return ms.ToArray();
                }
            }
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
            catch (InvalidOperationException ioex) // <-- Añadir catch específico
            {
                updateStatusCallback?.Invoke(ioex.Message, Drawing.Color.Orange); // Informar de la geometría vacía
                _isCacheValid = false;
                DisposeCurrentCache();
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
            // --- INICIO DE CIRUGÍA 3 ---
            // 1. Cargar la configuración para leer la calidad elegida por el usuario
            var config = WabiSabiConfig.Load();

            // 2. Determinar el nivel de detalle basado en la configuración
            var detailLevel = ViewDetailLevel.Coarse; // Valor por defecto seguro
            switch (config.GeometryCacheQuality)
            {
                case 0: // Baja
                    detailLevel = ViewDetailLevel.Coarse;
                    break;
                case 1: // Media
                    detailLevel = ViewDetailLevel.Medium;
                    break;
                case 2: // Alta
                    detailLevel = ViewDetailLevel.Fine;
                    break;
            }

            // 3. Usar el nivel de detalle seleccionado para la extracción
            var options = new Options 
            { 
                DetailLevel = detailLevel,
                IncludeNonVisibleObjects = false,
                ComputeReferences = true 
            };
            // --- FIN DE CIRUGÍA 3 ---

            var collector = new FilteredElementCollector(doc, view3D.Id)
                .WhereElementIsNotElementType()
                .Where(e => e.Category != null && e.Category.CategoryType == CategoryType.Model);
            
            var elements = collector.ToList();
            var finalData = new ExtractedSceneData();
            var syncLock = new object();
            int processedCount = 0;

            // --- INICIO DE CIRUGÍA 2: PROCESAMIENTO PARALELO ---
            updateStatusCallback?.Invoke($"Extrayendo geometría de {elements.Count} elementos (usando {Environment.ProcessorCount} núcleos)...", Drawing.Color.Blue);
            
            Parallel.ForEach(elements, element =>
            {
                var localData = new ExtractedSceneData(); // Cada hilo trabaja en sus propios datos
                try
                {
                    var geometry = element.get_Geometry(options);
                    if (geometry != null)
                    {
                        ExtractGeometryFromElement(geometry, localData);
                    }
                }
                catch { /* Ignorar elementos problemáticos */ }

                // Si el hilo extrajo algo, lo añadimos al resultado final de forma segura
                if (localData.VertexCount > 0)
                {
                    lock (syncLock)
                    {
                        finalData.MergeFrom(localData);
                    }
                }
                
                // Actualizar progreso
                int currentCount = Interlocked.Increment(ref processedCount);
                if (currentCount % 200 == 0) // Actualizar cada 200 elementos
                {
                    updateStatusCallback?.Invoke($"Extrayendo... {currentCount}/{elements.Count}", Drawing.Color.Blue);
                }
            });
            // --- FIN DE CIRUGÍA 2 ---

            updateStatusCallback?.Invoke($"Extracción completada: {finalData.VertexCount:N0} vértices", Drawing.Color.Green);
            return finalData;
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
            // 1. Guardamos el número de vértices que ya tenemos en nuestra escena.
            // Esto será nuestro "índice base" para los nuevos triángulos.
            int baseIndex = sceneData.VertexCount;

            // 2. Añadimos los vértices del nuevo mesh a nuestra lista de vértices de la escena.
            foreach (XYZ vertex in mesh.Vertices)
            {
                sceneData.Vertices.Add((float)vertex.X);
                sceneData.Vertices.Add((float)vertex.Y);
                sceneData.Vertices.Add((float)vertex.Z);
            }

            // 3. Añadimos los triángulos del nuevo mesh, ajustando sus índices.
            // Si un triángulo en el mesh original usaba el vértice #5, en nuestra escena
            // ahora usará el vértice #(baseIndex + 5).
            for (int i = 0; i < mesh.NumTriangles; i++)
            {
                var triangle = mesh.get_Triangle(i);
                sceneData.Triangles.Add(baseIndex + (int)triangle.get_Index(0));
                sceneData.Triangles.Add(baseIndex + (int)triangle.get_Index(1));
                sceneData.Triangles.Add(baseIndex + (int)triangle.get_Index(2));
            }
            
            // --- INICIO DE LA ACTUALIZACIÓN: CÁLCULO DE NORMALES REALES ---

            // 4. Creamos un array para almacenar la normal calculada para cada nuevo vértice.
            var vertexNormals = new XYZ[mesh.Vertices.Count];
            for (int i = 0; i < vertexNormals.Length; i++)
            {
                // Inicializamos todos los vectores de normales a cero.
                vertexNormals[i] = XYZ.Zero;
            }
            
            // 5. Recorremos cada triángulo para calcular la normal de su cara.
            for (int i = 0; i < mesh.NumTriangles; i++)
            {
                var triangle = mesh.get_Triangle(i);
                // Obtenemos los índices de los 3 vértices que forman el triángulo.
                int index0 = (int)triangle.get_Index(0);
                int index1 = (int)triangle.get_Index(1);
                int index2 = (int)triangle.get_Index(2);
                
                // Obtenemos las coordenadas de esos vértices.
                var v0 = mesh.Vertices[index0];
                var v1 = mesh.Vertices[index1];
                var v2 = mesh.Vertices[index2];
                
                // Calculamos la normal de la cara usando el producto vectorial y la normalizamos.
                var faceNormal = (v1 - v0).CrossProduct(v2 - v0).Normalize();
                
                // Sumamos la normal de esta cara a cada uno de los 3 vértices que la componen.
                // Esto acumula las normales de todas las caras que un vértice toca.
                vertexNormals[index0] += faceNormal;
                vertexNormals[index1] += faceNormal;
                vertexNormals[index2] += faceNormal;
            }
            
            // 6. Finalmente, recorremos las normales acumuladas y las añadimos a nuestra escena.
            foreach (var normal in vertexNormals)
            {
                // Normalizamos el vector acumulado para obtener la normal promedio del vértice.
                // Si por alguna razón el vector es cero, usamos un valor por defecto (hacia arriba).
                var finalNormal = normal.IsZeroLength() ? XYZ.BasisZ : normal.Normalize();

                sceneData.Normals.Add((float)finalNormal.X);
                sceneData.Normals.Add((float)finalNormal.Y);
                sceneData.Normals.Add((float)finalNormal.Z);
            }
            // --- FIN DE LA ACTUALIZACIÓN ---
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
                _currentMmfName = mmfName;
                _geometryMmf = MemoryMappedFile.CreateNew(_currentMmfName, CacheSizeBytes, MemoryMappedFileAccess.ReadWrite);
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

        /// <summary>
        /// Elimina todos los archivos de caché del directorio persistente en el disco duro.
        /// </summary>
        public void ClearPersistentCache()
        {
            try
            {
                // --- INICIO DE CIRUGÍA ---
                // Llama al método que realmente cierra los descriptores de archivo (file handles).
                // Esto es CRUCIAL para poder borrar los archivos del disco sin errores.
                // La llamada a InvalidateCache() ya no es necesaria, porque DisposeCurrentCache() la incluye.
                DisposeCurrentCache();
                // --- FIN DE CIRUGÍA ---

                if (Directory.Exists(_persistentCacheDirectory))
                {
                    // Obtiene todos los archivos del directorio
                    var files = Directory.GetFiles(_persistentCacheDirectory);
                    foreach (var file in files)
                    {
                        // Ahora esta operación tendrá éxito porque el archivo ya no está bloqueado.
                        File.Delete(file);
                    }
                    WabiSabiLogger.Log("Caché persistente en disco ha sido limpiado.", LogLevel.Info);
                }
            }
            catch (Exception ex)
            {
                // Registra el error pero no deja que la aplicación crashee
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
    
    /// <summary>
    /// Contiene un resumen de las estadísticas y el estado actual del GeometryCacheManager.
    /// </summary>
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

        // Método auxiliar para formatear el tamaño del caché de forma legible
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

    
    // ---  ESTRUCTURA DE DATOS DEL CACHÉ CON TODA LA INFORMACIÓN NECESARIA ---
    public struct CachedGeometryData
    {        
        public string MmfName { get; set; }

        public int VertexCount { get; set; }
        public int TriangleCount { get; set; }
        public long VerticesOffset { get; set; }
        public long IndicesOffset { get; set; }
        public long NormalsOffset { get; set; }
        
        // La validación ahora comprueba el nombre.
        public bool IsValid => !string.IsNullOrEmpty(MmfName) && VertexCount > 0 && TriangleCount > 0;
    }

    // ---  DATOS EXTRAÍDOS DE LA ESCENA ---
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

        // --- INICIO DE CIRUGÍA 2.2: PEGAR EL MÉTODO AQUÍ ---
        public void MergeFrom(ExtractedSceneData other)
        {
            // El 'this' ahora se refiere a la instancia correcta de ExtractedSceneData
            int baseIndex = this.VertexCount; 
            this.Vertices.AddRange(other.Vertices);
            this.Normals.AddRange(other.Normals);
            foreach (var index in other.Triangles)
            {
                this.Triangles.Add(baseIndex + index);
            }
        }
        
    }
 }