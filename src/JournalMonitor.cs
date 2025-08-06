// EN UN NUEVO ARCHIVO: JournalMonitor.cs
using Autodesk.Revit.DB;
using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using WabiSabiBridge.DirectContext;

namespace WabiSabiBridge
{
    public class JournalMonitor
    {
        private readonly LockFreeCameraRingBuffer _cameraBuffer;
        private CancellationTokenSource? _cts;
        private Task? _monitoringTask;
        private static readonly Regex _cameraDirectiveRegex = new Regex(
            // Anclaje 1: Busca "Jrn.Directive"
            @"Jrn\.Directive" + 
            // Comodín: Ignora CUALQUIER caracter (incluidos saltos de línea) hasta encontrar el siguiente anclaje
            @".*?" + 
            // Anclaje 2: Busca ""AutoCamCamera""
            @"""AutoCamCamera""" + 
            // Comodín: Ignora CUALQUIER caracter hasta encontrar la primera coma de los datos
            @".*?" + 
            // Ahora, captura los 9 números, siendo muy tolerante con los espacios y saltos de línea
            @",\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*_\s*" + // Grupo 1,2,3 (Eye)
            @",\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*_\s*" + // Grupo 4,5,6 (Target)
            @",\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*,\s*([-.\dE]+)",          // Grupo 7,8,9 (Up)
            RegexOptions.Compiled | RegexOptions.Singleline);

        private int _sequenceNumber = 0;
        private readonly StringBuilder _contentBuffer = new StringBuilder();
        public JournalMonitor(LockFreeCameraRingBuffer cameraBuffer)
        {
            _cameraBuffer = cameraBuffer;
        }

        public void Start()
        {
            // Opcional pero recomendado: un guardia para evitar iniciar si ya está corriendo.
            if (_monitoringTask != null && !_monitoringTask.IsCompleted)
            {
                WabiSabiLogger.Log("Journal Monitor: Start() llamado pero la tarea ya está corriendo.", LogLevel.Debug);
                return;
            }
            
            // El método Start ahora solo crea y empieza.
            _cts = new CancellationTokenSource();
            var token = _cts.Token;

            _monitoringTask = Task.Run(() => MonitorLoop(token), token);
        }

        public void Stop()
        {
            _cts?.Cancel();
            _monitoringTask?.Wait(TimeSpan.FromSeconds(1));
            _cts?.Dispose();
        }

        private async Task MonitorLoop(CancellationToken token)
        {
            WabiSabiLogger.Log("Journal Monitor: Iniciando hilo de monitorización.", LogLevel.Info);
            string? journalPath = FindLatestJournalFile();

            if (string.IsNullOrEmpty(journalPath))
            {
                WabiSabiLogger.LogError("Journal Monitor: No se pudo encontrar el archivo journal. El hilo se detendrá.");
                return;
            }
            
            WabiSabiLogger.Log($"Journal Monitor: Monitorizando archivo: {journalPath}", LogLevel.Info);

            long lastPosition = new FileInfo(journalPath).Length; // Empezar desde el final del archivo

            while (!token.IsCancellationRequested)
            {
                try
                {
                    long currentLength = new FileInfo(journalPath).Length;
                    if (currentLength > lastPosition)
                    {
                        using (var fs = new FileStream(journalPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
                        {
                            fs.Seek(lastPosition, SeekOrigin.Begin);
                            using (var reader = new StreamReader(fs))
                            {
                                string newContent = await reader.ReadToEndAsync();
                                
                                // --- INICIO DE CIRUGÍA 2.1: USAR EL BUFFER ---
                                // En lugar de procesar directamente, añadimos al buffer
                                _contentBuffer.Append(newContent);
                                // --- FIN DE CIRUGÍA 2.1 ---
                            }
                        }
                        lastPosition = currentLength;
                    }
                    
                    // --- INICIO DE CIRUGÍA 2.2: PROCESAR EL BUFFER ACUMULADO ---
                    // Solo procesamos si hay algo en el buffer
                    if (_contentBuffer.Length > 0)
                    {
                        ProcessContentBuffer();
                    }
                    // --- FIN DE CIRUGÍA 2.2 ---

                    await Task.Delay(100, token);
                }
                catch (TaskCanceledException) { break; }
                catch (Exception ex)
                {
                    WabiSabiLogger.LogError("Journal Monitor: Error en el bucle de monitorización.", ex);
                    await Task.Delay(1000, token);
                }
            }
            WabiSabiLogger.Log("Journal Monitor: Hilo de monitorización detenido.", LogLevel.Info);
        }

        // --- CREAR MÉTODO PARA PROCESAR EL BUFFER ---
        private void ProcessContentBuffer()
        {
            string contentToProcess = _contentBuffer.ToString();
            
            WabiSabiLogger.LogDiagnostic("JournalMonitor_Buffer", $"Analizando buffer ({contentToProcess.Length} caracteres).");

            var matches = _cameraDirectiveRegex.Matches(contentToProcess);
            
            // Si no se encuentra ninguna directiva de cámara, podemos limpiar el buffer
            // si crece demasiado para evitar consumir memoria con "ruido" del journal.
            if (matches.Count == 0)
            {
                const int MAX_BUFFER_SIZE = 15 * 1024; // 15 KB
                if (_contentBuffer.Length > MAX_BUFFER_SIZE)
                {
                    WabiSabiLogger.LogDiagnostic("JournalMonitor", "Buffer de ruido limpiado para evitar sobrecarga.");
                    _contentBuffer.Clear();
                }
                return;
            }

            // Solo nos interesa la última posición de la cámara para tener la más reciente.
            var lastMatch = matches[matches.Count - 1];
            try
            {
                // --- INICIO DE LA CORRECCIÓN CRÍTICA ---
                // El journal guarda las coordenadas en el orden: Y, Z, X
                // El constructor de Revit XYZ es: new XYZ(X, Y, Z)
                // Por lo tanto, debemos mapear los grupos de la Regex correctamente.
                //
                // Grupos de la Regex para Eye Position:
                // Group[1] -> Y
                // Group[2] -> Z
                // Group[3] -> X
                
                var eyePos = new XYZ(
                    double.Parse(lastMatch.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture), // Asignar Group[3] (X del journal) a la X de Revit
                    double.Parse(lastMatch.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture), // Asignar Group[1] (Y del journal) a la Y de Revit
                    double.Parse(lastMatch.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture)  // Asignar Group[2] (Z del journal) a la Z de Revit
                );
                
                // Grupos para Target Position:
                // Group[4] -> Y
                // Group[5] -> Z
                // Group[6] -> X
                var targetPos = new XYZ(
                    double.Parse(lastMatch.Groups[6].Value, System.Globalization.CultureInfo.InvariantCulture), // X
                    double.Parse(lastMatch.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture), // Y
                    double.Parse(lastMatch.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture)  // Z
                );

                // Grupos para Up Direction:
                // Group[7] -> Y
                // Group[8] -> Z
                // Group[9] -> X
                var upDir = new XYZ(
                    double.Parse(lastMatch.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture), // X
                    double.Parse(lastMatch.Groups[7].Value, System.Globalization.CultureInfo.InvariantCulture), // Y
                    double.Parse(lastMatch.Groups[8].Value, System.Globalization.CultureInfo.InvariantCulture)  // Z
                );

                // --- FIN DE LA CORRECCIÓN CRÍTICA ---

                // El resto de la lógica para calcular la dirección y encolar los datos es correcta.
                var viewDir = (targetPos - eyePos).Normalize();

                var cameraData = new CameraData
                {
                    EyePosition = eyePos,
                    ViewDirection = viewDir,
                    UpDirection = upDir,
                    SequenceNumber = Interlocked.Increment(ref _sequenceNumber),
                    Timestamp = System.Diagnostics.Stopwatch.GetTimestamp()
                };

                if (_cameraBuffer.TryWrite(cameraData))
                {
                    WabiSabiLogger.LogDiagnostic("JournalMonitor_Success", $"¡ÉXITO! Datos de cámara extraídos y encolados. Seq: {cameraData.SequenceNumber}");
                }

                // Limpiamos el buffer hasta el final de lo que acabamos de procesar
                // para no volver a analizarlo.
                _contentBuffer.Remove(0, lastMatch.Index + lastMatch.Length);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Journal Monitor: Error al parsear datos de cámara del buffer. El buffer será limpiado.", ex);
                _contentBuffer.Clear(); // Limpiar en caso de error para evitar bucles infinitos.
            }
        }

        private void ProcessNewContent(string content)
        {
            var matches = _cameraDirectiveRegex.Matches(content);
            if (matches.Count == 0) return;

            // Solo procesamos el último match para no saturar el buffer
            var lastMatch = matches[matches.Count - 1];
            try
            {
                var eyePos = new XYZ(
                    double.Parse(lastMatch.Groups[1].Value),
                    double.Parse(lastMatch.Groups[2].Value),
                    double.Parse(lastMatch.Groups[3].Value));
                
                var targetPos = new XYZ(
                    double.Parse(lastMatch.Groups[4].Value),
                    double.Parse(lastMatch.Groups[5].Value),
                    double.Parse(lastMatch.Groups[6].Value));

                var upDir = new XYZ(
                    double.Parse(lastMatch.Groups[7].Value),
                    double.Parse(lastMatch.Groups[8].Value),
                    double.Parse(lastMatch.Groups[9].Value));
                    
                var viewDir = (targetPos - eyePos).Normalize();

                var cameraData = new CameraData
                {
                    EyePosition = eyePos,
                    ViewDirection = viewDir,
                    UpDirection = upDir,
                    SequenceNumber = Interlocked.Increment(ref WabiSabiBridgeApp._globalSequenceNumber),
                    Timestamp = System.Diagnostics.Stopwatch.GetTimestamp()
                };

                _cameraBuffer.TryWrite(cameraData);
                WabiSabiLogger.LogDiagnostic("JournalMonitor", $"Datos de cámara extraídos del journal. Seq: {cameraData.SequenceNumber}");
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Journal Monitor: Error al parsear datos de cámara.", ex);
            }
        }

        private static string? FindLatestJournalFile()
        {
            try
            {
                string journalDir = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                    @"Autodesk\Revit",
                    $"Autodesk Revit {WabiSabiBridgeApp.RevitVersion}", // Necesitaremos la versión de Revit
                    "Journals");

                if (!Directory.Exists(journalDir)) return null;

                var directory = new DirectoryInfo(journalDir);
                var latestJournal = directory.GetFiles("journal.*.txt")
                                             .OrderByDescending(f => f.LastWriteTime)
                                             .FirstOrDefault();
                return latestJournal?.FullName;
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Error buscando archivo journal.", ex);
                return null;
            }
        }
    }
}