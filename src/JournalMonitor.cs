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
            
            if (matches.Count == 0)
            {
                const int MAX_BUFFER_SIZE = 15 * 1024;
                if (_contentBuffer.Length > MAX_BUFFER_SIZE)
                {
                    WabiSabiLogger.LogDiagnostic("JournalMonitor", "Buffer de ruido limpiado para evitar sobrecarga.");
                    _contentBuffer.Clear();
                }
                return;
            }

            var lastMatch = matches[matches.Count - 1];
            try
            {
                // Los valores del journal vienen en orden Y, Z, X
                // Necesitamos reordenarlos a X, Y, Z
                
                var eyePos = new XYZ(
                    double.Parse(lastMatch.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture), // X = Value[3]
                    double.Parse(lastMatch.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture), // Y = Value[1]
                    double.Parse(lastMatch.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture)  // Z = Value[2]
                );
                
                var targetPos = new XYZ(
                    double.Parse(lastMatch.Groups[6].Value, System.Globalization.CultureInfo.InvariantCulture), // X = Value[6]
                    double.Parse(lastMatch.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture), // Y = Value[4]
                    double.Parse(lastMatch.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture)  // Z = Value[5]
                );

                var upDir = new XYZ(
                    double.Parse(lastMatch.Groups[9].Value, System.Globalization.CultureInfo.InvariantCulture), // X = Value[9]
                    double.Parse(lastMatch.Groups[7].Value, System.Globalization.CultureInfo.InvariantCulture), // Y = Value[7]
                    double.Parse(lastMatch.Groups[8].Value, System.Globalization.CultureInfo.InvariantCulture)  // Z = Value[8]
                );

                // Logging para verificar
                WabiSabiLogger.Log($"JournalMonitor - Reordered Eye: ({eyePos.X:F2}, {eyePos.Y:F2}, {eyePos.Z:F2})", LogLevel.Debug);
                WabiSabiLogger.Log($"JournalMonitor - Reordered Target: ({targetPos.X:F2}, {targetPos.Y:F2}, {targetPos.Z:F2})", LogLevel.Debug);
                WabiSabiLogger.Log($"JournalMonitor - Reordered Up: ({upDir.X:F2}, {upDir.Y:F2}, {upDir.Z:F2})", LogLevel.Debug);
                
                var viewDir = (targetPos - eyePos).Normalize();

                var cameraData = new CameraData
                {
                    EyePosition = eyePos,
                    ViewDirection = viewDir,
                    UpDirection = upDir,
                    SequenceNumber = Interlocked.Increment(ref _sequenceNumber),
                    Timestamp = System.Diagnostics.Stopwatch.GetTimestamp()
                };

                _cameraBuffer.TryWrite(cameraData);
                WabiSabiLogger.LogDiagnostic("JournalMonitor_Success", $"¡ÉXITO! Datos de cámara extraídos. Seq: {cameraData.SequenceNumber}");

                _contentBuffer.Remove(0, lastMatch.Index + lastMatch.Length);
            }
            catch (Exception ex)
            {
                WabiSabiLogger.LogError("Journal Monitor: Error al parsear datos de cámara del buffer.", ex);
                _contentBuffer.Clear();
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