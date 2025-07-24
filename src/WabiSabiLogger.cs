// WabiSabiLogger.cs - Sistema de logging robusto para WabiSabi Bridge
using System;
using System.IO;
using System.Text;
using System.Diagnostics;
using Autodesk.Revit.UI;
using TaskDialog = Autodesk.Revit.UI.TaskDialog;

namespace WabiSabiBridge
{
    /// <summary>
    /// Sistema de logging que escribe tanto a Debug como a archivo
    /// </summary>
    public static class WabiSabiLogger
    {
        private static readonly object _lockObject = new object();
        private static string _logFilePath;
        private static bool _isInitialized = false;
        
        static WabiSabiLogger()
        {
            Initialize();
        }
        
        private static void Initialize()
        {
            try
            {
                string logDir = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), "WabiSabiBridge", "Logs");
                if (!Directory.Exists(logDir))
                {
                    Directory.CreateDirectory(logDir);
                }
                
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                _logFilePath = Path.Combine(logDir, $"WabiSabi_Log_{timestamp}.txt");
                
                _isInitialized = true;
                
                // Escribir encabezado
                Log("========================================");
                Log($"WabiSabi Bridge Log - {DateTime.Now}");
                Log("========================================");
                Log($"Log file: {_logFilePath}");
                Log("");
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WabiSabiLogger] Error inicializando: {ex.Message}");
                _isInitialized = false;
            }
        }
        
        public static void Log(string message, LogLevel level = LogLevel.Info)
        {
            string timestamp = DateTime.Now.ToString("HH:mm:ss.fff");
            string levelStr = level.ToString().ToUpper().PadRight(7);
            string formattedMessage = $"[{timestamp}] [{levelStr}] {message}";
            
            // Siempre intentar escribir a Debug
            Debug.WriteLine(formattedMessage);
            
            // También escribir a archivo
            if (_isInitialized)
            {
                WriteToFile(formattedMessage);
            }
            
            // Para errores críticos, también mostrar TaskDialog
            if (level == LogLevel.Critical)
            {
                try
                {
                    TaskDialog.Show("WabiSabi Error", message);
                }
                catch { }
            }
        }
        
        public static void LogError(string message, Exception ex = null)
        {
            Log($"ERROR: {message}", LogLevel.Error);
            if (ex != null)
            {
                Log($"Exception: {ex.GetType().Name}: {ex.Message}", LogLevel.Error);
                Log($"StackTrace: {ex.StackTrace}", LogLevel.Error);
            }
        }
        
        public static void LogDiagnostic(string category, string message)
        {
            Log($"[{category}] {message}", LogLevel.Debug);
        }
        
        private static void WriteToFile(string message)
        {
            try
            {
                lock (_lockObject)
                {
                    File.AppendAllText(_logFilePath, message + Environment.NewLine);
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"[WabiSabiLogger] Error escribiendo a archivo: {ex.Message}");
            }
        }
        
        public static string GetLogFilePath()
        {
            return _logFilePath;
        }
        
        public static void ShowLogFile()
        {
            try
            {
                if (File.Exists(_logFilePath))
                {
                    Process.Start(new ProcessStartInfo
                    {
                        FileName = _logFilePath,
                        UseShellExecute = true
                    });
                }
                else
                {
                    TaskDialog.Show("WabiSabi", "No se encontró archivo de log.");
                }
            }
            catch (Exception ex)
            {
                TaskDialog.Show("Error", $"No se pudo abrir el archivo de log: {ex.Message}");
            }
        }
    }
    
    public enum LogLevel
    {
        Debug,
        Info,
        Warning,
        Error,
        Critical
    }
}