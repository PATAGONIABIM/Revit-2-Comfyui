// WabiSabiBridge.cs - Implementación MVP para Revit 2026
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using Newtonsoft.Json;

// Revit API
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;

// Windows Forms con alias para evitar conflictos
using WinForms = System.Windows.Forms;
using Drawing = System.Drawing;

namespace WabiSabiBridge
{
    /// <summary>
    /// Comando principal del plugin WabiSabi Bridge
    /// </summary>
    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    public class WabiSabiBridgeCommand : IExternalCommand
    {
        private static WabiSabiBridgeWindow _window;
        private static ExternalEvent _externalEvent;
        private static ExportEventHandler _eventHandler;
        
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            try
            {
                UIApplication uiApp = commandData.Application;
                UIDocument uiDoc = uiApp.ActiveUIDocument;
                Document doc = uiDoc.Document;
                
                // Verificar que hay una vista 3D activa
                View3D view3D = doc.ActiveView as View3D;
                if (view3D == null)
                {
                    TaskDialog.Show("WabiSabi Bridge", "Por favor, activa una vista 3D antes de ejecutar el comando.");
                    return Result.Failed;
                }
                
                // Crear el handler y evento si no existen
                if (_eventHandler == null)
                {
                    _eventHandler = new ExportEventHandler();
                    _externalEvent = ExternalEvent.Create(_eventHandler);
                }
                
                // Crear o mostrar la ventana principal
                if (_window == null || _window.IsDisposed)
                {
                    _window = new WabiSabiBridgeWindow(uiApp, _externalEvent, _eventHandler);
                    _window.Show();
                }
                else
                {
                    _window.Show();
                    _window.Focus();
                }
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                message = ex.Message;
                return Result.Failed;
            }
        }
    }
    
    /// <summary>
    /// Handler para ejecutar exportaciones en el contexto válido de Revit
    /// </summary>
    public class ExportEventHandler : IExternalEventHandler
    {
        public UIApplication UiApp { get; set; }
        public string OutputPath { get; set; }
        public Action<string, Drawing.Color> UpdateStatusCallback { get; set; }
        
        public void Execute(UIApplication app)
        {
            try
            {
                UiApp = app;
                Document doc = app.ActiveUIDocument.Document;
                View3D view3D = doc.ActiveView as View3D;
                
                if (view3D == null)
                {
                    UpdateStatusCallback?.Invoke("Error: No hay vista 3D activa", Drawing.Color.Red);
                    return;
                }
                
                // Crear directorio si no existe
                if (!Directory.Exists(OutputPath))
                {
                    Directory.CreateDirectory(OutputPath);
                }
                
                // Timestamp para archivos únicos
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                
                // 1. Exportar imagen de líneas ocultas
                ExportHiddenLineImage(doc, view3D, OutputPath, timestamp);
                
                // 2. Exportar metadatos básicos
                ExportMetadata(doc, view3D, OutputPath, timestamp);
                
                // 3. Crear archivo de notificación para ComfyUI
                CreateNotificationFile(OutputPath, timestamp);
                
                UpdateStatusCallback?.Invoke($"Exportado: {timestamp}", Drawing.Color.Green);
            }
            catch (Exception ex)
            {
                UpdateStatusCallback?.Invoke($"Error: {ex.Message}", Drawing.Color.Red);
            }
        }
        
        private void ExportHiddenLineImage(Document doc, View3D view3D, string outputPath, string timestamp)
        {
            // Configurar opciones de exportación
            ImageExportOptions options = new ImageExportOptions
            {
                FilePath = Path.Combine(outputPath, $"render_{timestamp}"),
                FitDirection = FitDirectionType.Horizontal,
                HLRandWFViewsFileType = ImageFileType.PNG,
                ImageResolution = ImageResolution.DPI_150,
                PixelSize = 1920,
                ExportRange = ExportRange.CurrentView
            };
            
            // Exportar usando transacción
            using (Transaction trans = new Transaction(doc, "Export Image"))
            {
                trans.Start();
                doc.ExportImage(options);
                trans.Commit();
            }
            
            // Renombrar archivo para consistencia
            string generatedFile = Path.Combine(outputPath, $"render_{timestamp}.png");
            string targetFile = Path.Combine(outputPath, "current_render.png");
            
            if (File.Exists(generatedFile))
            {
                File.Copy(generatedFile, targetFile, true);
            }
        }
        
        private void ExportMetadata(Document doc, View3D view3D, string outputPath, string timestamp)
        {
            var metadata = new
            {
                timestamp = timestamp,
                view_name = view3D.Name,
                view_type = "3D",
                scale = view3D.Scale,
                detail_level = view3D.DetailLevel.ToString(),
                display_style = view3D.DisplayStyle.ToString(),
                camera = new
                {
                    eye_position = new { x = 0, y = 0, z = 0 }, // TODO: Extraer posición real
                    target_position = new { x = 0, y = 0, z = 0 },
                    up_vector = new { x = 0, y = 1, z = 0 }
                },
                project_info = new
                {
                    name = doc.Title,
                    path = doc.PathName
                }
            };
            
            string metadataPath = Path.Combine(outputPath, "current_metadata.json");
            File.WriteAllText(metadataPath, JsonConvert.SerializeObject(metadata, Formatting.Indented));
        }
        
        private void CreateNotificationFile(string outputPath, string timestamp)
        {
            // Archivo que ComfyUI puede vigilar para saber que hay nuevos datos
            string notificationPath = Path.Combine(outputPath, "last_update.txt");
            File.WriteAllText(notificationPath, timestamp);
        }
        
        public string GetName()
        {
            return "WabiSabi Bridge Export Event";
        }
    }
    
    /// <summary>
    /// Aplicación principal del plugin
    /// </summary>
    public class WabiSabiBridgeApp : IExternalApplication
    {
        public Result OnStartup(UIControlledApplication application)
        {
            // Crear ribbon panel
            string tabName = "WabiSabi";
            application.CreateRibbonTab(tabName);
            
            RibbonPanel panel = application.CreateRibbonPanel(tabName, "Bridge");
            
            // Crear botón
            string thisAssemblyPath = System.Reflection.Assembly.GetExecutingAssembly().Location;
            PushButtonData buttonData = new PushButtonData(
                "WabiSabiBridge",
                "WabiSabi\nBridge",
                thisAssemblyPath,
                "WabiSabiBridge.WabiSabiBridgeCommand");
            
            buttonData.ToolTip = "Exportar vista actual a ComfyUI";
            buttonData.LongDescription = "Extrae imágenes y datos de la vista 3D activa para usar en ComfyUI";
            
            PushButton button = panel.AddItem(buttonData) as PushButton;
            
            return Result.Succeeded;
        }
        
        public Result OnShutdown(UIControlledApplication application)
        {
            return Result.Succeeded;
        }
    }
    
    /// <summary>
    /// Ventana principal del plugin (MVP)
    /// </summary>
    public class WabiSabiBridgeWindow : WinForms.Form
    {
        private UIApplication _uiApp;
        private ExternalEvent _externalEvent;
        private ExportEventHandler _eventHandler;
        private WinForms.Button _exportButton;
        private WinForms.TextBox _outputPathTextBox;
        private WinForms.Button _browseButton;
        private WinForms.Label _statusLabel;
        private WinForms.CheckBox _autoExportCheckBox;
        private WinForms.Timer _autoExportTimer;
        
        // Configuración
        private WabiSabiConfig _config;
        
        public WabiSabiBridgeWindow(UIApplication uiApp, ExternalEvent externalEvent, ExportEventHandler eventHandler)
        {
            _uiApp = uiApp;
            _externalEvent = externalEvent;
            _eventHandler = eventHandler;
            _eventHandler.UpdateStatusCallback = UpdateStatus;
            _config = WabiSabiConfig.Load();
            InitializeUI();
        }
        
        private void InitializeUI()
        {
            // Configuración de la ventana
            Text = "WabiSabi Bridge - MVP";
            Size = new Drawing.Size(400, 250);
            StartPosition = WinForms.FormStartPosition.CenterScreen;
            FormBorderStyle = WinForms.FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            
            // Layout principal
            WinForms.TableLayoutPanel mainLayout = new WinForms.TableLayoutPanel
            {
                Dock = WinForms.DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 5,
                Padding = new WinForms.Padding(10)
            };
            
            // Fila 1: Ruta de salida
            WinForms.Panel pathPanel = new WinForms.Panel { Height = 30, Dock = WinForms.DockStyle.Fill };
            WinForms.Label pathLabel = new WinForms.Label { Text = "Carpeta de salida:", Width = 100, TextAlign = Drawing.ContentAlignment.MiddleLeft };
            _outputPathTextBox = new WinForms.TextBox 
            { 
                Text = _config.OutputPath,
                Left = 105, 
                Width = 200,
                Top = 3
            };
            _browseButton = new WinForms.Button 
            { 
                Text = "...", 
                Left = 310, 
                Width = 30,
                Top = 2
            };
            _browseButton.Click += BrowseButton_Click;
            
            pathPanel.Controls.Add(pathLabel);
            pathPanel.Controls.Add(_outputPathTextBox);
            pathPanel.Controls.Add(_browseButton);
            
            // Fila 2: Botón de exportación
            _exportButton = new WinForms.Button
            {
                Text = "Exportar Vista Actual",
                Height = 40,
                Dock = WinForms.DockStyle.Fill,
                Font = new Drawing.Font(Font.FontFamily, 10, Drawing.FontStyle.Bold)
            };
            _exportButton.Click += ExportButton_Click;
            
            // Fila 3: Auto-exportación
            _autoExportCheckBox = new WinForms.CheckBox
            {
                Text = "Exportación automática (experimental)",
                Dock = WinForms.DockStyle.Fill,
                Checked = _config.AutoExport
            };
            _autoExportCheckBox.CheckedChanged += AutoExportCheckBox_CheckedChanged;
            
            // Fila 4: Estado
            _statusLabel = new WinForms.Label
            {
                Text = "Listo para exportar",
                Dock = WinForms.DockStyle.Fill,
                TextAlign = Drawing.ContentAlignment.MiddleCenter,
                ForeColor = Drawing.Color.Green,
                Font = new Drawing.Font(Font.FontFamily, 9, Drawing.FontStyle.Italic)
            };
            
            // Agregar controles al layout
            mainLayout.Controls.Add(pathPanel, 0, 0);
            mainLayout.Controls.Add(_exportButton, 0, 1);
            mainLayout.Controls.Add(_autoExportCheckBox, 0, 2);
            mainLayout.Controls.Add(_statusLabel, 0, 3);
            
            Controls.Add(mainLayout);
            
            // Timer para auto-exportación
            _autoExportTimer = new WinForms.Timer();
            _autoExportTimer.Interval = 2000; // 2 segundos
            _autoExportTimer.Tick += AutoExportTimer_Tick;
            
            if (_config.AutoExport)
            {
                _autoExportTimer.Start();
            }
        }
        
        private void BrowseButton_Click(object sender, EventArgs e)
        {
            using (WinForms.FolderBrowserDialog dialog = new WinForms.FolderBrowserDialog())
            {
                dialog.SelectedPath = _outputPathTextBox.Text;
                if (dialog.ShowDialog() == WinForms.DialogResult.OK)
                {
                    _outputPathTextBox.Text = dialog.SelectedPath;
                    _config.OutputPath = dialog.SelectedPath;
                    _config.Save();
                }
            }
        }
        
        private void ExportButton_Click(object sender, EventArgs e)
        {
            ExportCurrentView();
        }
        
        private void AutoExportCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            _config.AutoExport = _autoExportCheckBox.Checked;
            _config.Save();
            
            if (_autoExportCheckBox.Checked)
            {
                _autoExportTimer.Start();
            }
            else
            {
                _autoExportTimer.Stop();
            }
        }
        
        private void AutoExportTimer_Tick(object sender, EventArgs e)
        {
            // Verificar si la vista ha cambiado antes de exportar
            if (HasViewChanged())
            {
                ExportCurrentView();
            }
        }
        
        private bool HasViewChanged()
        {
            // MVP: Siempre retorna true para exportar cada vez
            // TODO: Implementar detección real de cambios
            return true;
        }
        
        private void ExportCurrentView()
        {
            try
            {
                UpdateStatus("Preparando exportación...", Drawing.Color.Blue);
                
                // Configurar el handler con la ruta actual
                _eventHandler.OutputPath = _outputPathTextBox.Text;
                _eventHandler.UiApp = _uiApp;
                
                // Ejecutar la exportación a través del ExternalEvent
                _externalEvent.Raise();
            }
            catch (Exception ex)
            {
                UpdateStatus($"Error: {ex.Message}", Drawing.Color.Red);
            }
        }
        
        private void UpdateStatus(string message, Drawing.Color color)
        {
            if (InvokeRequired)
            {
                Invoke(new Action(() => UpdateStatus(message, color)));
                return;
            }
            
            _statusLabel.Text = message;
            _statusLabel.ForeColor = color;
        }
        
        protected override void OnFormClosing(WinForms.FormClosingEventArgs e)
        {
            _autoExportTimer?.Stop();
            _autoExportTimer?.Dispose();
            base.OnFormClosing(e);
        }
    }
    
    /// <summary>
    /// Configuración del plugin
    /// </summary>
    public class WabiSabiConfig
    {
        public string OutputPath { get; set; } = Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments), 
            "WabiSabiBridge"
        );
        
        public bool AutoExport { get; set; } = false;
        
        private static string ConfigPath => Path.Combine(
            Environment.GetFolderPath(Environment.SpecialFolder.ApplicationData),
            "WabiSabiBridge",
            "config.json"
        );
        
        public static WabiSabiConfig Load()
        {
            try
            {
                if (File.Exists(ConfigPath))
                {
                    string json = File.ReadAllText(ConfigPath);
                    return JsonConvert.DeserializeObject<WabiSabiConfig>(json);
                }
            }
            catch { }
            
            return new WabiSabiConfig();
        }
        
        public void Save()
        {
            try
            {
                string dir = Path.GetDirectoryName(ConfigPath);
                if (!Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                
                string json = JsonConvert.SerializeObject(this, Formatting.Indented);
                File.WriteAllText(ConfigPath, json);
            }
            catch { }
        }
    }
}