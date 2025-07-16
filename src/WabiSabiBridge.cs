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

// Extractores
using WabiSabiBridge.Extractors;

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
        public bool ExportDepth { get; set; }
        public int DepthResolution { get; set; } = 512;
        public int DepthQuality { get; set; } = 1; // 0=Rápida, 1=Normal, 2=Alta
        public Action<string, Drawing.Color> UpdateStatusCallback { get; set; }
        
        public void Execute(UIApplication app)
        {
            try
            {
                UiApp = app;
                UIDocument uiDoc = app.ActiveUIDocument;
                Document doc = uiDoc.Document;
                View3D view3D = doc.ActiveView as View3D;
                
                if (view3D == null)
                {
                    UpdateStatusCallback?.Invoke("Error: No hay vista 3D activa", Drawing.Color.Red);
                    return;
                }

                UIView uiView = uiDoc.GetOpenUIViews().FirstOrDefault(v => v.ViewId == view3D.Id);
                if (uiView == null)
                {
                    UpdateStatusCallback?.Invoke("Error: No se pudo obtener la ventana de la vista para el encuadre.", Drawing.Color.Red);
                    return;
                }

                // <-- CAMBIO: Obtener las esquinas de la vista para garantizar un encuadre perfecto.
                IList<XYZ> viewCorners = uiView.GetZoomCorners();

                var viewRect = uiView.GetWindowRectangle();
                double viewWidth = viewRect.Right - viewRect.Left;
                double viewHeight = viewRect.Bottom - viewRect.Top;

                if (viewWidth <= 0 || viewHeight <= 0)
                {
                    UpdateStatusCallback?.Invoke("Advertencia: Dimensiones de vista inválidas. Usando proporción por defecto 16:9.", Drawing.Color.Orange);
                    viewWidth = 16;
                    viewHeight = 9;
                }
                
                double aspectRatio = viewWidth / viewHeight;

                int depthWidth = this.DepthResolution;
                int depthHeight = (int)Math.Round(depthWidth / aspectRatio);
                
                if (!Directory.Exists(OutputPath))
                {
                    Directory.CreateDirectory(OutputPath);
                }
                
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                
                UpdateStatusCallback?.Invoke("Exportando vista...", Drawing.Color.Blue);
                ExportHiddenLineImage(doc, view3D, OutputPath, timestamp);
                
                if (ExportDepth)
                {
                    string depthStatus = DepthQuality == 0 ? "Generando mapa de profundidad (modo rápido)..." :
                                       DepthQuality == 2 ? "Generando mapa de profundidad (alta calidad)..." :
                                       "Generando mapa de profundidad...";
                    UpdateStatusCallback?.Invoke(depthStatus, Drawing.Color.Blue);
                    
                    try
                    {
                        if (DepthQuality == 0) // Rápida
                        {
                            var depthExtractor = new DepthExtractorFast(app, DepthResolution, 4);
                            // <-- CAMBIO: Pasar las esquinas de la vista al extractor
                            depthExtractor.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                        }
                        else if (DepthQuality == 2) // Alta
                        {
                            var depthExtractor = new DepthExtractor(app, DepthResolution);
                            // <-- CAMBIO: Pasar las esquinas de la vista al extractor
                            depthExtractor.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                        }
                        else // Normal
                        {
                            var depthExtractor = new DepthExtractorFast(app, DepthResolution, 2);
                            // <-- CAMBIO: Pasar las esquinas de la vista al extractor
                            depthExtractor.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                        }
                    }
                    catch (Exception ex)
                    {
                        UpdateStatusCallback?.Invoke($"Advertencia: Error en profundidad - {ex.Message}", Drawing.Color.Orange);
                    }
                }
                
                ExportMetadata(doc, view3D, OutputPath, timestamp);
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
            ImageExportOptions options = new ImageExportOptions
            {
                FilePath = Path.Combine(outputPath, $"render_{timestamp}"),
                FitDirection = FitDirectionType.Horizontal,
                HLRandWFViewsFileType = ImageFileType.PNG,
                ImageResolution = ImageResolution.DPI_150,
                PixelSize = 1920,
                ExportRange = ExportRange.CurrentView
            };
            
            using (Transaction trans = new Transaction(doc, "Export Image"))
            {
                trans.Start();
                doc.ExportImage(options);
                trans.Commit();
            }
            
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
                    eye_position = new { x = 0, y = 0, z = 0 },
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
        private WinForms.CheckBox _exportDepthCheckBox;
        private WinForms.ComboBox _depthResolutionCombo;
        private WinForms.ComboBox _depthQualityCombo;
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
            Text = "WabiSabi Bridge - v0.2.1";
            Size = new Drawing.Size(400, 340);
            StartPosition = WinForms.FormStartPosition.CenterScreen;
            FormBorderStyle = WinForms.FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            
            // Layout principal
            WinForms.TableLayoutPanel mainLayout = new WinForms.TableLayoutPanel
            {
                Dock = WinForms.DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 7,
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
            
            // Fila 3: Opciones de exportación
            WinForms.GroupBox exportOptionsGroup = new WinForms.GroupBox
            {
                Text = "Opciones de exportación",
                Height = 105,
                Dock = WinForms.DockStyle.Fill
            };
            
            // Checkbox para mapa de profundidad
            _exportDepthCheckBox = new WinForms.CheckBox
            {
                Text = "Generar mapa de profundidad",
                Location = new Drawing.Point(10, 20),
                Width = 200,
                Checked = _config.ExportDepth
            };
            _exportDepthCheckBox.CheckedChanged += ExportDepthCheckBox_CheckedChanged;
            
            // ComboBox para resolución de profundidad
            WinForms.Label depthResLabel = new WinForms.Label 
            { 
                Text = "Resolución:", 
                Location = new Drawing.Point(30, 45),
                Width = 70
            };
            
            _depthResolutionCombo = new WinForms.ComboBox
            {
                Location = new Drawing.Point(100, 43),
                Width = 100,
                DropDownStyle = WinForms.ComboBoxStyle.DropDownList,
                Enabled = _config.ExportDepth
            };
            _depthResolutionCombo.Items.AddRange(new object[] { "256", "512", "1024", "2048" });
            _depthResolutionCombo.SelectedItem = _config.DepthResolution.ToString();
            _depthResolutionCombo.SelectedIndexChanged += DepthResolutionCombo_SelectedIndexChanged;
            
            // ComboBox para calidad
            WinForms.Label depthQualityLabel = new WinForms.Label 
            { 
                Text = "Calidad:", 
                Location = new Drawing.Point(220, 45),
                Width = 50
            };
            
            _depthQualityCombo = new WinForms.ComboBox
            {
                Location = new Drawing.Point(270, 43),
                Width = 80,
                DropDownStyle = WinForms.ComboBoxStyle.DropDownList,
                Enabled = _config.ExportDepth
            };
            _depthQualityCombo.Items.AddRange(new object[] { "Rápida", "Normal", "Alta" });
            _depthQualityCombo.SelectedIndex = _config.DepthQuality;
            _depthQualityCombo.SelectedIndexChanged += DepthQualityCombo_SelectedIndexChanged;
            
            // Label de tiempo estimado
            WinForms.Label timeEstimateLabel = new WinForms.Label
            {
                Name = "timeEstimateLabel",
                Text = GetTimeEstimate(),
                Location = new Drawing.Point(30, 70),
                Width = 320,
                ForeColor = Drawing.Color.Gray,
                Font = new Drawing.Font(Font.FontFamily, 8)
            };
            
            exportOptionsGroup.Controls.Add(_exportDepthCheckBox);
            exportOptionsGroup.Controls.Add(depthResLabel);
            exportOptionsGroup.Controls.Add(_depthResolutionCombo);
            exportOptionsGroup.Controls.Add(depthQualityLabel);
            exportOptionsGroup.Controls.Add(_depthQualityCombo);
            exportOptionsGroup.Controls.Add(timeEstimateLabel);
            
            // Fila 4: Auto-exportación
            _autoExportCheckBox = new WinForms.CheckBox
            {
                Text = "Exportación automática (experimental)",
                Dock = WinForms.DockStyle.Fill,
                Checked = _config.AutoExport
            };
            _autoExportCheckBox.CheckedChanged += AutoExportCheckBox_CheckedChanged;
            
            // Fila 5: Estado
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
            mainLayout.Controls.Add(exportOptionsGroup, 0, 2);
            mainLayout.Controls.Add(_autoExportCheckBox, 0, 3);
            mainLayout.Controls.Add(_statusLabel, 0, 4);
            
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
        
        private void ExportDepthCheckBox_CheckedChanged(object sender, EventArgs e)
        {
            _config.ExportDepth = _exportDepthCheckBox.Checked;
            _depthResolutionCombo.Enabled = _exportDepthCheckBox.Checked;
            _depthQualityCombo.Enabled = _exportDepthCheckBox.Checked;
            _config.Save();
            UpdateTimeEstimate();
        }
        
        private void DepthResolutionCombo_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (int.TryParse(_depthResolutionCombo.SelectedItem.ToString(), out int resolution))
            {
                _config.DepthResolution = resolution;
                _config.Save();
                UpdateTimeEstimate();
            }
        }
        
        private void DepthQualityCombo_SelectedIndexChanged(object sender, EventArgs e)
        {
            _config.DepthQuality = _depthQualityCombo.SelectedIndex;
            _config.Save();
            UpdateTimeEstimate();
        }
        
        private void UpdateTimeEstimate()
        {
            var timeLabel = Controls.Find("timeEstimateLabel", true).FirstOrDefault() as WinForms.Label;
            if (timeLabel != null)
            {
                timeLabel.Text = GetTimeEstimate();
            }
        }
        
        private string GetTimeEstimate()
        {
            if (!_exportDepthCheckBox.Checked)
                return "";
                
            int resolution = _config.DepthResolution;
            int quality = _config.DepthQuality;
            
            // Estimaciones basadas en pruebas
            int[,] timeMatrix = new int[,] {
                // Rápida, Normal, Alta
                { 1, 3, 5 },      // 256
                { 3, 10, 20 },    // 512
                { 10, 40, 80 },   // 1024
                { 40, 160, 320 }  // 2048
            };
            
            int resIndex = resolution == 256 ? 0 : resolution == 512 ? 1 : resolution == 1024 ? 2 : 3;
            int seconds = timeMatrix[resIndex, quality];
            
            if (seconds < 60)
                return $"Tiempo estimado: ~{seconds} segundos";
            else
                return $"Tiempo estimado: ~{seconds / 60} minutos";
        }
        
        private void ExportCurrentView()
        {
            try
            {
                UpdateStatus("Preparando exportación...", Drawing.Color.Blue);
                
                // Configurar el handler con la ruta actual
                _eventHandler.OutputPath = _outputPathTextBox.Text;
                _eventHandler.UiApp = _uiApp;
                _eventHandler.ExportDepth = _exportDepthCheckBox.Checked;
                _eventHandler.DepthResolution = _config.DepthResolution;
                _eventHandler.DepthQuality = _config.DepthQuality;
                
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
        public bool ExportDepth { get; set; } = false;
        public int DepthResolution { get; set; } = 512;
        public int DepthQuality { get; set; } = 1; // 0=Rápida, 1=Normal, 2=Alta
        
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

// Namespace para extractores
namespace WabiSabiBridge.Extractors
{
    // Los extractores van aquí
}