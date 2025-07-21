// WabiSabiBridge.cs - Implementación con aceleración GPU v0.3.0
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
using WabiSabiBridge.Extractors.Gpu; // <<-- AÑADIDO para resolver ambigüedades

namespace WabiSabiBridge
{
    /// <summary>
    /// Comando principal del plugin WabiSabi Bridge
    /// </summary>
    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    public class WabiSabiBridgeCommand : IExternalCommand
    {
        private static WabiSabiBridgeWindow? _window;
        private static ExternalEvent? _externalEvent;
        private static ExportEventHandler? _eventHandler;
        
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            try
            {
                UIApplication uiApp = commandData.Application;
                UIDocument uiDoc = uiApp.ActiveUIDocument;
                Document doc = uiDoc.Document;
                
                View3D? view3D = doc.ActiveView as View3D;
                if (view3D == null)
                {
                    Autodesk.Revit.UI.TaskDialog.Show("WabiSabi Bridge", "Por favor, activa una vista 3D antes de ejecutar el comando.");
                    return Result.Failed;
                }
                
                if (_eventHandler == null)
                {
                    _eventHandler = new ExportEventHandler();
                    _externalEvent = ExternalEvent.Create(_eventHandler);
                }
                
                if (_window == null || _window.IsDisposed)
                {
                    // CORREGIDO: Usar el operador "null-forgiving" (!) porque sabemos que no son nulos aquí.
                    _window = new WabiSabiBridgeWindow(uiApp, _externalEvent!, _eventHandler!);
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
        public UIApplication? UiApp { get; set; }
        public string OutputPath { get; set; } = string.Empty;
        public bool ExportDepth { get; set; }
        public int DepthResolution { get; set; } = 512;
        public int DepthQuality { get; set; } = 1; // 0=Rápida, 1=Normal, 2=Alta
        public bool AutoDepthRange { get; set; } = true;
        public double DepthRangeDistance { get; set; } = 50.0;
        public bool UseGpuAcceleration { get; set; } = true;
        public bool UseGeometryExtraction { get; set; } = false;
        public Action<string, Drawing.Color>? UpdateStatusCallback { get; set; }
        
        private DepthExtractor? _depthExtractor;
        private DepthExtractorFast? _depthExtractorFast;
        
        public void Execute(UIApplication app)
        {
            try
            {
                UiApp = app;
                UIDocument uiDoc = app.ActiveUIDocument;
                Document doc = uiDoc.Document;
                View3D? view3D = doc.ActiveView as View3D;
                
                if (view3D == null)
                {
                    UpdateStatusCallback?.Invoke("Error: No hay vista 3D activa", Drawing.Color.Red);
                    return;
                }

                UIView? uiView = uiDoc.GetOpenUIViews().FirstOrDefault(v => v.ViewId == view3D.Id);
                if (uiView == null)
                {
                    UpdateStatusCallback?.Invoke("Error: No se pudo obtener la ventana de la vista para el encuadre.", Drawing.Color.Red);
                    return;
                }

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
                
                double rawDepthHeight = depthWidth / aspectRatio;
                int depthHeight = (int)(Math.Round(rawDepthHeight / 2.0) * 2.0);
                
                if (!Directory.Exists(OutputPath))
                {
                    Directory.CreateDirectory(OutputPath);
                }
                
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                
                UpdateStatusCallback?.Invoke("Exportando vista...", Drawing.Color.Blue);
                ExportHiddenLineImage(doc, view3D, OutputPath, timestamp);
                
                if (ExportDepth)
                {
                    string gpuStatus = UseGpuAcceleration ? " (GPU)" : "";
                    string depthStatus = DepthQuality == 0 ? $"Generando mapa de profundidad (modo rápido{gpuStatus})..." :
                                       DepthQuality == 2 ? $"Generando mapa de profundidad (alta calidad{gpuStatus})..." :
                                       $"Generando mapa de profundidad{gpuStatus}...";
                    UpdateStatusCallback?.Invoke(depthStatus, Drawing.Color.Blue);
                    
                    try
                    {
                        if (DepthQuality == 0) // Rápida
                        {
                            _depthExtractorFast ??= new DepthExtractorFast(app, DepthResolution, 4);
                            _depthExtractorFast.AutoDepthRange = this.AutoDepthRange;
                            _depthExtractorFast.ManualDepthDistance = this.DepthRangeDistance;
                            _depthExtractorFast.UseGpuAcceleration = this.UseGpuAcceleration;
                            _depthExtractorFast.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                        }
                        else if (DepthQuality == 2) // Alta
                        {
                            _depthExtractor ??= new DepthExtractor(app, DepthResolution);
                            _depthExtractor.AutoDepthRange = this.AutoDepthRange;
                            _depthExtractor.ManualDepthDistance = this.DepthRangeDistance;
                            _depthExtractor.UseGpuAcceleration = this.UseGpuAcceleration;
                            _depthExtractor.UseGeometryExtraction = this.UseGeometryExtraction;
                            _depthExtractor.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                        }
                        else // Normal
                        {
                            _depthExtractorFast ??= new DepthExtractorFast(app, DepthResolution, 2);
                            _depthExtractorFast.AutoDepthRange = this.AutoDepthRange;
                            _depthExtractorFast.ManualDepthDistance = this.DepthRangeDistance;
                            _depthExtractorFast.UseGpuAcceleration = this.UseGpuAcceleration;
                            _depthExtractorFast.ExtractDepthMap(view3D, OutputPath, timestamp, depthWidth, depthHeight, viewCorners);
                        }
                    }
                    catch (Exception ex)
                    {
                        UpdateStatusCallback?.Invoke($"Advertencia: Error en profundidad - {ex.Message}", Drawing.Color.Orange);
                        if (UseGpuAcceleration)
                        {
                            UpdateStatusCallback?.Invoke("Reintentando sin GPU...", Drawing.Color.Orange);
                            UseGpuAcceleration = false;
                            Execute(app); // Reintentar
                            return;
                        }
                    }
                }
                
                ExportMetadata(doc, view3D, OutputPath, timestamp);
                CreateNotificationFile(OutputPath, timestamp);
                
                string gpuInfo = UseGpuAcceleration ? " [GPU]" : "";
                UpdateStatusCallback?.Invoke($"Exportado: {timestamp}{gpuInfo}", Drawing.Color.Green);
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
                byte[] fileBytes = File.ReadAllBytes(generatedFile);
                using (var ms = new MemoryStream(fileBytes))
                using (var originalImage = Drawing.Image.FromStream(ms))
                {
                    int newHeight = originalImage.Height;

                    if (newHeight % 2 != 0)
                    {
                        newHeight--;
                    }

                    if (newHeight == originalImage.Height)
                    {
                        File.Copy(generatedFile, targetFile, true);
                    }
                    else
                    {
                        var cropRect = new Drawing.Rectangle(0, 0, originalImage.Width, newHeight);
                        using (var newBitmap = new Drawing.Bitmap(cropRect.Width, cropRect.Height))
                        {
                            newBitmap.SetResolution(originalImage.HorizontalResolution, originalImage.VerticalResolution);
                            using (var g = Drawing.Graphics.FromImage(newBitmap))
                            {
                                g.DrawImage(originalImage, new Drawing.Rectangle(0, 0, newBitmap.Width, newBitmap.Height), cropRect, Drawing.GraphicsUnit.Pixel);
                            }
                            newBitmap.Save(targetFile, Drawing.Imaging.ImageFormat.Png);
                        }
                    }
                }
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
                },
                gpu_acceleration = UseGpuAcceleration
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
        
        public void Dispose()
        {
            _depthExtractor?.Dispose();
            _depthExtractorFast?.Dispose();
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
            
            PushButton? button = panel.AddItem(buttonData) as PushButton;
            
            return Result.Succeeded;
        }
        
        public Result OnShutdown(UIControlledApplication application)
        {
            return Result.Succeeded;
        }
    }
    
    /// <summary>
    /// Ventana principal del plugin (v0.3.0 con GPU)
    /// </summary>
    public class WabiSabiBridgeWindow : WinForms.Form
    {
        private readonly UIApplication _uiApp;
        private readonly ExternalEvent _externalEvent;
        private readonly ExportEventHandler _eventHandler;
        
        // CORREGIDO: Se elimina 'readonly' y se inicializa con 'null!' para suprimir la advertencia CS8618.
        // El compilador no puede saber que InitializeComponent() los inicializa.
        private WinForms.Button _exportButton = null!;
        private WinForms.TextBox _outputPathTextBox = null!;
        private WinForms.Button _browseButton = null!;
        private WinForms.Label _statusLabel = null!;
        private WinForms.CheckBox _autoExportCheckBox = null!;
        private WinForms.CheckBox _exportDepthCheckBox = null!;
        private WinForms.ComboBox _depthResolutionCombo = null!;
        private WinForms.ComboBox _depthQualityCombo = null!;
        private WinForms.Timer _autoExportTimer = null!;
        private WinForms.TrackBar _depthRangeTrackBar = null!;
        private WinForms.Label _depthRangeLabel = null!;
        private WinForms.Label _depthRangeValueLabel = null!;
        private WinForms.CheckBox _autoDepthCheckBox = null!;
        private WinForms.CheckBox _gpuAccelerationCheckBox = null!;
        private WinForms.CheckBox _geometryExtractionCheckBox = null!;
        private WinForms.Label _gpuStatusLabel = null!;
        
        // Configuración
        private readonly WabiSabiConfig _config;
        
        public WabiSabiBridgeWindow(UIApplication uiApp, ExternalEvent externalEvent, ExportEventHandler eventHandler)
        {
            _uiApp = uiApp;
            _externalEvent = externalEvent;
            _eventHandler = eventHandler;
            _eventHandler.UpdateStatusCallback = UpdateStatus;
            _config = WabiSabiConfig.Load();
            InitializeComponent(); // Este método inicializa los campos de arriba
            CheckGpuStatus();
        }
        
        private void InitializeComponent()
        {
            // Configuración de la ventana
            Text = "WabiSabi Bridge - v0.3.0 (GPU Enhanced)";
            Size = new Drawing.Size(420, 550);
            StartPosition = WinForms.FormStartPosition.CenterScreen;
            FormBorderStyle = WinForms.FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            
            // Layout principal
            WinForms.TableLayoutPanel mainLayout = new WinForms.TableLayoutPanel
            {
                Dock = WinForms.DockStyle.Fill,
                ColumnCount = 1,
                RowCount = 10,
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
                Height = 125,
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
            
            // Checkbox para GPU
            _gpuAccelerationCheckBox = new WinForms.CheckBox
            {
                Text = "Aceleración GPU",
                Location = new Drawing.Point(30, 95),
                Width = 130,
                Checked = _config.UseGpuAcceleration,
                Enabled = _config.ExportDepth
            };
            _gpuAccelerationCheckBox.CheckedChanged += GpuAccelerationCheckBox_CheckedChanged;
            
            // Checkbox para extracción de geometría (experimental)
            _geometryExtractionCheckBox = new WinForms.CheckBox
            {
                Text = "Modo Experimental",
                Location = new Drawing.Point(170, 95),
                Width = 140,
                Checked = _config.UseGeometryExtraction,
                Enabled = _config.ExportDepth && _config.UseGpuAcceleration,
                ForeColor = Drawing.Color.DarkOrange
            };
            _geometryExtractionCheckBox.CheckedChanged += GeometryExtractionCheckBox_CheckedChanged;
            
            exportOptionsGroup.Controls.Add(_exportDepthCheckBox);
            exportOptionsGroup.Controls.Add(depthResLabel);
            exportOptionsGroup.Controls.Add(_depthResolutionCombo);
            exportOptionsGroup.Controls.Add(depthQualityLabel);
            exportOptionsGroup.Controls.Add(_depthQualityCombo);
            exportOptionsGroup.Controls.Add(timeEstimateLabel);
            exportOptionsGroup.Controls.Add(_gpuAccelerationCheckBox);
            exportOptionsGroup.Controls.Add(_geometryExtractionCheckBox);

            // Fila 4: Control de rango de profundidad
            WinForms.Panel depthRangePanel = new WinForms.Panel { Height = 60, Dock = WinForms.DockStyle.Fill };

            _depthRangeLabel = new WinForms.Label
            {
                Text = "Distancia focal:",
                Location = new Drawing.Point(10, 5),
                Width = 100
            };

            _autoDepthCheckBox = new WinForms.CheckBox
            {
                Text = "Auto",
                Location = new Drawing.Point(280, 5),
                Width = 50,
                Checked = _config.AutoDepthRange
            };
            _autoDepthCheckBox.CheckedChanged += AutoDepthCheckBox_CheckedChanged;

            _depthRangeTrackBar = new WinForms.TrackBar
            {
                Location = new Drawing.Point(10, 25),
                Width = 260,
                Height = 30,
                Minimum = 10,
                Maximum = 500,
                Value = (int)_config.DepthRangeDistance,
                TickFrequency = 50,
                Enabled = !_config.AutoDepthRange
            };
            _depthRangeTrackBar.ValueChanged += DepthRangeTrackBar_ValueChanged;

            _depthRangeValueLabel = new WinForms.Label
            {
                Text = _config.AutoDepthRange ? "Auto" : $"{_config.DepthRangeDistance} ft",
                Location = new Drawing.Point(280, 30),
                Width = 80,
                TextAlign = Drawing.ContentAlignment.MiddleLeft
            };

            depthRangePanel.Controls.Add(_depthRangeLabel);
            depthRangePanel.Controls.Add(_autoDepthCheckBox);
            depthRangePanel.Controls.Add(_depthRangeTrackBar);
            depthRangePanel.Controls.Add(_depthRangeValueLabel);

            // Fila 5: Auto-exportación
            _autoExportCheckBox = new WinForms.CheckBox
            {
                Text = "Exportación automática (experimental)",
                Dock = WinForms.DockStyle.Fill,
                Checked = _config.AutoExport
            };
            _autoExportCheckBox.CheckedChanged += AutoExportCheckBox_CheckedChanged;
            
            // Fila 6: Estado GPU
            _gpuStatusLabel = new WinForms.Label
            {
                Text = "Verificando GPU...",
                Dock = WinForms.DockStyle.Fill,
                TextAlign = Drawing.ContentAlignment.MiddleCenter,
                ForeColor = Drawing.Color.Gray,
                Font = new Drawing.Font(Font.FontFamily, 8, Drawing.FontStyle.Italic)
            };
            
            // Fila 7: Estado
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
            mainLayout.Controls.Add(depthRangePanel, 0, 3);
            mainLayout.Controls.Add(_autoExportCheckBox, 0, 4);
            mainLayout.Controls.Add(_gpuStatusLabel, 0, 5);
            mainLayout.Controls.Add(_statusLabel, 0, 6);

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
        
        private void CheckGpuStatus()
        {
            try
            {
                // Pasa el `this` (la ventana) al manager para que pueda mostrar un diálogo
                IGpuAccelerationManager? gpuManager = new GpuAccelerationManager(this);
                
                if (gpuManager.IsGpuAvailable)
                {
                    _gpuStatusLabel.Text = "GPU: Disponible ✓";
                    _gpuStatusLabel.ForeColor = System.Drawing.Color.Green;
                }
                else
                {
                    // El mensaje de por qué no está disponible ya se habría mostrado
                    _gpuStatusLabel.Text = "GPU: No disponible (usando CPU paralela)";
                    _gpuStatusLabel.ForeColor = System.Drawing.Color.Orange;
                }
                gpuManager.Dispose();
            }
            catch (Exception ex) // Captura cualquier otra excepción inesperada
            {
                _gpuStatusLabel.Text = "GPU: Error al verificar";
                _gpuStatusLabel.ForeColor = System.Drawing.Color.Red;
                // Muestra el error detallado
                Autodesk.Revit.UI.TaskDialog.Show("Error de GPU", $"Error detallado al verificar el estado de la GPU:\n\n{ex.ToString()}");
            }
        }
        
        private void BrowseButton_Click(object? sender, EventArgs e)
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
        
        private void ExportButton_Click(object? sender, EventArgs e)
        {
            ExportCurrentView();
        }
        
        private void AutoExportCheckBox_CheckedChanged(object? sender, EventArgs e)
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
        
        private void AutoExportTimer_Tick(object? sender, EventArgs e)
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
        
        private void ExportDepthCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _config.ExportDepth = _exportDepthCheckBox.Checked;
            _depthResolutionCombo.Enabled = _exportDepthCheckBox.Checked;
            _depthQualityCombo.Enabled = _exportDepthCheckBox.Checked;
            _gpuAccelerationCheckBox.Enabled = _exportDepthCheckBox.Checked;
            _geometryExtractionCheckBox.Enabled = _exportDepthCheckBox.Checked && _gpuAccelerationCheckBox.Checked;
            _config.Save();
            UpdateTimeEstimate();
        }
        
        private void DepthResolutionCombo_SelectedIndexChanged(object? sender, EventArgs e)
        {
            if (int.TryParse(_depthResolutionCombo.SelectedItem?.ToString(), out int resolution))
            {
                _config.DepthResolution = resolution;
                _config.Save();
                UpdateTimeEstimate();
            }
        }
        
        private void DepthQualityCombo_SelectedIndexChanged(object? sender, EventArgs e)
        {
            _config.DepthQuality = _depthQualityCombo.SelectedIndex;
            _config.Save();
            UpdateTimeEstimate();
        }

        private void AutoDepthCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _depthRangeTrackBar.Enabled = !_autoDepthCheckBox.Checked;
            _depthRangeValueLabel.Text = _autoDepthCheckBox.Checked ? "Auto" : $"{_depthRangeTrackBar.Value} ft";
            _config.AutoDepthRange = _autoDepthCheckBox.Checked;
            _config.Save();
        }

        private void DepthRangeTrackBar_ValueChanged(object? sender, EventArgs e)
        {
            _depthRangeValueLabel.Text = $"{_depthRangeTrackBar.Value} ft";
            _config.DepthRangeDistance = _depthRangeTrackBar.Value;
            _config.Save();
        }
        
        private void GpuAccelerationCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _config.UseGpuAcceleration = _gpuAccelerationCheckBox.Checked;
            _geometryExtractionCheckBox.Enabled = _exportDepthCheckBox.Checked && _gpuAccelerationCheckBox.Checked;
            if (!_gpuAccelerationCheckBox.Checked)
            {
                _geometryExtractionCheckBox.Checked = false;
            }
            _config.Save();
            UpdateTimeEstimate();
        }
        
        private void GeometryExtractionCheckBox_CheckedChanged(object? sender, EventArgs e)
        {
            _config.UseGeometryExtraction = _geometryExtractionCheckBox.Checked;
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
            bool useGpu = _config.UseGpuAcceleration;
            bool useGeometry = _config.UseGeometryExtraction;
            
            // Estimaciones basadas en pruebas
            int[,] timeMatrixCpu = new int[,] {
                // Rápida, Normal, Alta
                { 1, 3, 5 },      // 256
                { 3, 10, 20 },    // 512
                { 10, 40, 80 },   // 1024
                { 40, 160, 320 }  // 2048
            };
            
            int[,] timeMatrixGpu = new int[,] {
                // Rápida, Normal, Alta
                { 1, 1, 2 },      // 256
                { 1, 3, 5 },      // 512
                { 2, 8, 15 },     // 1024
                { 8, 30, 60 }     // 2048
            };
            
            int resIndex = resolution == 256 ? 0 : resolution == 512 ? 1 : resolution == 1024 ? 2 : 3;
            int seconds = useGpu ? timeMatrixGpu[resIndex, quality] : timeMatrixCpu[resIndex, quality];
            
            if (useGeometry)
            {
                seconds = (int)(seconds * 1.5); // Modo experimental es más lento
            }
            
            string gpuText = useGpu ? " (GPU)" : " (CPU)";
            
            if (seconds < 60)
                return $"Tiempo estimado: ~{seconds} segundos{gpuText}";
            else
                return $"Tiempo estimado: ~{seconds / 60} minutos{gpuText}";
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
                _eventHandler.AutoDepthRange = _autoDepthCheckBox.Checked;
                _eventHandler.DepthRangeDistance = _depthRangeTrackBar.Value;
                _eventHandler.UseGpuAcceleration = _gpuAccelerationCheckBox.Checked;
                _eventHandler.UseGeometryExtraction = _geometryExtractionCheckBox.Checked;

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
            _autoExportTimer.Stop();
            _autoExportTimer.Dispose();
            _eventHandler.Dispose();
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
        public bool AutoDepthRange { get; set; } = true;
        public double DepthRangeDistance { get; set; } = 50.0;
        public bool UseGpuAcceleration { get; set; } = true;
        public bool UseGeometryExtraction { get; set; } = false;

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
                    return JsonConvert.DeserializeObject<WabiSabiConfig>(json) ?? new WabiSabiConfig();
                }
            }
            catch
            {
                // Ignorar errores al cargar la configuración y devolver una nueva
            }
            
            return new WabiSabiConfig();
        }
        
        public void Save()
        {
            try
            {
                string? dir = Path.GetDirectoryName(ConfigPath);
                if (dir != null && !Directory.Exists(dir))
                {
                    Directory.CreateDirectory(dir);
                }
                
                string json = JsonConvert.SerializeObject(this, Formatting.Indented);
                File.WriteAllText(ConfigPath, json);
            }
            catch 
            {
                // Ignorar errores al guardar la configuración
            }
        }
    }
}

// <<-- INICIO DE LA CORRECCIÓN -->>
// Se eliminó el espacio de nombres 'WabiSabiBridge.Extractors' duplicado y sus clases.
// El código original contenía aquí definiciones "dummy" que causaban los errores de
// compilación CS0104 y CS0111. Al eliminarlas, el compilador ahora resuelve
// correctamente las clases a sus definiciones completas en los archivos
// DepthExtractor.cs, DepthExtractorFast.cs y GpuAcceleration*.cs.
// <<-- FIN DE LA CORRECCIÓN -->>