
// ===== WabiSabiApplication.cs =====
using System;
using System.Reflection;
using System.Windows.Media.Imaging;
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using WabiSabiRevitBridge.Core.Interfaces;
using WabiSabiRevitBridge.Core.Services;
using WabiSabiRevitBridge.Communication;
using WabiSabiRevitBridge.UI.Views;

namespace WabiSabiRevitBridge
{
    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    public class WabiSabiApplication : IExternalApplication
    {
        private static WabiSabiApplication _instance;
        private MainWindow _mainWindow;
        private DependencyContainer _container;
        
        public static WabiSabiApplication Instance => _instance;
        public UIControlledApplication UIApp { get; private set; }
        
        public Result OnStartup(UIControlledApplication application)
        {
            _instance = this;
            UIApp = application;
            
            try
            {
                // Initialize dependency injection
                InitializeDependencyInjection();
                
                // Create ribbon panel and button
                CreateRibbonUI(application);
                
                // Register event handlers
                RegisterEventHandlers(application);
                
                // Load configuration
                var configService = _container.Resolve<IConfigurationService>();
                var config = configService.LoadConfigurationAsync().GetAwaiter().GetResult();
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                TaskDialog.Show("WabiSabi Bridge Error", 
                    $"Failed to initialize WabiSabi Bridge:\n{ex.Message}");
                return Result.Failed;
            }
        }

        public Result OnShutdown(UIControlledApplication application)
        {
            try
            {
                // Clean up resources
                _mainWindow?.Close();
                _mainWindow = null;
                
                // Dispose communication channels
                var commManager = _container.Resolve<CommunicationManager>();
                commManager?.Dispose();
                
                // Unregister event handlers
                UnregisterEventHandlers(application);
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                TaskDialog.Show("WabiSabi Bridge Error", 
                    $"Error during shutdown:\n{ex.Message}");
                return Result.Failed;
            }
        }

        private void InitializeDependencyInjection()
        {
            _container = DependencyContainer.Instance;
            
            // Register services
            _container.RegisterSingleton<IConfigurationService, ConfigurationService>();
            _container.RegisterSingleton<CommunicationManager>();
            
            // Register extractors (will be implemented in next phase)
            // _container.RegisterTransient<IDataExtractor, HiddenLineExtractor>();
            // _container.RegisterTransient<IDataExtractor, DepthExtractor>();
            // etc...
        }

        private void CreateRibbonUI(UIControlledApplication application)
        {
            // Create ribbon tab
            string tabName = "WabiSabi";
            try
            {
                application.CreateRibbonTab(tabName);
            }
            catch (Exception)
            {
                // Tab already exists
            }

            // Create ribbon panel
            RibbonPanel panel = application.CreateRibbonPanel(tabName, "Bridge Tools");
            
            // Create push button
            string assemblyPath = Assembly.GetExecutingAssembly().Location;
            PushButtonData buttonData = new PushButtonData(
                "WabiSabiBridge",
                "WabiSabi\nBridge",
                assemblyPath,
                "WabiSabiRevitBridge.Commands.ShowMainWindowCommand"
            );

            PushButton pushButton = panel.AddItem(buttonData) as PushButton;
            
            // Set button image
            pushButton.LargeImage = GetEmbeddedImage("WabiSabiRevitBridge.Resources.icon_large.png");
            pushButton.Image = GetEmbeddedImage("WabiSabiRevitBridge.Resources.icon_small.png");
            
            // Set tooltip
            pushButton.ToolTip = "Launch WabiSabi Bridge";
            pushButton.LongDescription = "Opens the WabiSabi Bridge panel for real-time data export to ComfyUI";
        }

        private BitmapImage GetEmbeddedImage(string resourceName)
        {
            try
            {
                var assembly = Assembly.GetExecutingAssembly();
                var stream = assembly.GetManifestResourceStream(resourceName);
                
                if (stream != null)
                {
                    BitmapImage image = new BitmapImage();
                    image.BeginInit();
                    image.StreamSource = stream;
                    image.EndInit();
                    return image;
                }
            }
            catch (Exception)
            {
                // Return default image or null
            }
            
            return null;
        }

        private void RegisterEventHandlers(UIControlledApplication application)
        {
            // Register for view activated event to detect view changes
            application.ViewActivated += OnViewActivated;
            
            // Register for document changed event
            application.ControlledApplication.DocumentChanged += OnDocumentChanged;
        }

        private void UnregisterEventHandlers(UIControlledApplication application)
        {
            application.ViewActivated -= OnViewActivated;
            application.ControlledApplication.DocumentChanged -= OnDocumentChanged;
        }

        private void OnViewActivated(object sender, ViewActivatedEventArgs e)
        {
            // Notify main window of view change if it's open
            _mainWindow?.OnViewChanged(e.CurrentActiveView);
        }

        private void OnDocumentChanged(object sender, DocumentChangedEventArgs e)
        {
            // Notify main window of document changes if it's open
            _mainWindow?.OnDocumentChanged(e);
        }

        public void ShowMainWindow(UIDocument uiDoc)
        {
            if (_mainWindow == null || !_mainWindow.IsLoaded)
            {
                _mainWindow = new MainWindow(uiDoc, _container);
                _mainWindow.Show();
            }
            else
            {
                _mainWindow.Activate();
            }
        }
    }
}

