// ===== UI/Views/MainWindow.xaml.cs =====
using System;
using System.Windows;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;
using WabiSabiRevitBridge.Core.Services;
using WabiSabiRevitBridge.UI.ViewModels;

namespace WabiSabiRevitBridge.UI.Views
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private readonly MainViewModel _viewModel;
        private readonly UIDocument _uiDocument;

        public MainWindow(UIDocument uiDocument, DependencyContainer container)
        {
            InitializeComponent();
            
            _uiDocument = uiDocument ?? throw new ArgumentNullException(nameof(uiDocument));
            
            // Create and set view model
            _viewModel = new MainViewModel(uiDocument, container);
            DataContext = _viewModel;
            
            // Handle window events
            Loaded += OnLoaded;
            Closing += OnClosing;
        }

        private async void OnLoaded(object sender, RoutedEventArgs e)
        {
            // Initialize view model
            await _viewModel.InitializeAsync();
        }

        private void OnClosing(object sender, System.ComponentModel.CancelEventArgs e)
        {
            // Clean up
            _viewModel.Cleanup();
        }

        public void OnViewChanged(View newView)
        {
            if (newView is View3D view3D)
            {
                _viewModel.UpdateView(view3D);
            }
        }

        public void OnDocumentChanged(DocumentChangedEventArgs args)
        {
            _viewModel.OnDocumentChanged(args);
        }
    }
}
