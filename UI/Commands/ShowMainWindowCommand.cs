// ===== Commands/ShowMainWindowCommand.cs =====
using Autodesk.Revit.Attributes;
using Autodesk.Revit.DB;
using Autodesk.Revit.UI;

namespace WabiSabiRevitBridge.Commands
{
    [Transaction(TransactionMode.Manual)]
    [Regeneration(RegenerationOption.Manual)]
    public class ShowMainWindowCommand : IExternalCommand
    {
        public Result Execute(ExternalCommandData commandData, ref string message, ElementSet elements)
        {
            try
            {
                var uiDoc = commandData.Application.ActiveUIDocument;
                
                if (uiDoc == null)
                {
                    TaskDialog.Show("WabiSabi Bridge", "Please open a Revit document first.");
                    return Result.Cancelled;
                }

                // Check if we have a 3D view
                var view3D = uiDoc.ActiveView as View3D;
                if (view3D == null)
                {
                    TaskDialog.Show("WabiSabi Bridge", 
                        "Please activate a 3D view to use WabiSabi Bridge.");
                    return Result.Cancelled;
                }

                // Show main window
                WabiSabiApplication.Instance.ShowMainWindow(uiDoc);
                
                return Result.Succeeded;
            }
            catch (Exception ex)
            {
                message = ex.Message;
                return Result.Failed;
            }
        }
    }
}