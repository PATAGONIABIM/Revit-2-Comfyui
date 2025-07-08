// ===== Core/Interfaces/IConfigurationService.cs =====
using System.Threading.Tasks;

namespace WabiSabiRevitBridge.Core.Interfaces
{
    public interface IConfigurationService
    {
        WabiSabiConfiguration GetConfiguration();
        Task SaveConfigurationAsync(WabiSabiConfiguration config);
        Task<WabiSabiConfiguration> LoadConfigurationAsync();
        void ResetToDefaults();
        
        // Profile management
        string[] GetProfileNames();
        Task<WabiSabiConfiguration> LoadProfileAsync(string profileName);
        Task SaveProfileAsync(string profileName, WabiSabiConfiguration config);
    }
}
