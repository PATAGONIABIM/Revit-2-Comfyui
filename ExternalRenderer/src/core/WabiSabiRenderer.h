#pragma once
#include <string>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>      
#include <chrono>
#include <cuda_runtime.h>
#include <vector>
#include <unordered_map> // <-- ¡SOLUCIÓN! Añadir esta línea

class WabiSabiRenderer {
public:
        
    struct RenderConfig {
        int width = 1280;
        int height = 720;
        float minDepth = 0.1f;
        float maxDepth = 100.0f;
        float depthThreshold = 0.03f; // Debe llamarse así
        float normalThreshold = 0.4f; // Y este debe existir
        float cameraFov = 45.0f;
        std::string outputPath;
        std::string csvPath;
        bool enableSegmentation = true;
        bool enableDepth = true;
        bool enableNormals = true;
        bool enableLines = true;
        bool enableWebViewer = false;
        int webSocketPort = 9001;
        int maxFPS = 30;
        int cudaDevice = 0;
        int blockSizeX = 16;
        int blockSizeY = 16;
        std::string depthFilename = "current_depth.png";
        std::string normalFilename = "current_normal.png";
        std::string linesFilename = "current_lines.png";
        std::string segmentationFilename = "current_segmentation.png";
    };

    struct CameraData {
    float3 eyePosition;
    float3 lower_left_corner; // Esquina inferior izquierda del plano de visión
    float3 horizontal_vec;    // Vector horizontal que abarca el plano de visión
    float3 vertical_vec;      // Vector vertical que abarca el plano de visión
    uint64_t timestamp;       // Mantener para la lógica de actualización
    };

    struct Statistics {
        float fps;
        float avgFrameTime;
        uint64_t totalFrames;
    };

    WabiSabiRenderer(const RenderConfig& config);
    ~WabiSabiRenderer();

    void Start();
    void Stop();
    void UpdateCamera(const CameraData& camera);
    Statistics GetStatistics() const;

private:
    RenderConfig config;
    std::atomic<bool> isRunning;
    std::thread renderThread;
    
    std::thread journalWatcher;  
    std::atomic<bool> wsConnected{false};  
    
    // Camera state
    CameraData currentCamera;  
    std::mutex cameraMutex;
    std::condition_variable cameraCv;
    bool cameraInitialized = false;    
    
    // Memory mapped files
    void* geometryMMF;
    
    // GPU buffers
    float* d_vertices;
    int* d_triangles;
    float* d_normals;
    int* d_elementIds;
    int* d_categoryIds;
    
    // Output buffers
    float* d_depthMap;
    float* d_normalMap;
    float* d_linesMap;
    float* d_segmentationMap;
    
    // Información de geometría
    int vertexCount = 0;
    int triangleCount = 0; 
    int categoryCount = 0;
    
    // Colores de categorías
    float3* d_categoryColors = nullptr;
    std::vector<float3> h_categoryColors;

    // Statistics
    std::atomic<uint64_t> totalFrames;
    std::atomic<float> currentFPS;
    std::atomic<float> avgFrameTime;
    
    void RenderLoop();
    void OpenGeometryMMF();
    void RenderAllMaps(const CameraData& camera);
    void SaveMapsToFiles();
    std::string FindLatestWabiSabiMMF();
    void ReadGeometryHeader();
    void LoadColorMappingCSV();
    void WatchJournal();           
    void StartWebSocketServer();   
    void SendMapsToWebSocket();    
    CameraData GetLatestCamera();

    std::unordered_map<int, std::string> categoryIndexToNameMap;
};