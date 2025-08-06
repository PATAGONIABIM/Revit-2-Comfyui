// WabiSabiExternalRenderer.h
#pragma once
#include <cuda_runtime.h>
#include <memory>
#include <thread>
#include <atomic>

class WabiSabiRenderer {
public:
    struct RenderConfig {
        int width = 1280;
        int height = 720;
        float minDepth = 0.1f;
        float maxDepth = 100.0f;
        std::string outputPath;
        std::string csvPath;
        bool enableSegmentation = true;
        bool enableDepth = true;
        bool enableNormals = true;
        bool enableLines = true;
    };

    struct CameraData {
        float3 eyePosition;
        float3 viewDirection;
        float3 upDirection;
        float3 rightDirection;
        uint64_t timestamp;
    };

private:
    // Memory mapped files
    void* geometryMMF;
    void* journalMMF;
    
    // GPU buffers
    float* d_vertices;
    int* d_triangles;
    float* d_normals;
    int* d_elementIds;
    int* d_categoryIds;
    
    // Output buffers (4 maps)
    float* d_depthMap;
    float* d_normalMap;
    float* d_linesMap;
    float* d_segmentationMap;
    
    // CSV color mapping
    std::unordered_map<int, float3> categoryColors;
    std::filesystem::file_time_type lastCsvModified;
    
    // Rendering state
    std::atomic<bool> isRunning;
    std::thread renderThread;
    std::thread journalWatcher;
    
public:
    WabiSabiRenderer(const RenderConfig& config);
    ~WabiSabiRenderer();
    
    void Start();
    void Stop();
    
private:
    void RenderLoop();
    void UpdateCameraFromJournal();
    void LoadColorMappingCSV();
    void RenderMaps(const CameraData& camera);
    void SaveMapsToFiles();

public:
    struct Statistics {
        float fps;
        float avgFrameTime;
        uint64_t totalFrames;
    };
    
    void UpdateCamera(const CameraData& camera);
    Statistics GetStatistics() const;
};