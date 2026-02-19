#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cuda_runtime.h>
#include <future>
#include <json/json.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <websocketpp/client.hpp>
#include <websocketpp/config/asio_no_tls_client.hpp>


#include "ImageMMFWriter.h" // <--- NUEVO INCLUDE

typedef websocketpp::client<websocketpp::config::asio_client> client;

class WabiSabiRenderer {
public:
  struct RenderConfig {
    int width = 1280;
    int height = 720;
    float minDepth = 0.1f;
    float maxDepth = 100.0f;
    float depthThreshold = 0.03f;
    float normalThreshold = 0.4f;
    float cameraFov = 45.0f;
    float gamma = 0.35f;
    std::string outputPath;
    std::string csvPath;
    bool enableSegmentation = true;
    bool enableDepth = true;
    bool enableNormals = true;
    bool enableLines = true;
    bool enableWebViewer = false;

    bool enableDiskIO =
        false; // <--- NUEVO: Desactivar guardado en disco por defecto
    bool enableMMF = true; // <--- NUEVO: Activar MMF por defecto

    int webSocketPort = 9001;
    int maxFPS = 60; // <--- AUMENTADO OJO
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
    float3 lower_left_corner;
    float3 horizontal_vec;
    float3 vertical_vec;
    uint64_t timestamp;
    int64_t sequenceNumber; // <--- NUEVO
  };

  struct Statistics {
    float fps;
    float avgFrameTime;
    uint64_t totalFrames;
  };

  WabiSabiRenderer(const RenderConfig &config);
  ~WabiSabiRenderer();

  void Start();
  void Stop();
  void UpdateCamera(const CameraData &camera);

  void UpdateConfig(const RenderConfig &newConfig);

  Statistics GetStatistics() const;

private:
  struct HostFrameData {
    std::vector<float> depth;
    std::vector<float3> normals;
    std::vector<float> lines;
    std::vector<float4> segmentation;
    int width, height;
    int64_t sequenceNumber; // <--- NUEVO
  };

  // <--- NUEVO OBJETO MMF WRITER --->
  std::unique_ptr<ImageMMFWriter> mmfWriter;

  HostFrameData CopyGpuToHost(const RenderConfig &localConfig, int64_t seqNum);
  void SaveToDiskInternal(const HostFrameData &data,
                          const RenderConfig &localConfig);
  void WriteToMMFInternal(const HostFrameData &data); // <--- NUEVO MÉTODO

  RenderConfig config;
  mutable std::shared_mutex configMutex;

  std::atomic<bool> isRunning;
  std::thread renderThread;
  client ws_client;
  std::thread ws_thread;
  websocketpp::connection_hdl ws_connection_hdl;
  std::mutex ws_mutex;

  std::thread journalWatcher;
  std::atomic<bool> wsConnected{false};

  CameraData currentCamera;
  std::mutex cameraMutex;
  std::condition_variable cameraCv;
  bool cameraInitialized = false;

  void *geometryMMF;

  float *d_vertices = nullptr;
  int *d_triangles = nullptr;
  float *d_normals = nullptr;
  int *d_elementIds = nullptr;
  int *d_categoryIds = nullptr;

  float *d_depthMap = nullptr;
  float *d_normalMap = nullptr;
  float *d_linesMap = nullptr;
  float *d_segmentationMap = nullptr;
  int *d_elementIdPixelMap = nullptr;
  unsigned long long *d_zBuffer = nullptr; // <--- NUEVO Z-BUFFER

  int vertexCount = 0;
  int triangleCount = 0;
  int categoryCount = 0;

  float3 *d_categoryColors = nullptr;
  std::vector<float3> h_categoryColors;

  std::atomic<uint64_t> totalFrames;
  std::atomic<float> currentFPS;
  std::atomic<float> avgFrameTime;

  std::string lastKnownTimestamp;
  void CheckForUpdates();
  void ReloadGeometry();

  void RenderLoop();
  void OpenGeometryMMF();

  void RenderAllMaps(const CameraData &camera,
                     const RenderConfig &currentConfig, cudaStream_t stream);

  std::string FindLatestWabiSabiMMF();
  void ReadGeometryHeader();
  void LoadColorMappingCSV();
  void WatchJournal();
  void StartWebSocketServer();
  void SendMapsToWebSocket(const RenderConfig &currentConfig);
  CameraData GetLatestCamera();

  std::unordered_map<int, std::string> categoryIndexToNameMap;
};
