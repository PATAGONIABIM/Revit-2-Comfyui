// ExternalRenderer/src/main.cpp
#define NOMINMAX
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <json/json.h> // Necesitarás instalar jsoncpp
#include <thread>


#include "core/WabiSabiRenderer.h"
#include "cuda/CudaHelpers.h"
#include "cuda/CudaMath.h"
#include "utils/JournalParser.h"


// Variables globales para manejo de señales
std::atomic<bool> g_shouldExit(false);
std::unique_ptr<WabiSabiRenderer> g_renderer;
std::unique_ptr<JournalParser> g_journalParser;

// Función auxiliar para obtener el tiempo de modificación del archivo
// Se usa std::filesystem::file_time_type para compatibilidad
std::filesystem::file_time_type getLastWriteTime(const std::string &path) {
  try {
    return std::filesystem::last_write_time(path);
  } catch (...) {
    // En caso de error (ej. archivo bloqueado momentáneamente), devolvemos el
    // tiempo mínimo
    return std::filesystem::file_time_type::min();
  }
}

// Manejador de señales para cierre limpio
void signalHandler(int signal) {
  std::cout << "\n[INFO] Señal recibida (" << signal
            << "), cerrando aplicación..." << std::endl;
  g_shouldExit = true;
}

// Cargar configuración desde JSON
WabiSabiRenderer::RenderConfig loadConfig(const std::string &configPath) {
  WabiSabiRenderer::RenderConfig config;

  try {
    std::ifstream configFile(configPath);
    if (!configFile.is_open()) {
      throw std::runtime_error("No se pudo abrir archivo de configuración");
    }

    Json::Value root;
    Json::CharReaderBuilder builder;
    std::string errors;

    if (!Json::parseFromStream(builder, configFile, &root, &errors)) {
      throw std::runtime_error("Error parseando JSON: " + errors);
    }

    // Leer configuración de render
    const Json::Value &render = root["render"];
    config.width = render.get("width", 1280).asInt();
    config.height = render.get("height", 720).asInt();
    config.outputPath = render.get("outputPath", ".").asString();
    config.enableDiskIO = render.get("enableDiskIO", false)
                              .asBool(); // <--- Leer opción de configuración

    // CORRECCIÓN 1: Ajustar el FOV por defecto.
    config.cameraFov = render.get("cameraFov", 50.0f).asFloat();

    // Configuración de mapas
    const Json::Value &maps = render["maps"];

    const Json::Value &depth = maps["depth"];
    config.gamma = depth.get("gamma", 0.35f).asFloat(); // Leer gamma
    config.enableDepth = depth.get("enabled", true).asBool();
    config.depthFilename =
        depth.get("filename", "current_depth.png").asString();
    config.minDepth = depth.get("minDepth", 0.1f).asFloat();
    config.maxDepth = depth.get("maxDepth", 100.0f).asFloat();

    const Json::Value &normal = maps["normal"];
    config.enableNormals = normal.get("enabled", true).asBool();
    config.normalFilename =
        normal.get("filename", "current_normal.png").asString();

    const Json::Value &lines = maps["lines"];
    config.enableLines = lines.get("enabled", true).asBool();
    config.linesFilename =
        lines.get("filename", "current_lines.png").asString();
    config.depthThreshold = lines.get("depthThreshold", 0.03f).asFloat();
    config.normalThreshold = lines.get("normalThreshold", 0.4f).asFloat();

    const Json::Value &segmentation = maps["segmentation"];
    config.enableSegmentation = segmentation.get("enabled", true).asBool();
    config.segmentationFilename =
        segmentation.get("filename", "current_segmentation.png").asString();
    config.csvPath =
        segmentation.get("csvPath", "category_colors.csv").asString();

    // Configuración de rendimiento
    const Json::Value &performance = root["performance"];
    config.maxFPS = performance.get("maxFPS", 30).asInt();
    config.cudaDevice = performance.get("cudaDevice", 0).asInt();
    config.blockSizeX = performance.get("blockSizeX", 16).asInt();
    config.blockSizeY = performance.get("blockSizeY", 16).asInt();

    // Configuración del visualizador web
    const Json::Value &webViewer = root["webViewer"];
    config.enableWebViewer = webViewer.get("enabled", true).asBool();
    config.webSocketPort = webViewer.get("port", 9001).asInt();

    std::cout << "[CONFIG] Configuración cargada:" << std::endl;
    std::cout << "  - Resolución: " << config.width << "x" << config.height
              << std::endl;
    std::cout << "  - Output: " << config.outputPath << std::endl;
    std::cout << "  - FOV de Cámara: " << config.cameraFov << std::endl;
    std::cout << "  - Mapas habilitados: ";
    if (config.enableDepth)
      std::cout << "Depth ";
    if (config.enableNormals)
      std::cout << "Normal ";
    if (config.enableLines)
      std::cout << "Lines ";
    if (config.enableSegmentation)
      std::cout << "Segmentation";
    std::cout << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Error cargando configuración: " << e.what()
              << std::endl;
    std::cerr << "[INFO] Usando configuración por defecto." << std::endl;
  }

  return config;
}

// Función para imprimir banner
void printBanner() {
  std::cout << R"(
╔══════════════════════════════════════════════════════════════╗
║           WabiSabi Bridge External Renderer v1.0             ║
║                                                              ║
║  Renderizador GPU en tiempo real para integración           ║
║  Revit -> ComfyUI con soporte para múltiples mapas         ║
╚══════════════════════════════════════════════════════════════╝
)" << std::endl;
}

// Función para verificar requisitos del sistema
bool checkSystemRequirements() {
  std::cout << "[SISTEMA] Verificando requisitos..." << std::endl;

  // Verificar CUDA
  int deviceCount = 0;
  cudaError_t error = cudaGetDeviceCount(&deviceCount);

  if (error != cudaSuccess || deviceCount == 0) {
    std::cerr << "[ERROR] No se detectó GPU CUDA compatible." << std::endl;
    return false;
  }

  std::cout << "[CUDA] Dispositivos detectados: " << deviceCount << std::endl;

  // Imprimir información del dispositivo
  printCudaDeviceInfo(0);

  // Verificar que existe el directorio de salida
  auto config = loadConfig("renderer_config.json");
  if (!std::filesystem::exists(config.outputPath)) {
    std::cout << "[INFO] Creando directorio de salida: " << config.outputPath
              << std::endl;
    std::filesystem::create_directories(config.outputPath);
  }

  return true;
}

// Callback para actualizaciones de cámara desde el journal
void onCameraUpdate(const JournalCameraData &cameraData,
                    const WabiSabiRenderer::RenderConfig &config) {
  static int updateCount = 0;
  updateCount++;

  std::cout << "[CAMERA UPDATE " << updateCount << "] "
            << "Eye: (" << cameraData.eyePosition.x << ", "
            << cameraData.eyePosition.y << ", " << cameraData.eyePosition.z
            << ") "
            << "Target: (" << cameraData.targetPosition.x << ", "
            << cameraData.targetPosition.y << ", "
            << cameraData.targetPosition.z << ")" << std::endl;

  if (g_renderer) {
    // --- LÓGICA DE CÁMARA (ENCUADRE) ---
    WabiSabiRenderer::CameraData camera;

    float fov_vertical_grados = config.cameraFov;

    // 1. Definir posición del ojo
    camera.eyePosition = cameraData.eyePosition;

    // 2. Calcular vectores de base (Normalizados)
    float3 viewDir =
        CudaMath::normalize(cameraData.targetPosition - cameraData.eyePosition);
    float3 worldUp = cameraData.upDirection;
    float3 rightDir = CudaMath::normalize(CudaMath::cross(viewDir, worldUp));
    float3 upDir = CudaMath::normalize(CudaMath::cross(rightDir, viewDir));

    // 3. Calcular dimensiones del plano de visión (View Plane)
    float aspectRatio =
        static_cast<float>(config.width) / static_cast<float>(config.height);
    float fov_vertical_rad = fov_vertical_grados * (3.1415926535f / 180.0f);

    float viewPlaneHeight = 2.0f * tan(fov_vertical_rad / 2.0f);
    float viewPlaneWidth = aspectRatio * viewPlaneHeight;

    // 4. Construir los vectores que definen el "frustum" para el Ray Tracing
    camera.horizontal_vec = viewPlaneWidth * rightDir;
    camera.vertical_vec = viewPlaneHeight * upDir;

    // 5. Calcular la esquina inferior izquierda del plano de visión
    camera.lower_left_corner = camera.eyePosition + viewDir -
                               (0.5f * camera.horizontal_vec) -
                               (0.5f * camera.vertical_vec);

    camera.timestamp = cameraData.timestamp;
    camera.sequenceNumber = updateCount; // <--- CRITICAL FIX: Assign sequence
                                         // number so ComfyUI detects change

    g_renderer->UpdateCamera(camera);
  }
}

int main(int argc, char *argv[]) {
  // Configurar manejadores de señales
  signal(SIGINT, signalHandler);
  signal(SIGTERM, signalHandler);

  // Imprimir banner
  printBanner();

  // Verificar requisitos del sistema
  if (!checkSystemRequirements()) {
    std::cerr << "[ERROR] El sistema no cumple los requisitos mínimos."
              << std::endl;
    return 1;
  }

  try {
    // Buscar archivo de configuración
    std::string configPath = "renderer_config.json";
    if (argc > 1) {
      configPath = argv[1];
    }

    // Si no existe en el directorio actual, buscar en el directorio del
    // ejecutable
    if (!std::filesystem::exists(configPath)) {
      auto exePath = std::filesystem::path(argv[0]).parent_path();
      auto altConfigPath = exePath / "renderer_config.json";
      if (std::filesystem::exists(altConfigPath)) {
        configPath = altConfigPath.string();
      }
    }

    std::cout << "[CONFIG] Usando archivo de configuración: " << configPath
              << std::endl;

    // Cargar configuración inicial
    auto config = loadConfig(configPath);

    // Crear renderer
    std::cout << "[RENDERER] Inicializando renderer..." << std::endl;
    g_renderer = std::make_unique<WabiSabiRenderer>(config);

    // Crear parser del journal
    std::cout << "[JOURNAL] Inicializando monitor de journal..." << std::endl;
    g_journalParser = std::make_unique<JournalParser>();

    // Detectar versión de Revit
    std::string revitVersion = "2026";
    if (argc > 2) {
      revitVersion = argv[2];
    }

    if (!g_journalParser->Initialize(revitVersion)) {
      std::cerr << "[ERROR] No se pudo inicializar el monitor de journal."
                << std::endl;
      std::cerr << "[INFO] Asegúrate de que Revit esté ejecutándose."
                << std::endl;
      return 1;
    }

    // Crear un lambda que "capture" la variable 'config'.
    // NOTA: Como 'config' puede cambiar por el Hot Swap, capturamos una
    // referencia pero en este contexto, 'config' es una variable local de main.
    // Para soportar que el callback use la configuración actualizada,
    // deberíamos consultar la config actual del renderer o pasarla de forma
    // segura. Por simplicidad en este ejemplo, seguiremos usando la config
    // capturada localmente, pero ten en cuenta que el cambio de resolución "en
    // caliente" en el callback requeriría lógica adicional si la lambda captura
    // por valor. Capturamos 'config' por referencia para que si la actualizamos
    // abajo, el callback lo vea.
    auto cameraCallback = [&](const JournalCameraData &cameraData) {
      onCameraUpdate(cameraData, config);
    };

    // Procesa la última cámara del journal al iniciar
    g_journalParser->ProcessInitialCamera(cameraCallback);

    // Iniciar monitoreo del journal
    g_journalParser->StartWatching(cameraCallback);

    // Iniciar renderer
    std::cout << "[RENDERER] Iniciando renderizado..." << std::endl;
    g_renderer->Start();

    // Mensaje de estado
    std::cout << "\n═════════════════════════════════════════════════════"
              << std::endl;
    std::cout << "[OK] Sistema iniciado correctamente" << std::endl;
    std::cout << "[INFO] Renderizando a: " << config.outputPath << std::endl;
    if (config.enableWebViewer) {
      std::cout << "[INFO] Visualizador web en: http://localhost:"
                << config.webSocketPort << std::endl;
    }
    std::cout << "[INFO] Monitoreando cambios en: " << configPath << std::endl;
    std::cout << "[INFO] Presiona Ctrl+C para salir" << std::endl;
    std::cout << "═════════════════════════════════════════════════════\n"
              << std::endl;

    // --- LOOP PRINCIPAL CON HOT SWAP ---

    // Inicializar tiempo de última escritura del config
    auto lastConfigTime = getLastWriteTime(configPath);
    auto lastStatusTime = std::chrono::steady_clock::now();

    while (!g_shouldExit) {
      // Dormir un poco para no saturar CPU (200ms para dar tiempo al IO)
      std::this_thread::sleep_for(std::chrono::milliseconds(200));

      // 1. CHEQUEO DE HOT SWAP DEL CONFIG
      try {
        auto currentConfigTime = getLastWriteTime(configPath);

        // Si el archivo es más nuevo que la última vez que lo leímos
        if (currentConfigTime > lastConfigTime) {
          std::cout << "\n[HOT SWAP] Cambio detectado en " << configPath
                    << ". Recargando..." << std::endl;

          // Esperar un momento breve para asegurar que la escritura del archivo
          // terminó
          std::this_thread::sleep_for(std::chrono::milliseconds(100));

          // Recargar el JSON
          auto newConfig = loadConfig(configPath);

          // Actualizar variable local 'config' (afecta al lambda cameraCallback
          // si fue capturada por referencia)
          config = newConfig;

          // Aplicar al renderer
          if (g_renderer) {
            g_renderer->UpdateConfig(newConfig);
          }

          lastConfigTime = currentConfigTime;
          std::cout << "[HOT SWAP] Configuración actualizada correctamente."
                    << std::endl;
        }
      } catch (const std::exception &e) {
        std::cerr << "[HOT SWAP ERROR] " << e.what() << std::endl;
        // Actualizamos el tiempo para no spammear el error si el archivo está
        // corrupto temporalmente
        lastConfigTime = getLastWriteTime(configPath);
      }

      // 2. Imprimir estadísticas cada 5 segundos
      auto now = std::chrono::steady_clock::now();
      if (std::chrono::duration_cast<std::chrono::seconds>(now - lastStatusTime)
              .count() >= 5) {
        if (g_renderer) {
          auto stats = g_renderer->GetStatistics();
          std::cout << "[STATS] FPS: " << stats.fps
                    << " | Frame time: " << stats.avgFrameTime << "ms"
                    << " | Frames: " << stats.totalFrames << std::endl;
        }
        lastStatusTime = now;
      }
    }

    // Limpieza
    std::cout << "\n[INFO] Deteniendo servicios..." << std::endl;

    g_journalParser->StopWatching();
    g_renderer->Stop();

    std::cout << "[OK] Aplicación cerrada correctamente." << std::endl;

  } catch (const std::exception &e) {
    std::cerr << "[ERROR FATAL] " << e.what() << std::endl;
    return 1;
  }

  return 0;
}