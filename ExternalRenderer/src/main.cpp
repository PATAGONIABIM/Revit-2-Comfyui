// ExternalRenderer/src/main.cpp
#include <iostream>
#include <csignal>
#include <thread>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <json/json.h> // Necesitarás instalar jsoncpp

#include "core/WabiSabiRenderer.h"
#include "utils/JournalParser.h"
#include "cuda/CudaHelpers.h"
#include "cuda/CudaMath.h"

// Variables globales para manejo de señales
std::atomic<bool> g_shouldExit(false);
std::unique_ptr<WabiSabiRenderer> g_renderer;
std::unique_ptr<JournalParser> g_journalParser;

// Manejador de señales para cierre limpio
void signalHandler(int signal) {
    std::cout << "\n[INFO] Señal recibida (" << signal << "), cerrando aplicación..." << std::endl;
    g_shouldExit = true;
}

// Cargar configuración desde JSON
WabiSabiRenderer::RenderConfig loadConfig(const std::string& configPath) {
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
        const Json::Value& render = root["render"];
        config.width = render.get("width", 1280).asInt();
        config.height = render.get("height", 720).asInt();
        config.outputPath = render.get("outputPath", ".").asString();
        config.cameraFov = render.get("cameraFov", 75.0f).asFloat(); 
        
        // Configuración de mapas
        const Json::Value& maps = render["maps"];
        
        const Json::Value& depth = maps["depth"];
        config.enableDepth = depth.get("enabled", true).asBool();
        config.depthFilename = depth.get("filename", "current_depth.png").asString();
        config.minDepth = depth.get("minDepth", 0.1f).asFloat();
        config.maxDepth = depth.get("maxDepth", 100.0f).asFloat();
        
        const Json::Value& normal = maps["normal"];
        config.enableNormals = normal.get("enabled", true).asBool();
        config.normalFilename = normal.get("filename", "current_normal.png").asString();
        
        const Json::Value& lines = maps["lines"];
        config.enableLines = lines.get("enabled", true).asBool();
        config.linesFilename = lines.get("filename", "current_lines.png").asString();
        config.depthThreshold = lines.get("depthThreshold", 0.03f).asFloat();
        config.normalThreshold = lines.get("normalThreshold", 0.4f).asFloat();
        //config.lineThreshold = lines.get("threshold", 0.1f).asFloat();
        
        const Json::Value& segmentation = maps["segmentation"];
        config.enableSegmentation = segmentation.get("enabled", true).asBool();
        config.segmentationFilename = segmentation.get("filename", "current_segmentation.png").asString();
        config.csvPath = segmentation.get("csvPath", "category_colors.csv").asString();
        
        // Configuración de rendimiento
        const Json::Value& performance = root["performance"];
        config.maxFPS = performance.get("maxFPS", 30).asInt();
        config.cudaDevice = performance.get("cudaDevice", 0).asInt();
        config.blockSizeX = performance.get("blockSizeX", 16).asInt();
        config.blockSizeY = performance.get("blockSizeY", 16).asInt();
        
        // Configuración del visualizador web
        const Json::Value& webViewer = root["webViewer"];
        config.enableWebViewer = webViewer.get("enabled", true).asBool();
        config.webSocketPort = webViewer.get("port", 9001).asInt();
        
        std::cout << "[CONFIG] Configuración cargada:" << std::endl;
        std::cout << "  - Resolución: " << config.width << "x" << config.height << std::endl;
        std::cout << "  - Output: " << config.outputPath << std::endl;
        std::cout << "  - FOV de Cámara: " << config.cameraFov << std::endl;
        std::cout << "  - Mapas habilitados: ";
        if (config.enableDepth) std::cout << "Depth ";
        if (config.enableNormals) std::cout << "Normal ";
        if (config.enableLines) std::cout << "Lines ";
        if (config.enableSegmentation) std::cout << "Segmentation";
        std::cout << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] Error cargando configuración: " << e.what() << std::endl;
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
        std::cout << "[INFO] Creando directorio de salida: " << config.outputPath << std::endl;
        std::filesystem::create_directories(config.outputPath);
    }
    
    return true;
}

// Callback para actualizaciones de cámara desde el journal
void onCameraUpdate(const JournalCameraData& cameraData, const WabiSabiRenderer::RenderConfig& config) {
    static int updateCount = 0;
    updateCount++;
    
    // El target ahora es un dato primario, no calculado. Lo imprimimos directamente.
    std::cout << "[CAMERA UPDATE " << updateCount << "] " 
              << "Eye: (" << cameraData.eyePosition.x << ", " 
              << cameraData.eyePosition.y << ", " 
              << cameraData.eyePosition.z << ") "
              << "Target: (" << cameraData.targetPosition.x << ", " // <-- Imprimimos el target real
              << cameraData.targetPosition.y << ", "
              << cameraData.targetPosition.z << ")" << std::endl;
    
    if (g_renderer) {
        // --- NUEVA LÓGICA DE CÁMARA (CORREGIDA) ---
        WabiSabiRenderer::CameraData camera;

        float fov_vertical_grados = config.cameraFov;

        // 2. Definir vectores de la base de la cámara
        camera.eyePosition = cameraData.eyePosition;

        // ¡ESTA ES LA CORRECCIÓN CLAVE!
        // Calcular el vector de dirección real y normalizarlo.
        float3 viewDir = CudaMath::normalize(cameraData.targetPosition - cameraData.eyePosition); 
        
        float3 worldUp = cameraData.upDirection;
        float3 rightDir = CudaMath::normalize(CudaMath::cross(viewDir, worldUp));
        float3 upDir = CudaMath::normalize(CudaMath::cross(rightDir, viewDir));

        // 3. Calcular dimensiones del plano de visión (sin cambios)
        float aspectRatio = static_cast<float>(config.width) / static_cast<float>(config.height);
        float fov_vertical_rad = fov_vertical_grados * (3.1415926535f / 180.0f);
        float viewPlaneHeight = 2.0f * tan(fov_vertical_rad / 2.0f);
        float viewPlaneWidth = aspectRatio * viewPlaneHeight;

        camera.horizontal_vec = viewPlaneWidth * rightDir;
        camera.vertical_vec = viewPlaneHeight * upDir;
        
        // El cálculo de la esquina inferior izquierda sigue siendo válido con el viewDir correcto.
        camera.lower_left_corner = camera.eyePosition
                                 + viewDir 
                                 - (0.5f * camera.horizontal_vec)
                                 - (0.5f * camera.vertical_vec);

        camera.timestamp = cameraData.timestamp;
        
        g_renderer->UpdateCamera(camera);
    }
}

int main(int argc, char* argv[]) {
    // Configurar manejadores de señales
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Imprimir banner
    printBanner();
    
    // Verificar requisitos del sistema
    if (!checkSystemRequirements()) {
        std::cerr << "[ERROR] El sistema no cumple los requisitos mínimos." << std::endl;
        return 1;
    }
    
    try {
        // Buscar archivo de configuración
        std::string configPath = "renderer_config.json";
        if (argc > 1) {
            configPath = argv[1];
        }
        
        // Si no existe en el directorio actual, buscar en el directorio del ejecutable
        if (!std::filesystem::exists(configPath)) {
            auto exePath = std::filesystem::path(argv[0]).parent_path();
            auto altConfigPath = exePath / "renderer_config.json";
            if (std::filesystem::exists(altConfigPath)) {
                configPath = altConfigPath.string();
            }
        }
        
        std::cout << "[CONFIG] Usando archivo de configuración: " << configPath << std::endl;
        
        // Cargar configuración
        auto config = loadConfig(configPath);
        
        // Crear renderer
        std::cout << "[RENDERER] Inicializando renderer..." << std::endl;
        g_renderer = std::make_unique<WabiSabiRenderer>(config);
        
        // Crear parser del journal
        std::cout << "[JOURNAL] Inicializando monitor de journal..." << std::endl;
        g_journalParser = std::make_unique<JournalParser>();
        
        // Detectar versión de Revit (puedes hacer esto más sofisticado)
        std::string revitVersion = "2026"; // Por defecto
        if (argc > 2) {
            revitVersion = argv[2];
        }
        
        if (!g_journalParser->Initialize(revitVersion)) {
            std::cerr << "[ERROR] No se pudo inicializar el monitor de journal." << std::endl;
            std::cerr << "[INFO] Asegúrate de que Revit esté ejecutándose." << std::endl;
            return 1;
        }

        // --- INICIO DE LA CORRECCIÓN ---
        // Crear un lambda que "capture" la variable 'config'.
        // Este lambda SÍ tiene la firma que el JournalParser espera: void(const JournalCameraData&).
        auto cameraCallback = [&](const JournalCameraData& cameraData) {
            // Dentro del lambda, llamamos a nuestra función original con el 'config' capturado.
            onCameraUpdate(cameraData, config);
        };

        // Procesa la última cámara del journal al iniciar para tener un estado inicial.
        // Ahora pasamos el lambda en lugar de la función directamente.
        g_journalParser->ProcessInitialCamera(cameraCallback);
        
        // Iniciar monitoreo del journal
        g_journalParser->StartWatching(cameraCallback);
        // --- FIN DE LA CORRECCIÓN ---

        // Iniciar renderer
        std::cout << "[RENDERER] Iniciando renderizado..." << std::endl;
        g_renderer->Start();
        
        // Mensaje de estado
        std::cout << "\n═════════════════════════════════════════════════════" << std::endl;
        std::cout << "[OK] Sistema iniciado correctamente" << std::endl;
        std::cout << "[INFO] Renderizando a: " << config.outputPath << std::endl;
        if (config.enableWebViewer) {
            std::cout << "[INFO] Visualizador web en: http://localhost:" << config.webSocketPort << std::endl;
        }
        std::cout << "[INFO] Presiona Ctrl+C para salir" << std::endl;
        std::cout << "═════════════════════════════════════════════════════\n" << std::endl;
        
        // Loop principal
        auto lastStatusTime = std::chrono::steady_clock::now();
        while (!g_shouldExit) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Imprimir estadísticas cada 5 segundos
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - lastStatusTime).count() >= 5) {
                auto stats = g_renderer->GetStatistics();
                std::cout << "[STATS] FPS: " << stats.fps 
                         << " | Frame time: " << stats.avgFrameTime << "ms"
                         << " | Frames: " << stats.totalFrames << std::endl;
                lastStatusTime = now;
            }
        }
        
        // Limpieza
        std::cout << "\n[INFO] Deteniendo servicios..." << std::endl;
        
        g_journalParser->StopWatching();
        g_renderer->Stop();
        
        std::cout << "[OK] Aplicación cerrada correctamente." << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "[ERROR FATAL] " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}