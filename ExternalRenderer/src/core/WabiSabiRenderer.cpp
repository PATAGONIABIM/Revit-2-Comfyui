// WabiSabiRenderer.cpp - Implementación completa con renderizado real
#include "core/WabiSabiRenderer.h"
#include <iostream>
#include <json/json.h>
#include <fstream>
#include <Windows.h>
#include <cuda_runtime.h>
#include <chrono>
#include <filesystem>
#include <cstdint>
#include "../cuda/WabiSabiKernels.cuh"
#include "../cuda/CudaHelpers.h"
#include "utils/CSVReader.h"

// Para guardar PNG
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

// Estructura del header en el MMF
#pragma pack(push, 1)
struct GeometryHeader {
    int32_t vertexCount;
    int32_t triangleCount;
    int32_t categoryCount;
    int64_t verticesOffset;
    int64_t indicesOffset;
    int64_t normalsOffset;
    int64_t elementIdsOffset;
    int64_t categoryIdsOffset;
    int64_t categoryMappingOffset;
};
#pragma pack(pop)

WabiSabiRenderer::WabiSabiRenderer(const RenderConfig& config) 
    : config(config), isRunning(false), totalFrames(0), currentFPS(0), avgFrameTime(0) {
    
    std::cout << "[RENDERER] Inicializando WabiSabiRenderer..." << std::endl;
    
    // Inicializar CUDA
    CUDA_CHECK(cudaSetDevice(config.cudaDevice));
    printCudaDeviceInfo(config.cudaDevice);
    
    // Abrir Memory Mapped Files
    OpenGeometryMMF();
    
    // Alocar buffers de salida en GPU
    size_t pixelCount = config.width * config.height;
    CUDA_CHECK(cudaMalloc(&d_depthMap, pixelCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normalMap, pixelCount * 3 * sizeof(float))); // Es float3, así que son 3 floats
    CUDA_CHECK(cudaMalloc(&d_linesMap, pixelCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_segmentationMap, pixelCount * 4 * sizeof(float))); // Es float4, así que son 4 floats

    // --- ELIMINA LAS SIGUIENTES LÍNEAS ---
    // currentCamera.eyePosition = make_float3(10.0f, 10.0f, 10.0f);
    // currentCamera.viewDirection = make_float3(-1.0f, -1.0f, -1.0f);
    // currentCamera.upDirection = make_float3(0.0f, 0.0f, 1.0f);
    // currentCamera.rightDirection = make_float3(1.0f, 0.0f, 0.0f);
    // --- FIN DE LA ELIMINACIÓN ---
    
    // Cargar mapeo de colores
    LoadColorMappingCSV();
    
    std::cout << "[RENDERER] Inicialización completa" << std::endl;
}

void WabiSabiRenderer::OpenGeometryMMF() {
    // Buscar el MMF más reciente de WabiSabi
    std::string mmfName = FindLatestWabiSabiMMF();
    
    HANDLE hMapFile = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        mmfName.c_str()
    );
    
    if (hMapFile == NULL) {
        throw std::runtime_error("No se pudo abrir el MMF de geometría: " + mmfName);
    }
    
    geometryMMF = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0, 0, 0
    );
    
    if (!geometryMMF) {
        CloseHandle(hMapFile);
        throw std::runtime_error("No se pudo mapear el MMF");
    }
    
    // Leer header para obtener offsets y configurar punteros
    ReadGeometryHeader();
    
    CloseHandle(hMapFile); // Ya no necesitamos el handle
}

void WabiSabiRenderer::ReadGeometryHeader() {
    if (!geometryMMF) return;
    
    // Leer header desde el inicio del MMF
    GeometryHeader* header = static_cast<GeometryHeader*>(geometryMMF);
    
    vertexCount = header->vertexCount;
    triangleCount = header->triangleCount;
    categoryCount = header->categoryCount;
    
    std::cout << "[MMF] Geometría cargada:" << std::endl;
    std::cout << "  - Vertices: " << vertexCount << std::endl;
    std::cout << "  - Triangles: " << triangleCount << std::endl;
    std::cout << "  - Categories: " << categoryCount << std::endl;

    // --- DEPURACIÓN NUMÉRICA EXHAUSTIVA ---
    
    std::cout << "[DEBUG] Offsets leídos del Header:" << std::endl;
    std::cout << "  - verticesOffset: " << header->verticesOffset << std::endl;
    std::cout << "  - indicesOffset: " << header->indicesOffset << std::endl;
    std::cout << "  - normalsOffset: " << header->normalsOffset << std::endl;
    std::cout << "  - elementIdsOffset: " << header->elementIdsOffset << std::endl;
    std::cout << "  - categoryIdsOffset: " << header->categoryIdsOffset << std::endl;
    std::cout << "  - categoryMappingOffset: " << header->categoryMappingOffset << std::endl;

    char* basePtr = static_cast<char*>(geometryMMF);
    float* h_vertices = reinterpret_cast<float*>(basePtr + header->verticesOffset);
    int* h_triangles = reinterpret_cast<int*>(basePtr + header->indicesOffset);
    float* h_normals = reinterpret_cast<float*>(basePtr + header->normalsOffset);
    int* h_elementIds = reinterpret_cast<int*>(basePtr + header->elementIdsOffset);
    int* h_categoryIds = reinterpret_cast<int*>(basePtr + header->categoryIdsOffset);

    size_t vertexDataSize = (size_t)vertexCount * 3 * sizeof(float);
    size_t triangleDataSize = (size_t)triangleCount * 3 * sizeof(int);
    size_t normalDataSize = (size_t)vertexCount * 3 * sizeof(float);
    size_t idDataSize = (size_t)vertexCount * sizeof(int);

    std::cout << "[DEBUG] Tamaños calculados en bytes:" << std::endl;
    std::cout << "  - vertexDataSize: " << vertexDataSize << std::endl;
    std::cout << "  - triangleDataSize: " << triangleDataSize << std::endl;
    std::cout << "  - normalDataSize: " << normalDataSize << std::endl;
    std::cout << "  - idDataSize: " << idDataSize << std::endl;

    // Esto nos dirá si la lectura se saldría del archivo.
    // Necesitamos el tamaño total del MMF. Vamos a obtenerlo.
    MEMORY_BASIC_INFORMATION mbi;
    VirtualQuery(geometryMMF, &mbi, sizeof(mbi));
    size_t mmfSize = mbi.RegionSize;
    std::cout << "[DEBUG] Tamaño total del MMF detectado: " << mmfSize << " bytes" << std::endl;

    std::cout << "[DEBUG] Verificando límites de lectura:" << std::endl;
    std::cout << "  - Lectura de vértices termina en: " << (header->verticesOffset + vertexDataSize) 
              << " (OK si <= " << mmfSize << ")" << std::endl;
    std::cout << "  - Lectura de índices termina en: " << (header->indicesOffset + triangleDataSize) 
              << " (OK si <= " << mmfSize << ")" << std::endl;
    std::cout << "  - Lectura de normales termina en: " << (header->normalsOffset + normalDataSize) 
              << " (OK si <= " << mmfSize << ")" << std::endl;
    std::cout << "  - Lectura de elementIds termina en: " << (header->elementIdsOffset + idDataSize) 
              << " (OK si <= " << mmfSize << ")" << std::endl;
    std::cout << "  - Lectura de categoryIds termina en: " << (header->categoryIdsOffset + idDataSize) 
              << " (OK si <= " << mmfSize << ")" << std::endl;

    // --- FIN DE LA DEPURACIÓN NUMÉRICA ---
    
    std::cout << "[DEBUG] Paso 3: Reservando memoria en GPU (cudaMalloc)..." << std::endl;
    CUDA_CHECK(cudaMalloc(&d_vertices, vertexDataSize));
    CUDA_CHECK(cudaMalloc(&d_triangles, triangleDataSize));
    CUDA_CHECK(cudaMalloc(&d_normals, normalDataSize));
    CUDA_CHECK(cudaMalloc(&d_elementIds, idDataSize));
    CUDA_CHECK(cudaMalloc(&d_categoryIds, idDataSize));
    std::cout << "[DEBUG] Paso 3: OK" << std::endl;

    std::cout << "[DEBUG] Paso 4: Copiando datos a la GPU (cudaMemcpy)..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_vertices, h_vertices, vertexDataSize, cudaMemcpyHostToDevice));
    std::cout << "[DEBUG]   - d_vertices OK" << std::endl;
    CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles, triangleDataSize, cudaMemcpyHostToDevice));
    std::cout << "[DEBUG]   - d_triangles OK" << std::endl;
    CUDA_CHECK(cudaMemcpy(d_normals, h_normals, normalDataSize, cudaMemcpyHostToDevice));
    std::cout << "[DEBUG]   - d_normals OK" << std::endl;
    CUDA_CHECK(cudaMemcpy(d_elementIds, h_elementIds, idDataSize, cudaMemcpyHostToDevice));
    std::cout << "[DEBUG]   - d_elementIds OK" << std::endl;
    CUDA_CHECK(cudaMemcpy(d_categoryIds, h_categoryIds, idDataSize, cudaMemcpyHostToDevice));
    std::cout << "[DEBUG]   - d_categoryIds OK" << std::endl;
    std::cout << "[DEBUG] Paso 4: OK" << std::endl;
    
    std::cout << "[RENDERER] Datos copiados a GPU" << std::endl;
}

void WabiSabiRenderer::LoadColorMappingCSV() {
    std::cout << "[CSV] Cargando mapeo de colores desde: " << config.csvPath << std::endl;
    
    auto colorList = CSVReader::ReadCategoryColors(config.csvPath);
    
    h_categoryColors.assign(256, make_float3(0.5f, 0.5f, 0.5f)); // Rellena con gris por defecto

    // Mapear categorías en el orden del CSV
    // Asumimos que el plugin de Revit asigna IDs 0, 1, 2... que corresponden
    // al orden de las categorías en el archivo CSV (después del encabezado).
    // NOTA: Se ignora la primera entrada ("Category", "Color") si está presente.
    int categoryIndex = 0;
    for (const auto& entry : colorList) {
        // Simple heurística para saltar la fila del header si el lector la incluyó
        if (entry.category == "Category") continue; 
        
        if (categoryIndex < 256) {
            h_categoryColors[categoryIndex] = entry.color;
            categoryIndex++;
        }
    }
    
    // Copiar a GPU
    CUDA_CHECK(cudaMalloc(&d_categoryColors, 256 * sizeof(float3)));
    CUDA_CHECK(cudaMemcpy(d_categoryColors, h_categoryColors.data(), 
                          256 * sizeof(float3), cudaMemcpyHostToDevice));
}

void WabiSabiRenderer::RenderAllMaps(const CameraData& camera) {
    if (!d_vertices || vertexCount == 0 || triangleCount == 0) {
        std::cerr << "[RENDERER] No hay geometría para renderizar" << std::endl;
        return;
    }
    
    // Configuración para kernels
    dim3 blockSize(config.blockSizeX, config.blockSizeY);
    dim3 gridSize = calculateGridSize(config.width, config.height, blockSize);
    
    // Configuración de render para kernels - convertir de WabiSabiRenderer::RenderConfig a ::RenderConfig
    ::RenderConfig kernelConfig;
    kernelConfig.width = config.width;
    kernelConfig.height = config.height;
    kernelConfig.minDepth = config.minDepth;
    kernelConfig.maxDepth = config.maxDepth;
    
    kernelConfig.depthThreshold = config.depthThreshold;
    kernelConfig.normalThreshold = config.normalThreshold;
    
    // Convertir WabiSabiRenderer::CameraData a ::CameraData
    ::CameraData kernelCamera;
    kernelCamera.eyePosition = camera.eyePosition;
    kernelCamera.lower_left_corner = camera.lower_left_corner;
    kernelCamera.horizontal_vec = camera.horizontal_vec;
    kernelCamera.vertical_vec = camera.vertical_vec;
    
    // Crear streams CUDA para paralelización
    cudaStream_t depthStream, normalStream, linesStream, segmentStream;
    CUDA_CHECK(cudaStreamCreate(&depthStream));
    CUDA_CHECK(cudaStreamCreate(&normalStream));
    CUDA_CHECK(cudaStreamCreate(&linesStream));
    CUDA_CHECK(cudaStreamCreate(&segmentStream));
    
    // Lanzar kernels en paralelo
    if (config.enableDepth) {
        LaunchDepthKernel(
            reinterpret_cast<float3*>(d_vertices),
            vertexCount,
            d_triangles,
            triangleCount,
            kernelCamera, 
            kernelConfig,
            d_depthMap,
            depthStream
        );
    }
    
    if (config.enableNormals) {
        LaunchNormalKernel(
            reinterpret_cast<float3*>(d_vertices),
            vertexCount,
            d_triangles,
            reinterpret_cast<float3*>(d_normals),
            triangleCount,
            kernelCamera, 
            kernelConfig,
            reinterpret_cast<float3*>(d_normalMap),
            normalStream
        );
    }
    
    if (config.enableLines) {
    // La llamada ahora es simple y solo depende del mapa de normales.
    LaunchLinesKernel(
        reinterpret_cast<float3*>(d_normalMap), // <-- SOLO PASAR EL MAPA DE NORMALES
        kernelConfig,
        d_linesMap,
        linesStream
    );
}
    
    if (config.enableSegmentation) {
        LaunchSegmentationKernel(
            reinterpret_cast<float3*>(d_vertices),
            vertexCount,
            d_triangles,
            d_categoryIds,
            d_categoryColors,
            triangleCount,
            categoryCount, // <--- ARGUMENTO DESCOMENTADO Y RESTAURADO
            kernelCamera, 
            kernelConfig,
            reinterpret_cast<float4*>(d_segmentationMap),
            segmentStream
        );
    }
    
    // Sincronizar todos los streams
    CUDA_CHECK(cudaStreamSynchronize(depthStream));
    CUDA_CHECK(cudaStreamSynchronize(normalStream));
    CUDA_CHECK(cudaStreamSynchronize(linesStream));
    CUDA_CHECK(cudaStreamSynchronize(segmentStream));
    
    // Limpiar streams
    CUDA_CHECK(cudaStreamDestroy(depthStream));
    CUDA_CHECK(cudaStreamDestroy(normalStream));
    CUDA_CHECK(cudaStreamDestroy(linesStream));
    CUDA_CHECK(cudaStreamDestroy(segmentStream));
    
    totalFrames++;
}

void WabiSabiRenderer::SaveMapsToFiles() {
    std::filesystem::create_directories(config.outputPath);
    
    size_t pixelCount = config.width * config.height;
    
    // Búferes temporales en el host (CPU)
    std::vector<float> h_buffer(pixelCount * 4);
    std::vector<unsigned char> imageBuffer(pixelCount * 4);

    // --- ¡LA LÍNEA MÁGICA! ---
    // Le decimos a la librería stb que debe voltear verticalmente todas las imágenes que guarde.
    stbi_flip_vertically_on_write(1); // 1 = true

    // --- GUARDAR MAPA DE PROFUNDIDAD (lógica simplificada) ---
    if (config.enableDepth) {
        CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_depthMap, 
                              pixelCount * sizeof(float), cudaMemcpyDeviceToHost));
        
        for (size_t i = 0; i < pixelCount; i++) {
            float distance = h_buffer[i];
            unsigned char pixel_value = 255; // Fondo blanco por defecto
            if (distance < config.maxDepth) {
                float normalized = 1.0f - fmaxf(0.0f, fminf(1.0f, (distance - config.minDepth) / (config.maxDepth - config.minDepth)));
                pixel_value = static_cast<unsigned char>(normalized * 255.0f);
            }
            imageBuffer[i] = pixel_value;
        }
        std::string path = config.outputPath + "/" + config.depthFilename;
        stbi_write_png(path.c_str(), config.width, config.height, 1, imageBuffer.data(), config.width);
    }
    
    // --- GUARDAR MAPA DE NORMALES (lógica simplificada) ---
    if (config.enableNormals) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(h_buffer.data()), d_normalMap,
                              pixelCount * sizeof(float3), cudaMemcpyDeviceToHost));
        
        convertFloat3ToRGB(reinterpret_cast<const float3*>(h_buffer.data()), imageBuffer.data(), pixelCount);
        
        std::string path = config.outputPath + "/" + config.normalFilename;
        stbi_write_png(path.c_str(), config.width, config.height, 3, imageBuffer.data(), config.width * 3);
    }
    
    // --- GUARDAR MAPA DE LÍNEAS (lógica simplificada) ---
    if (config.enableLines) {
        CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_linesMap, 
                              pixelCount * sizeof(float), cudaMemcpyDeviceToHost));

        convertFloatToUchar(h_buffer.data(), imageBuffer.data(), pixelCount, false);

        std::string path = config.outputPath + "/" + config.linesFilename;
        stbi_write_png(path.c_str(), config.width, config.height, 1, imageBuffer.data(), config.width);
    }
    
    // --- GUARDAR MAPA DE SEGMENTACIÓN (lógica simplificada) ---
    if (config.enableSegmentation) {
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(h_buffer.data()), d_segmentationMap,
                              pixelCount * sizeof(float4), cudaMemcpyDeviceToHost));

        convertFloat4ToRGBA(reinterpret_cast<const float4*>(h_buffer.data()), imageBuffer.data(), pixelCount);
        
        std::string path = config.outputPath + "/" + config.segmentationFilename;
        stbi_write_png(path.c_str(), config.width, config.height, 4, imageBuffer.data(), config.width * 4);
    }
}
std::string WabiSabiRenderer::FindLatestWabiSabiMMF() {
    std::cout << "[MMF] Buscando Memory Mapped File de WabiSabi..." << std::endl;
    
    // Primero intentar con el nombre fijo
    std::string fixedName = "WabiSabi_Geometry_Cache";
    
    HANDLE hMapFile = OpenFileMappingA(
        FILE_MAP_READ,
        FALSE,
        fixedName.c_str()
    );
    
    if (hMapFile != NULL) {
        CloseHandle(hMapFile);
        std::cout << "[MMF] Encontrado MMF: " << fixedName << std::endl;
        return fixedName;
    }
    
    // Si no se encuentra, buscar archivo de estado
    try {
        std::string appDataPath = std::string(getenv("LOCALAPPDATA"));
        std::string stateFilePath = appDataPath + "\\WabiSabiBridge\\GeometryCache\\wabisabi_state.json";
        
        if (std::filesystem::exists(stateFilePath)) {
            std::ifstream stateFile(stateFilePath);
            if (stateFile.is_open()) {
                Json::Value root;
                Json::CharReaderBuilder builder;
                std::string errors;
                
                if (Json::parseFromStream(builder, stateFile, &root, &errors)) {
                    std::string mmfName = root.get("MmfName", "").asString();
                    
                    if (!mmfName.empty()) {
                        // Verificar que el MMF existe
                        HANDLE hTest = OpenFileMappingA(
                            FILE_MAP_READ,
                            FALSE,
                            mmfName.c_str()
                        );
                        
                        if (hTest != NULL) {
                            CloseHandle(hTest);
                            std::cout << "[MMF] Encontrado MMF desde estado: " << mmfName << std::endl;
                            return mmfName;
                        }
                    }
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[MMF] Error leyendo archivo de estado: " << e.what() << std::endl;
    }
    
    throw std::runtime_error("No se encontró ningún Memory Mapped File de WabiSabi");
}

// Implementación del resto de métodos...
void WabiSabiRenderer::Start() {
    isRunning = true;
    renderThread = std::thread(&WabiSabiRenderer::RenderLoop, this);
    journalWatcher = std::thread(&WabiSabiRenderer::WatchJournal, this);
    
    if (config.enableWebViewer) {
        StartWebSocketServer();
    }
}

void WabiSabiRenderer::RenderLoop() {
    std::cout << "[RENDERER] Bucle de renderizado en espera de la primera cámara..." << std::endl;
    {
        std::unique_lock<std::mutex> lock(cameraMutex);
        // El hilo esperará aquí hasta que cameraInitialized sea true.
        cameraCv.wait(lock, [this]{ return cameraInitialized; });
    }
    std::cout << "[RENDERER] Cámara inicializada. Iniciando renderizado de fotogramas." << std::endl;
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    while (isRunning) {
        auto frameStart = std::chrono::steady_clock::now();
        
        // Obtener última posición de cámara
        CameraData camera = GetLatestCamera();
        
        // Renderizar los 4 mapas
        RenderAllMaps(camera);
        
        // Guardar a archivos PNG
        SaveMapsToFiles();
        
        // Enviar a WebSocket si está conectado
        if (wsConnected) {
            SendMapsToWebSocket();
        }
        
        auto frameEnd = std::chrono::steady_clock::now();
        auto frameDuration = std::chrono::duration_cast<std::chrono::milliseconds>(
            frameEnd - frameStart);
        
        // Actualizar estadísticas
        avgFrameTime = frameDuration.count();
        
        // Calcular FPS
        auto timeSinceLastFrame = std::chrono::duration_cast<std::chrono::milliseconds>(
            frameEnd - lastFrameTime);
        if (timeSinceLastFrame.count() > 0) {
            currentFPS = 1000.0f / timeSinceLastFrame.count();
        }
        lastFrameTime = frameEnd;
        
        // Limitar FPS
        int targetFrameTime = 1000 / config.maxFPS;
        if (frameDuration.count() < targetFrameTime) {
            std::this_thread::sleep_for(
                std::chrono::milliseconds(targetFrameTime - frameDuration.count()));
        }
    }
}

// Implementar el resto de métodos existentes...
WabiSabiRenderer::~WabiSabiRenderer() {
    Stop();
    
    // Limpiar buffers CUDA
    if (d_vertices) cudaFree(d_vertices);
    if (d_triangles) cudaFree(d_triangles);
    if (d_normals) cudaFree(d_normals);
    if (d_elementIds) cudaFree(d_elementIds);
    if (d_categoryIds) cudaFree(d_categoryIds);
    if (d_categoryColors) cudaFree(d_categoryColors);
    if (d_depthMap) cudaFree(d_depthMap);
    if (d_normalMap) cudaFree(d_normalMap);
    if (d_linesMap) cudaFree(d_linesMap);
    if (d_segmentationMap) cudaFree(d_segmentationMap);
    
    // Cerrar MMF
    if (geometryMMF) {
        UnmapViewOfFile(geometryMMF);
        geometryMMF = nullptr;
    }
}

void WabiSabiRenderer::Stop() {
    if (!isRunning) return;
    
    isRunning = false;
    
    if (renderThread.joinable()) {
        renderThread.join();
    }
    
    if (journalWatcher.joinable()) {
        journalWatcher.join();
    }
    
    std::cout << "[RENDERER] Detenido" << std::endl;
}

void WabiSabiRenderer::WatchJournal() {
    // Este método está vacío porque el journal se maneja desde main.cpp
    while (isRunning) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void WabiSabiRenderer::StartWebSocketServer() {
    // TODO: Implementar servidor WebSocket
    std::cout << "[WebSocket] Servidor no implementado aún" << std::endl;
}

void WabiSabiRenderer::SendMapsToWebSocket() {
    // TODO: Enviar mapas por WebSocket
}

WabiSabiRenderer::CameraData WabiSabiRenderer::GetLatestCamera() {
    std::lock_guard<std::mutex> lock(cameraMutex);
    return currentCamera;
}

void WabiSabiRenderer::UpdateCamera(const WabiSabiRenderer::CameraData& camera) {
    {
    std::lock_guard<std::mutex> lock(cameraMutex);
        currentCamera = camera;

        // Si es la primera vez que se inicializa la cámara, notifícalo.
        if (!cameraInitialized) {
            cameraInitialized = true;
            std::cout << "[RENDERER] ¡Primera cámara recibida! Desbloqueando el bucle de renderizado." << std::endl;
        }
    }
    cameraCv.notify_one(); // Notifica al hilo de renderizado que la cámara está lista.
}

WabiSabiRenderer::Statistics WabiSabiRenderer::GetStatistics() const {
    Statistics stats;
    stats.fps = currentFPS.load();
    stats.avgFrameTime = avgFrameTime.load();
    stats.totalFrames = totalFrames.load();
    return stats;
}