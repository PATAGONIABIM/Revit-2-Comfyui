// WabiSabiRenderer.cpp - Implementación completa con renderizado real
#include "core/WabiSabiRenderer.h"
#include "../cuda/CudaHelpers.h"
#include "../cuda/WabiSabiKernels.cuh"
#include "utils/Base64.h"
#include "utils/CSVReader.h"
#include <Windows.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <future> // Necesario para std::async
#include <iostream>
#include <json/json.h>
#include <shared_mutex> // <-- NECESARIO PARA std::shared_mutex

// Para guardar PNG
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../external/stb_image_write.h"

// Estructura del header en el MMF
#pragma pack(push, 1)
struct GeometryHeader {
  int32_t vertexCount;
  int32_t triangleCount;
  int32_t categoryCount; // <-- CAMPO NUEVO
  int64_t verticesOffset;
  int64_t indicesOffset;
  int64_t normalsOffset;
  int64_t elementIdsOffset;      // <-- CAMPO NUEVO
  int64_t categoryIdsOffset;     // <-- CAMPO NUEVO
  int64_t categoryMappingOffset; // <-- CAMPO NUEVO
};
#pragma pack(pop)

WabiSabiRenderer::WabiSabiRenderer(const RenderConfig &config)
    : config(config), isRunning(false), totalFrames(0), currentFPS(0),
      avgFrameTime(0) {

  std::cout << "[RENDERER] Inicializando WabiSabiRenderer..." << std::endl;

  CUDA_CHECK(cudaSetDevice(config.cudaDevice));
  printCudaDeviceInfo(config.cudaDevice);

  // 1. Calcular cantidad de píxeles según resolución
  size_t pixelCount = config.width * config.height;

  // 2. Alocar buffers de salida en GPU
  CUDA_CHECK(cudaMalloc(&d_depthMap, pixelCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_normalMap, pixelCount * 3 * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_linesMap, pixelCount * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_segmentationMap, pixelCount * 4 * sizeof(float)));

  // --- INSERTAR ESTA LÍNEA AQUÍ ---
  CUDA_CHECK(cudaMalloc(&d_elementIdPixelMap, pixelCount * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_zBuffer, pixelCount * sizeof(unsigned long long)));
  // --------------------------------

  // Inicializar MMF Writer
  if (config.enableMMF) {
    mmfWriter = std::make_unique<ImageMMFWriter>();
    if (!mmfWriter->Initialize(config.width, config.height)) {
      std::cerr
          << "[RENDERER] Warning: Failed to initialize Shared Memory Writer."
          << std::endl;
    }
  }

  // La carga inicial de geometría y colores
  ReloadGeometry();

  std::cout << "[RENDERER] Inicialización completa" << std::endl;
}

void WabiSabiRenderer::CheckForUpdates() {
  try {
    std::string notificationFilePath = config.outputPath + "/last_update.txt";
    if (!std::filesystem::exists(notificationFilePath)) {
      return; // Si no hay archivo, no hay nada que hacer
    }

    std::ifstream file(notificationFilePath);
    std::string currentTimestamp;
    file >> currentTimestamp;
    file.close();

    if (currentTimestamp != lastKnownTimestamp) {
      std::cout << "\n[UPDATE] ¡Cambio detectado! Nueva marca de tiempo: "
                << currentTimestamp << std::endl;
      lastKnownTimestamp = currentTimestamp;
      ReloadGeometry();
      std::cout << "[UPDATE] Geometría recargada exitosamente.\n" << std::endl;
    }
  } catch (const std::exception &e) {
    std::cerr << "[ERROR] Fallo al comprobar actualizaciones: " << e.what()
              << std::endl;
  }
}

void WabiSabiRenderer::ReloadGeometry() {
  // --- PASO 1: Liberar recursos antiguos ---
  if (d_vertices)
    cudaFree(d_vertices);
  if (d_triangles)
    cudaFree(d_triangles);
  if (d_normals)
    cudaFree(d_normals);
  if (d_elementIds)
    cudaFree(d_elementIds);
  if (d_categoryIds)
    cudaFree(d_categoryIds);
  d_vertices = nullptr;
  d_triangles = nullptr;
  d_normals = nullptr;
  d_elementIds = nullptr;
  d_categoryIds = nullptr;
  if (geometryMMF) {
    UnmapViewOfFile(geometryMMF);
    geometryMMF = nullptr;
  }

  // --- PASO 2: Cargar nueva geometría desde el MMF ---
  std::cout << "[RELOAD] Abriendo MMF y copiando datos a la GPU..."
            << std::endl;
  OpenGeometryMMF(); // Esto lee el header y carga los datos de geometría.

  // --- PASO 3 (LA SOLUCIÓN): Reconstruir la tabla de colores AHORA ---
  // Usando el categoryIndexToNameMap que se acaba de cargar desde el MMF.
  // Esto se ejecutará tanto en el inicio como en cada actualización.
  LoadColorMappingCSV();
}

void WabiSabiRenderer::OpenGeometryMMF() {
  // Buscar el MMF más reciente de WabiSabi
  std::string mmfName = FindLatestWabiSabiMMF();

  HANDLE hMapFile =
      OpenFileMappingA(FILE_MAP_ALL_ACCESS, FALSE, mmfName.c_str());

  if (hMapFile == NULL) {
    throw std::runtime_error("No se pudo abrir el MMF de geometría: " +
                             mmfName);
  }

  geometryMMF = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, 0);

  if (!geometryMMF) {
    CloseHandle(hMapFile);
    throw std::runtime_error("No se pudo mapear el MMF");
  }

  // Leer header para obtener offsets y configurar punteros
  ReadGeometryHeader();

  CloseHandle(hMapFile); // Ya no necesitamos el handle
}

void WabiSabiRenderer::ReadGeometryHeader() {
  if (!geometryMMF)
    return;

  // Leer header desde el inicio del MMF
  GeometryHeader *header = static_cast<GeometryHeader *>(geometryMMF);

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
  std::cout << "  - elementIdsOffset: " << header->elementIdsOffset
            << std::endl;
  std::cout << "  - categoryIdsOffset: " << header->categoryIdsOffset
            << std::endl;
  std::cout << "  - categoryMappingOffset: " << header->categoryMappingOffset
            << std::endl;

  char *basePtr = static_cast<char *>(geometryMMF);

  // --- INICIO DE LA MODIFICACIÓN ---
  // Leer el mapa de categorías desde el MMF
  char *mapPtr = basePtr + header->categoryMappingOffset;

  // Leer el número de entradas en el mapa
  int mapEntryCount = *reinterpret_cast<int *>(mapPtr);
  mapPtr += sizeof(int);

  categoryIndexToNameMap.clear();
  std::cout << "[MMF] Leyendo mapa de categorías..." << std::endl;
  for (int i = 0; i < mapEntryCount; ++i) {
    // Leer el índice compacto
    int compactIndex = *reinterpret_cast<int *>(mapPtr);
    mapPtr += sizeof(int);

    // --- INICIO DE LA MODIFICACIÓN ---
    // Eliminar el bucle do-while complejo.

    // 1. Leer la longitud de la cadena como un int estándar.
    int stringLength = *reinterpret_cast<int *>(mapPtr);
    mapPtr += sizeof(int);

    // 2. Leer exactamente esa cantidad de bytes para formar la cadena.
    std::string categoryName(mapPtr, stringLength);
    mapPtr += stringLength;
    // --- FIN DE LA MODIFICACIÓN ---

    categoryIndexToNameMap[compactIndex] = categoryName;
    std::cout << "  - Mapeo leído: " << compactIndex << " -> " << categoryName
              << std::endl;
  }

  float *h_vertices =
      reinterpret_cast<float *>(basePtr + header->verticesOffset);
  int *h_triangles = reinterpret_cast<int *>(basePtr + header->indicesOffset);
  float *h_normals = reinterpret_cast<float *>(basePtr + header->normalsOffset);
  int *h_elementIds =
      reinterpret_cast<int *>(basePtr + header->elementIdsOffset);
  int *h_categoryIds =
      reinterpret_cast<int *>(basePtr + header->categoryIdsOffset);

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
  std::cout << "[DEBUG] Tamaño total del MMF detectado: " << mmfSize << " bytes"
            << std::endl;

  std::cout << "[DEBUG] Verificando límites de lectura:" << std::endl;
  std::cout << "  - Lectura de vértices termina en: "
            << (header->verticesOffset + vertexDataSize)
            << " (OK si <= " << mmfSize << ")" << std::endl;
  std::cout << "  - Lectura de índices termina en: "
            << (header->indicesOffset + triangleDataSize)
            << " (OK si <= " << mmfSize << ")" << std::endl;
  std::cout << "  - Lectura de normales termina en: "
            << (header->normalsOffset + normalDataSize)
            << " (OK si <= " << mmfSize << ")" << std::endl;
  std::cout << "  - Lectura de elementIds termina en: "
            << (header->elementIdsOffset + idDataSize)
            << " (OK si <= " << mmfSize << ")" << std::endl;
  std::cout << "  - Lectura de categoryIds termina en: "
            << (header->categoryIdsOffset + idDataSize)
            << " (OK si <= " << mmfSize << ")" << std::endl;

  // --- FIN DE LA DEPURACIÓN NUMÉRICA ---

  std::cout << "[DEBUG] Paso 3: Reservando memoria en GPU (cudaMalloc)..."
            << std::endl;
  CUDA_CHECK(cudaMalloc(&d_vertices, vertexDataSize));
  CUDA_CHECK(cudaMalloc(&d_triangles, triangleDataSize));
  CUDA_CHECK(cudaMalloc(&d_normals, normalDataSize));
  CUDA_CHECK(cudaMalloc(&d_elementIds, idDataSize));
  CUDA_CHECK(cudaMalloc(&d_categoryIds, idDataSize));
  std::cout << "[DEBUG] Paso 3: OK" << std::endl;

  std::cout << "[DEBUG] Paso 4: Copiando datos a la GPU (cudaMemcpy)..."
            << std::endl;
  CUDA_CHECK(cudaMemcpy(d_vertices, h_vertices, vertexDataSize,
                        cudaMemcpyHostToDevice));
  std::cout << "[DEBUG]   - d_vertices OK" << std::endl;
  CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles, triangleDataSize,
                        cudaMemcpyHostToDevice));
  std::cout << "[DEBUG]   - d_triangles OK" << std::endl;
  CUDA_CHECK(
      cudaMemcpy(d_normals, h_normals, normalDataSize, cudaMemcpyHostToDevice));
  std::cout << "[DEBUG]   - d_normals OK" << std::endl;
  CUDA_CHECK(cudaMemcpy(d_elementIds, h_elementIds, idDataSize,
                        cudaMemcpyHostToDevice));
  std::cout << "[DEBUG]   - d_elementIds OK" << std::endl;
  CUDA_CHECK(cudaMemcpy(d_categoryIds, h_categoryIds, idDataSize,
                        cudaMemcpyHostToDevice));
  std::cout << "[DEBUG]   - d_categoryIds OK" << std::endl;
  std::cout << "[DEBUG] Paso 4: OK" << std::endl;

  std::cout << "[RENDERER] Datos copiados a GPU" << std::endl;
}

void WabiSabiRenderer::LoadColorMappingCSV() {
  std::cout << "[CSV] Cargando colores desde: " << config.csvPath << std::endl;

  // --- INICIO DE LA MODIFICACIÓN ---

  // 1. Cargar los colores del CSV a un mapa de [Nombre -> Color]
  auto colorsFromFile = CSVReader::ReadCategoryColors(config.csvPath);
  std::unordered_map<std::string, float3> nameToColorMap;
  for (const auto &entry : colorsFromFile) {
    nameToColorMap[entry.category] = entry.color;
  }

  // 2. Construir la tabla final de colores para la GPU (h_categoryColors)
  //    usando el mapa que leímos del MMF.
  h_categoryColors.assign(
      256, make_float3(0.5f, 0.5f, 0.5f)); // Rellena con gris por defecto

  // Iterar sobre nuestro mapa de [índice -> nombre] leído del MMF
  for (const auto &kvp : categoryIndexToNameMap) {
    int compactIndex = kvp.first;
    const std::string &categoryName = kvp.second;

    if (compactIndex < 256) {
      // Buscar el color para este nombre de categoría en el mapa del CSV
      auto it = nameToColorMap.find(categoryName);
      if (it != nameToColorMap.end()) {
        // Si se encuentra, colocar el color en la posición correcta del array
        h_categoryColors[compactIndex] = it->second;
      } else {
        // Si no se encuentra, se quedará el color gris por defecto.
        std::cout << "[Advertencia] No se encontró color para la categoría '"
                  << categoryName << "' (Índice " << compactIndex
                  << ") en el CSV." << std::endl;
      }
    }
  }
  // --- FIN DE LA MODIFICACIÓN ---

  // Copiar el array final y ordenado a la GPU
  CUDA_CHECK(cudaMalloc(&d_categoryColors, 256 * sizeof(float3)));
  CUDA_CHECK(cudaMemcpy(d_categoryColors, h_categoryColors.data(),
                        256 * sizeof(float3), cudaMemcpyHostToDevice));

  std::cout << "[CSV] Tabla de colores para GPU creada y transferida."
            << std::endl;
}

// Actualizado: acepta config como parámetro
void WabiSabiRenderer::RenderAllMaps(
    const WabiSabiRenderer::CameraData &camera,
    const WabiSabiRenderer::RenderConfig &localConfig, cudaStream_t stream) {
  if (!d_vertices || vertexCount == 0 || triangleCount == 0)
    return;

  // Configuración para el kernel (usando tipos globales de WabiSabiKernels.cuh)
  ::RenderConfig kConf = {
      localConfig.width,          localConfig.height,
      localConfig.minDepth,       localConfig.maxDepth,
      localConfig.depthThreshold, localConfig.normalThreshold};

  ::CameraData kCam = {camera.eyePosition, camera.lower_left_corner,
                       camera.horizontal_vec, camera.vertical_vec};

  // LLAMADA AL NUEVO MEGA-KERNEL (Reemplaza a las 4 llamadas anteriores)
  LaunchAllMapsKernel(
      reinterpret_cast<float3 *>(d_vertices), vertexCount, d_triangles,
      reinterpret_cast<float3 *>(d_normals), d_elementIds, d_categoryIds,
      d_categoryColors, triangleCount, categoryCount, kCam, kConf, d_depthMap,
      reinterpret_cast<float3 *>(d_normalMap), d_elementIdPixelMap,
      reinterpret_cast<float4 *>(d_segmentationMap),
      d_zBuffer, // <--- Passed here
      stream);

  // Sincronizar para poder procesar las líneas después (dentro del stream)
  // No necesitamos sincronizar CPU-GPU aquí, solo la coherencia del stream
  // Pero LaunchLinesKernel depende de los resultados anteriores en d_normalMap
  // y d_elementIdPixelMap

  if (localConfig.enableLines) {
    LaunchLinesKernel(reinterpret_cast<float3 *>(d_normalMap), d_depthMap,
                      d_elementIdPixelMap, kConf, d_linesMap, stream);
  }
}

std::string WabiSabiRenderer::FindLatestWabiSabiMMF() {
  std::cout << "[MMF] Buscando Memory Mapped File de WabiSabi..." << std::endl;

  // Primero intentar con el nombre fijo
  std::string fixedName = "WabiSabi_Geometry_Cache";

  HANDLE hMapFile = OpenFileMappingA(FILE_MAP_READ, FALSE, fixedName.c_str());

  if (hMapFile != NULL) {
    CloseHandle(hMapFile);
    std::cout << "[MMF] Encontrado MMF: " << fixedName << std::endl;
    return fixedName;
  }

  // Si no se encuentra, buscar archivo de estado
  try {
    std::string appDataPath = std::string(getenv("LOCALAPPDATA"));
    std::string stateFilePath =
        appDataPath + "\\WabiSabiBridge\\GeometryCache\\wabisabi_state.json";

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
            HANDLE hTest =
                OpenFileMappingA(FILE_MAP_READ, FALSE, mmfName.c_str());

            if (hTest != NULL) {
              CloseHandle(hTest);
              std::cout << "[MMF] Encontrado MMF desde estado: " << mmfName
                        << std::endl;
              return mmfName;
            }
          }
        }
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "[MMF] Error leyendo archivo de estado: " << e.what()
              << std::endl;
  }

  throw std::runtime_error(
      "No se encontró ningún Memory Mapped File de WabiSabi");
}

void WabiSabiRenderer::Start() {
  isRunning = true;
  renderThread = std::thread(&WabiSabiRenderer::RenderLoop, this);
  journalWatcher = std::thread(&WabiSabiRenderer::WatchJournal, this);

  if (config.enableWebViewer) {
    StartWebSocketServer();
  }
}

// Actualizado: RenderLoop con copia local de configuración y bloqueo seguro

void WabiSabiRenderer::RenderLoop() {
  std::cout << "[RENDERER] Bucle iniciado con optimización de Kernel Combinado "
               "y Grabado Asíncrono."
            << std::endl;

  {
    std::unique_lock<std::mutex> lock(cameraMutex);
    cameraCv.wait(lock, [this] { return cameraInitialized; });
  }

  auto lastFrameTime = std::chrono::steady_clock::now();

  while (isRunning) {
    CheckForUpdates();
    auto frameStart = std::chrono::steady_clock::now();

    CameraData camera = GetLatestCamera();
    RenderConfig localConfig;
    {
      std::shared_lock<std::shared_mutex> lock(configMutex);
      localConfig = config;
    }

    // --- PASO 1: CONFIGURAR ESTRUCTURAS PARA KERNELS ---
    ::RenderConfig kernelConfig;
    kernelConfig.width = localConfig.width;
    kernelConfig.height = localConfig.height;
    kernelConfig.minDepth = localConfig.minDepth;
    kernelConfig.maxDepth = localConfig.maxDepth;
    kernelConfig.depthThreshold = localConfig.depthThreshold;
    kernelConfig.normalThreshold = localConfig.normalThreshold;

    ::CameraData kernelCamera;
    kernelCamera.eyePosition = camera.eyePosition;
    kernelCamera.lower_left_corner = camera.lower_left_corner;
    kernelCamera.horizontal_vec = camera.horizontal_vec;
    kernelCamera.vertical_vec = camera.vertical_vec;

    // --- PASO 2: RENDERIZADO (RTX 3090 POWER) ---
    cudaStream_t renderStream;
    cudaStreamCreate(&renderStream);

    // LANZAMIENTO DEL KERNEL COMBINADO (Todo en un solo bucle de triángulos)
    RenderAllMaps(camera, localConfig,
                  renderStream); // <-- Usar la funcion miembro refactorizada

    // (Lines Kernel is called inside RenderAllMaps now)
    cudaStreamSynchronize(renderStream); // Wait for completion before copying

    // --- PASO 3: COPIADO RÁPIDO DE GPU A RAM ---
    // Esto es muy rápido, libera a la GPU casi al instante
    HostFrameData frameData = CopyGpuToHost(localConfig, camera.sequenceNumber);

    // --- PASO 4: GRABADO ASÍNCRONO (DISCO / MMF) ---

    // 4A: DISCO (Opcional, lento)
    if (localConfig.enableDiskIO) {
      std::thread([this, frameData, localConfig]() {
        this->SaveToDiskInternal(frameData, localConfig);
      }).detach();
    }

    // 4B: MMF (Rapidísimo)
    if (localConfig.enableMMF) {
      WriteToMMFInternal(frameData);
    }

    // Opcional: Enviar a WebSocket (también asíncrono si quieres más velocidad)
    if (localConfig.enableWebViewer && wsConnected) {
      SendMapsToWebSocket(localConfig);
    }

    cudaStreamDestroy(renderStream);

    // --- ESTADÍSTICAS ---
    auto frameEnd = std::chrono::steady_clock::now();
    avgFrameTime = std::chrono::duration_cast<std::chrono::milliseconds>(
                       frameEnd - frameStart)
                       .count();
    currentFPS =
        1000.0f / std::chrono::duration_cast<std::chrono::milliseconds>(
                      frameEnd - lastFrameTime)
                      .count();
    lastFrameTime = frameEnd;
    totalFrames++;

    int targetMs = 1000 / localConfig.maxFPS;
    if (avgFrameTime < targetMs) {
      std::this_thread::sleep_for(
          std::chrono::milliseconds(targetMs - (int)avgFrameTime));
    }
  }
}

// Función para mover datos de la GPU a la RAM del sistema lo más rápido posible
WabiSabiRenderer::HostFrameData WabiSabiRenderer::CopyGpuToHost(
    const WabiSabiRenderer::RenderConfig &localConfig, int64_t seqNum) {
  size_t pixelCount = localConfig.width * localConfig.height;
  HostFrameData data;
  data.sequenceNumber = seqNum; // <-- Asignar secuencia
  data.width = localConfig.width;
  data.height = localConfig.height;

  data.depth.resize(pixelCount);
  data.normals.resize(pixelCount);
  data.lines.resize(pixelCount);
  data.segmentation.resize(pixelCount);

  cudaMemcpy(data.depth.data(), d_depthMap, pixelCount * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data.normals.data(), d_normalMap, pixelCount * sizeof(float3),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data.lines.data(), d_linesMap, pixelCount * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(data.segmentation.data(), d_segmentationMap,
             pixelCount * sizeof(float4), cudaMemcpyDeviceToHost);

  return data;
}

void WabiSabiRenderer::WriteToMMFInternal(const HostFrameData &data) {
  if (mmfWriter) {
    // Enviar todos los buffers

    // 1. Flatten Normals (float3 -> float)
    std::vector<float> flattenedNormals;
    flattenedNormals.reserve(data.normals.size() * 3);
    for (const auto &n : data.normals) {
      flattenedNormals.push_back(n.x);
      flattenedNormals.push_back(n.y);
      flattenedNormals.push_back(n.z);
    }

    // 2. Flatten Segmentation (float4 -> float)
    std::vector<float> flattenedSegmentation;
    flattenedSegmentation.reserve(data.segmentation.size() * 4);
    for (const auto &s : data.segmentation) {
      flattenedSegmentation.push_back(s.x);
      flattenedSegmentation.push_back(s.y);
      flattenedSegmentation.push_back(s.z);
      flattenedSegmentation.push_back(s.w);
    }

    mmfWriter->WriteFrame(data.sequenceNumber, data.depth, flattenedNormals,
                          data.lines, flattenedSegmentation);

    std::cout << "[MMF WRITE] Seq: " << data.sequenceNumber << std::endl;
  }
}

// Esta función corre en un hilo separado de la CPU
void WabiSabiRenderer::SaveToDiskInternal(
    const HostFrameData &data,
    const WabiSabiRenderer::RenderConfig &localConfig) {
  size_t pixelCount = data.width * data.height;
  std::vector<unsigned char> img(pixelCount * 4);

  stbi_flip_vertically_on_write(1);

  // Guardar Depth
  if (localConfig.enableDepth) {
    for (size_t i = 0; i < pixelCount; i++) {
      float d = data.depth[i];
      float linear = fmaxf(
          0.0f, fminf(1.0f, (d - localConfig.minDepth) /
                                (localConfig.maxDepth - localConfig.minDepth)));
      img[i] = static_cast<unsigned char>(
          (1.0f - powf(linear, localConfig.gamma)) * 255.0f);
    }
    stbi_write_png(
        (localConfig.outputPath + "/" + localConfig.depthFilename).c_str(),
        data.width, data.height, 1, img.data(), data.width);
  }

  // Guardar Normals
  if (localConfig.enableNormals) {
    std::vector<unsigned char> imgRGB(pixelCount * 3);
    for (size_t i = 0; i < pixelCount; i++) {
      imgRGB[i * 3 + 0] = static_cast<unsigned char>(
          (data.normals[i].x * 0.5f + 0.5f) * 255.0f);
      imgRGB[i * 3 + 1] = static_cast<unsigned char>(
          (data.normals[i].y * 0.5f + 0.5f) * 255.0f);
      imgRGB[i * 3 + 2] = static_cast<unsigned char>(
          (data.normals[i].z * 0.5f + 0.5f) * 255.0f);
    }
    stbi_write_png(
        (localConfig.outputPath + "/" + localConfig.normalFilename).c_str(),
        data.width, data.height, 3, imgRGB.data(), data.width * 3);
  }

  // Guardar Lines
  if (localConfig.enableLines) {
    for (size_t i = 0; i < pixelCount; i++)
      img[i] = static_cast<unsigned char>(data.lines[i] * 255.0f);
    stbi_write_png(
        (localConfig.outputPath + "/" + localConfig.linesFilename).c_str(),
        data.width, data.height, 1, img.data(), data.width);
  }

  // Guardar Segmentation
  if (localConfig.enableSegmentation) {
    for (size_t i = 0; i < pixelCount; i++) {
      img[i * 4 + 0] =
          static_cast<unsigned char>(data.segmentation[i].x * 255.0f);
      img[i * 4 + 1] =
          static_cast<unsigned char>(data.segmentation[i].y * 255.0f);
      img[i * 4 + 2] =
          static_cast<unsigned char>(data.segmentation[i].z * 255.0f);
      img[i * 4 + 3] = 255;
    }
    stbi_write_png(
        (localConfig.outputPath + "/" + localConfig.segmentationFilename)
            .c_str(),
        data.width, data.height, 4, img.data(), data.width * 4);
  }
}

// Método para actualizar la configuración en tiempo real
void WabiSabiRenderer::UpdateConfig(
    const WabiSabiRenderer::RenderConfig &newConfig) {
  std::unique_lock<std::shared_mutex> lock(configMutex);

  bool resolutionChanged =
      (config.width != newConfig.width || config.height != newConfig.height);

  if (resolutionChanged) {
    // CRÍTICO: Esperar a que la GPU termine CUALQUIER tarea antes de liberar
    cudaDeviceSynchronize();

    std::cout << "[CONFIG] Resolución cambiada. Recreando buffers GPU..."
              << std::endl;

    // Liberar de forma segura
    if (d_depthMap)
      cudaFree(d_depthMap);
    if (d_normalMap)
      cudaFree(d_normalMap);
    if (d_linesMap)
      cudaFree(d_linesMap);
    if (d_segmentationMap)
      cudaFree(d_segmentationMap);
    if (d_elementIdPixelMap)
      cudaFree(d_elementIdPixelMap);
    if (d_zBuffer)
      cudaFree(d_zBuffer);
    if (d_zBuffer)
      cudaFree(d_zBuffer); // <--- Free

    // Actualizar valores después de la sincronización
    config = newConfig;
    size_t pixelCount = (size_t)config.width * config.height;

    // Re-asignar
    CUDA_CHECK(cudaMalloc(&d_depthMap, pixelCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_normalMap, pixelCount * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_linesMap, pixelCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_segmentationMap, pixelCount * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_elementIdPixelMap, pixelCount * sizeof(int)));
    CUDA_CHECK(
        cudaMalloc(&d_zBuffer,
                   pixelCount * sizeof(unsigned long long))); // <--- Allocation

    // Re-inicializar MMF
    if (config.enableMMF) {
      mmfWriter = std::make_unique<ImageMMFWriter>();
      mmfWriter->Initialize(config.width, config.height);
    }
  } else {
    config = newConfig;
  }
}

WabiSabiRenderer::~WabiSabiRenderer() {
  Stop();

  // Limpiar buffers CUDA
  if (d_vertices)
    cudaFree(d_vertices);
  if (d_triangles)
    cudaFree(d_triangles);
  if (d_normals)
    cudaFree(d_normals);
  if (d_elementIds)
    cudaFree(d_elementIds);
  if (d_categoryIds)
    cudaFree(d_categoryIds);
  if (d_categoryColors)
    cudaFree(d_categoryColors);
  if (d_depthMap)
    cudaFree(d_depthMap);
  if (d_normalMap)
    cudaFree(d_normalMap);
  if (d_linesMap)
    cudaFree(d_linesMap);
  if (d_segmentationMap)
    cudaFree(d_segmentationMap);
  if (d_zBuffer)
    cudaFree(d_zBuffer); // <--- Free

  // Cerrar MMF
  if (geometryMMF) {
    UnmapViewOfFile(geometryMMF);
    geometryMMF = nullptr;
  }
}

void WabiSabiRenderer::Stop() {
  if (!isRunning)
    return;

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
  if (!config.enableWebViewer)
    return;

  std::cout << "[WebSocket] Iniciando cliente..." << std::endl;
  ws_client.init_asio();
  ws_client.set_access_channels(
      websocketpp::log::alevel::none); // Desactivar logs

  ws_client.set_open_handler([this](websocketpp::connection_hdl hdl) {
    std::lock_guard<std::mutex> lock(ws_mutex);
    ws_connection_hdl = hdl;
    wsConnected = true;
    std::cout << "[WebSocket] Conexión establecida." << std::endl;

    // Identificarse como el renderer
    Json::Value identify_msg;
    identify_msg["type"] = "identify";
    identify_msg["client"] = "renderer";
    ws_client.send(hdl, identify_msg.toStyledString(),
                   websocketpp::frame::opcode::text);
  });

  ws_client.set_close_handler([this](websocketpp::connection_hdl hdl) {
    std::lock_guard<std::mutex> lock(ws_mutex);
    wsConnected = false;
    ws_connection_hdl.reset();
    std::cout << "[WebSocket] Conexión cerrada." << std::endl;
  });

  ws_client.set_fail_handler([this](websocketpp::connection_hdl hdl) {
    std::lock_guard<std::mutex> lock(ws_mutex);
    wsConnected = false;
    ws_connection_hdl.reset();
    std::cerr << "[WebSocket] Falló la conexión." << std::endl;
  });

  ws_thread = std::thread([this]() {
    while (isRunning) {
      try {
        if (!wsConnected) {
          std::cout << "[WebSocket] Intentando conectar a ws://localhost:"
                    << config.webSocketPort << std::endl;
          websocketpp::lib::error_code ec;
          client::connection_ptr con = ws_client.get_connection(
              "ws://localhost:" + std::to_string(config.webSocketPort), ec);
          if (ec) {
            std::cerr << "[WebSocket] Error al crear conexión: " << ec.message()
                      << std::endl;
          } else {
            ws_client.connect(con);
            ws_client.run();
            ws_client.reset(); // Reset para poder reintentar
          }
        }
      } catch (const std::exception &e) {
        std::cerr << "[WebSocket] Excepción: " << e.what() << std::endl;
      }
      // Esperar antes de reintentar
      std::this_thread::sleep_for(std::chrono::seconds(5));
    }
  });
}

// Actualizado: acepta config como parámetro
void WabiSabiRenderer::SendMapsToWebSocket(const RenderConfig &config) {
  std::lock_guard<std::mutex> lock(ws_mutex);
  if (!wsConnected || ws_connection_hdl.expired())
    return;

  size_t pixelCount = config.width * config.height;
  std::vector<unsigned char> png_buffer;

  // Función lambda para enviar un mapa
  auto send_map = [&](const std::string &mapType, const unsigned char *data,
                      int channels) {
    int len;
    unsigned char *png_data =
        stbi_write_png_to_mem(data, config.width * channels, config.width,
                              config.height, channels, &len);
    if (png_data) {
      std::string base64_data = base64_encode(png_data, len);

      Json::Value msg;
      msg["type"] = "texture_update";
      msg["mapType"] = mapType;
      msg["imageData"] = base64_data;

      ws_client.send(ws_connection_hdl, msg.toStyledString(),
                     websocketpp::frame::opcode::text);
      free(png_data);
    }
  };

  // Preparar y enviar cada mapa
  std::vector<float> h_buffer(pixelCount * 4);
  std::vector<unsigned char> imageBuffer(pixelCount * 4);

  stbi_flip_vertically_on_write(
      1); // Importante para que coincida con el frontend

  if (config.enableDepth) {
    CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_depthMap,
                          pixelCount * sizeof(float), cudaMemcpyDeviceToHost));

    float depthGamma =
        config.gamma; // Asegúrate de que coincida con el de arriba

    for (size_t i = 0; i < pixelCount; i++) {
      float distance = h_buffer[i];
      unsigned char val = 0;

      if (distance < config.maxDepth * 2.0f) {
        float linearDepth =
            (distance - config.minDepth) / (config.maxDepth - config.minDepth);
        linearDepth = fmaxf(0.0f, fminf(1.0f, linearDepth));

        // Aplicar Gamma
        float nonLinearDepth = powf(linearDepth, depthGamma);

        // Invertir
        val = static_cast<unsigned char>((1.0f - nonLinearDepth) * 255.0f);
      }
      imageBuffer[i] = val;
    }
    send_map("depth", imageBuffer.data(), 1);
  }
  if (config.enableNormals) {
    CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_normalMap,
                          pixelCount * sizeof(float3), cudaMemcpyDeviceToHost));
    convertFloat3ToRGB(reinterpret_cast<const float3 *>(h_buffer.data()),
                       imageBuffer.data(), pixelCount);
    send_map("normal", imageBuffer.data(), 3);
  }
  if (config.enableLines) {
    CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_linesMap,
                          pixelCount * sizeof(float), cudaMemcpyDeviceToHost));
    convertFloatToUchar(h_buffer.data(), imageBuffer.data(), pixelCount, false);
    send_map("lines", imageBuffer.data(), 1);
  }
  if (config.enableSegmentation) {
    CUDA_CHECK(cudaMemcpy(h_buffer.data(), d_segmentationMap,
                          pixelCount * sizeof(float4), cudaMemcpyDeviceToHost));
    convertFloat4ToRGBA(reinterpret_cast<const float4 *>(h_buffer.data()),
                        imageBuffer.data(), pixelCount);
    send_map("segmentation", imageBuffer.data(), 4);
  }
}

WabiSabiRenderer::CameraData WabiSabiRenderer::GetLatestCamera() {
  std::lock_guard<std::mutex> lock(cameraMutex);
  return currentCamera;
}

void WabiSabiRenderer::UpdateCamera(
    const WabiSabiRenderer::CameraData &camera) {
  {
    std::lock_guard<std::mutex> lock(cameraMutex);
    currentCamera = camera;

    // Si es la primera vez que se inicializa la cámara, notifícalo.
    if (!cameraInitialized) {
      cameraInitialized = true;
      std::cout << "[RENDERER] ¡Primera cámara recibida! Desbloqueando el "
                   "bucle de renderizado."
                << std::endl;
    }
  }
  cameraCv.notify_one(); // Notifica al hilo de renderizado que la cámara está
                         // lista.
}

WabiSabiRenderer::Statistics WabiSabiRenderer::GetStatistics() const {
  Statistics stats;
  stats.fps = currentFPS.load();
  stats.avgFrameTime = avgFrameTime.load();
  stats.totalFrames = totalFrames.load();
  return stats;
}