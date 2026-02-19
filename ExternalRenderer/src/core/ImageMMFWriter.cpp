#include "ImageMMFWriter.h"
#include <chrono>
#include <cstring>
#include <iostream>


ImageMMFWriter::ImageMMFWriter()
    : hMapFile(NULL), pBuf(nullptr), header(nullptr), width(0), height(0) {}

ImageMMFWriter::~ImageMMFWriter() {
  if (pBuf)
    UnmapViewOfFile(pBuf);
  if (hMapFile)
    CloseHandle(hMapFile);
}

bool ImageMMFWriter::Initialize(int w, int h) {
  width = w;
  height = h;
  pixelCount = (size_t)width * height;

  // Calcular tamaños necesarios
  size_t depthSize = pixelCount * sizeof(float);
  size_t normalSize = pixelCount * 3 * sizeof(float);
  size_t linesSize = pixelCount * sizeof(float);
  size_t segSize = pixelCount * 4 * sizeof(float); // RGBA

  size_t totalDataSize = depthSize + normalSize + linesSize + segSize;
  size_t headerSize = sizeof(ImageStreamHeader);
  size_t requiredSize = headerSize + totalDataSize;

  if (requiredSize > IMAGE_MMF_SIZE) {
    std::cerr
        << "[MMF Writer] Error: Resolución demasiado alta para el buffer de "
        << IMAGE_MMF_SIZE / (1024 * 1024) << "MB" << std::endl;
    return false;
  }

  // Crear MMF
  hMapFile = CreateFileMappingA(
      INVALID_HANDLE_VALUE,  // Use paging file
      NULL,                  // Default security
      PAGE_READWRITE,        // Read/write access
      0,                     // Max. object size (high-order DWORD)
      (DWORD)IMAGE_MMF_SIZE, // Max. object size (low-order DWORD)
      IMAGE_MMF_NAME);       // Name of mapping object

  if (hMapFile == NULL) {
    std::cerr << "[MMF Writer] Could not create file mapping object ("
              << GetLastError() << ")." << std::endl;
    return false;
  }

  pBuf = MapViewOfFile(hMapFile, FILE_MAP_ALL_ACCESS, 0, 0, IMAGE_MMF_SIZE);

  if (pBuf == NULL) {
    std::cerr << "[MMF Writer] Could not map view of file (" << GetLastError()
              << ")." << std::endl;
    CloseHandle(hMapFile);
    return false;
  }

  // Inicializar Header
  header = static_cast<ImageStreamHeader *>(pBuf);
  header->width = width;
  header->height = height;

  // Calcular offsets
  header->depthOffset = headerSize;
  header->normalOffset = header->depthOffset + depthSize;
  header->linesOffset = header->normalOffset + normalSize;
  header->segmentationOffset = header->linesOffset + linesSize;

  header->depthSize = (int32_t)depthSize;
  header->normalSize = (int32_t)normalSize;
  header->linesSize = (int32_t)linesSize;
  header->segmentationSize = (int32_t)segSize;

  // Inicializar punteros para escritura rápida
  char *base = static_cast<char *>(pBuf);
  pDepth = reinterpret_cast<float *>(base + header->depthOffset);
  pNormal = reinterpret_cast<float *>(base + header->normalOffset);
  pLines = reinterpret_cast<float *>(base + header->linesOffset);
  pSeg = reinterpret_cast<float *>(base + header->segmentationOffset);

  std::cout << "[MMF Writer] Initialized. Size: " << requiredSize / 1024
            << " KB" << std::endl;
  return true;
}

void ImageMMFWriter::WriteFrame(int64_t seqNum,
                                const std::vector<float> &depthData,
                                const std::vector<float> &normalData,
                                const std::vector<float> &linesData,
                                const std::vector<float> &segData) {
  if (!pBuf)
    return;

  // Copiar datos (Nota: Se asume que los vectores tienen el tamaño correcto)
  // Para mayor velocidad en C++, usamos memcpy directo.

  if (!depthData.empty())
    memcpy(pDepth, depthData.data(), header->depthSize);
  if (!normalData.empty())
    memcpy(pNormal, normalData.data(), header->normalSize);
  if (!linesData.empty())
    memcpy(pLines, linesData.data(), header->linesSize);
  if (!segData.empty())
    memcpy(pSeg, segData.data(), header->segmentationSize);

  // Actualizar timestamp y secuencia AL FINAL para atomicidad (o casi)
  header->sequenceNumber = seqNum;
  header->timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

  // FORZAR sincronización con otros procesos
  // Optimización: Flush solo del rango utilizado en lugar de todo el buffer
  // (256MB)
  size_t headerSize = sizeof(ImageStreamHeader);
  size_t totalSize = headerSize + header->depthSize + header->normalSize +
                     header->linesSize + header->segmentationSize;

  FlushViewOfFile(pBuf, totalSize);
}
