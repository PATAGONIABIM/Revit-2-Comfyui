#pragma once

#include <cstdint>

// Protocolo de Cámara (Revit -> Renderer)
// Debe coincidir con el struct de C# 'SerializableCameraState'
struct SerializableCameraState {
    // Orientación
    float EyePosX, EyePosY, EyePosZ;
    float ForwardX, ForwardY, ForwardZ;
    float UpX, UpY, UpZ;
    float RightX, RightY, RightZ;

    // Proyección
    float FieldOfView;
    float AspectRatio;
    float NearClipPlane;
    float FarClipPlane;

    // Sincronización
    int64_t SequenceNumber;
};

// Protocolo de Imagen (Renderer -> ComfyUI/Otros)
struct ImageStreamHeader {
    int32_t width;
    int32_t height;
    int64_t timestamp;      // Última actualización
    int64_t sequenceNumber; // Contador incremental
    
    // Offsets a los datos (en bytes desde el inicio del MMF)
    int64_t depthOffset;
    int64_t normalOffset;
    int64_t linesOffset;
    int64_t segmentationOffset;
    
    // Tamaños de los buffers en bytes
    int32_t depthSize;
    int32_t normalSize;
    int32_t linesSize;
    int32_t segmentationSize;
};

// Nombres constantes de los MMFs
constexpr const char* CAMERA_MMF_NAME = "WabiSabiBridge_CameraStream";
constexpr const char* IMAGE_MMF_NAME = "WabiSabiBridge_ImageStream";
constexpr const int64_t IMAGE_MMF_SIZE = 1024 * 1024 * 256; // 256 MB (Suficiente para 4 renders 4K)
