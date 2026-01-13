#pragma once
#include <cuda_runtime.h>

// Asegúrate de que las estructuras coincidan con las del .cu
struct CameraData {
    float3 eyePosition;
    float3 lower_left_corner;
    float3 horizontal_vec;
    float3 vertical_vec;
};

struct RenderConfig {
    int width;
    int height;
    float minDepth;
    float maxDepth;
    float depthThreshold;
    float normalThreshold;
};

// LA ÚNICA FUNCIÓN PARA RENDERIZAR TODO
void LaunchAllMapsKernel(
    const float3* d_vertices, int vertexCount, const int* d_triangles, 
    const float3* d_normals, const int* d_elementIds, const int* d_categoryIds,
    const float3* d_categoryColors, int triangleCount, int categoryCount,
    const CameraData& camera, const RenderConfig& config,
    float* d_depth, float3* d_normal, int* d_idPixel, float4* d_segment, 
    cudaStream_t stream
);

// LA FUNCIÓN PARA LAS LÍNEAS
void LaunchLinesKernel(
    const float3* d_normalMap, const float* d_depthMap, const int* d_idPixelMap,
    const RenderConfig& config, float* d_linesMap, cudaStream_t stream
);