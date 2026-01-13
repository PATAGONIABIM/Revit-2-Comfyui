#include "WabiSabiKernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>

// ======================================================================================
// OPERADORES MATEMÁTICOS PARA float3 (CORREGIDOS PARA EVITAR ERRORES DE COMPILACIÓN)
// ======================================================================================

__device__ inline float3 operator-(const float3& a, const float3& b) { 
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); 
}

__device__ inline float3 operator+(const float3& a, const float3& b) { 
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); 
}

// Escalar * Vector
__device__ inline float3 operator*(float s, const float3& v) { 
    return make_float3(s * v.x, s * v.y, s * v.z); 
}

// Vector * Escalar (Esto resuelve tu error de compilación previo)
__device__ inline float3 operator*(const float3& v, float s) { 
    return make_float3(v.x * s, v.y * s, v.z * s); 
}

// División de Vector por Escalar
__device__ inline float3 operator/(const float3& v, float s) { 
    float inv = 1.0f / s;
    return v * inv; 
}

__device__ inline float dot(const float3& a, const float3& b) { 
    return a.x * b.x + a.y * b.y + a.z * b.z; 
}

__device__ inline float3 cross(const float3& a, const float3& b) { 
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    ); 
}

__device__ inline float3 normalize(const float3& v) { 
    float len = sqrtf(dot(v, v));
    if (len > 0.000001f) {
        return v / len;
    }
    return v;
}

// ======================================================================================
// INTERSECCIÓN RAYO-TRIÁNGULO (Möller-Trumbore)
// ======================================================================================

__device__ float rayTriangleIntersect(
    const float3& orig, const float3& dir, 
    const float3& v0, const float3& v1, const float3& v2) 
{
    const float EPSILON = 0.0000001f;
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(dir, edge2);
    float a = dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON) return -1.0f;
    
    float f = 1.0f / a;
    float3 s = orig - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f) return -1.0f;
    
    float3 q = cross(s, edge1);
    float v = f * dot(dir, q);
    
    if (v < 0.0f || u + v > 1.0f) return -1.0f;
    
    float t = f * dot(edge2, q);
    return (t > EPSILON) ? t : -1.0f;
}

// ======================================================================================
// MEGA-KERNEL: RenderAllMapsCombined
// Procesa Depth, Normals, IDs y Segmentation en una sola pasada de triángulos.
// ======================================================================================

__global__ void RenderAllMapsCombinedKernel(
    const float3* vertices, int vertexCount, const int* triangles, 
    const float3* normals, const int* elementIds, const int* categoryIds,
    const float3* categoryColors, int triangleCount, int categoryCount,
    const CameraData camera, const RenderConfig config,
    float* depthMap, float3* normalMap, int* idPixelMap, float4* segmentationMap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= config.width || y >= config.height) return;

    // Generación del rayo
    float u = (float)x / (config.width - 1);
    float v = (float)y / (config.height - 1);
    float3 rayOrigin = camera.eyePosition;
    float3 target = camera.lower_left_corner + (u * camera.horizontal_vec) + (v * camera.vertical_vec);
    float3 rayDir = normalize(target - rayOrigin);

    float closestT = FLT_MAX;
    int hitTriangleIdx = -1;

    // --- BUCLE ÚNICO DE TRIÁNGULOS ---
    for (int i = 0; i < triangleCount; i++) {
        int i0 = triangles[i * 3];
        int i1 = triangles[i * 3 + 1];
        int i2 = triangles[i * 3 + 2];
        
        // Validación de seguridad para memoria CUDA
        if (i0 >= vertexCount || i0 < 0) continue;

        float t = rayTriangleIntersect(rayOrigin, rayDir, vertices[i0], vertices[i1], vertices[i2]);
        if (t > 0.0f && t < closestT) {
            closestT = t;
            hitTriangleIdx = i;
        }
    }

    int pixelIdx = y * config.width + x;

    if (hitTriangleIdx >= 0) {
        int firstVertexIdx = triangles[hitTriangleIdx * 3];
        
        // 1. Mapa de Profundidad
        depthMap[pixelIdx] = closestT;
        
        // 2. Mapa de Normales
        normalMap[pixelIdx] = normalize(normals[firstVertexIdx]);
        
        // 3. Mapa de IDs (para las líneas de Revit)
        idPixelMap[pixelIdx] = elementIds[firstVertexIdx];
        
        // 4. Mapa de Segmentación
        int catId = categoryIds[firstVertexIdx];
        if (catId >= 0 && catId < 256) {
            float3 c = categoryColors[catId];
            segmentationMap[pixelIdx] = make_float4(c.x, c.y, c.z, 1.0f);
        } else {
            segmentationMap[pixelIdx] = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
        }
    } else {
        // Valores por defecto (Fondo)
        depthMap[pixelIdx] = FLT_MAX;
        normalMap[pixelIdx] = make_float3(0.5f, 0.5f, 1.0f);
        idPixelMap[pixelIdx] = -1;
        segmentationMap[pixelIdx] = make_float4(0.1f, 0.1f, 0.1f, 1.0f);
    }
}

// ======================================================================================
// KERNEL: CombinedEdgeDetection (Post-procesado de líneas)
// ======================================================================================

__global__ void CombinedEdgeDetectionKernel(
    const float3* normalMap, 
    const float* depthMap, 
    const int* idMap, 
    float* linesMap, 
    int w, int h, 
    float normalThresh, // <--- Este ahora sí se usa
    float depthThresh)   // <--- Este ahora sí se usa
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Margen de 1 píxel para poder comparar con vecinos
    if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1) return;

    int idxC = y * w + x;
    int idC = idMap[idxC];
    float dC = depthMap[idxC];
    float3 nC = normalMap[idxC];

    // Si es fondo o está demasiado lejos, pintar blanco
    if (idC == -1 || dC > 10000.0f) {
        linesMap[idxC] = 1.0f;
        return;
    }

    // Índices de vecinos (Izquierda y Arriba)
    int idxL = y * w + (x - 1);
    int idxU = (y + 1) * w + x;

    // --- 1. DETECCIÓN POR ID (Bordes de familia/objeto) ---
    bool isIdEdge = (idC != idMap[idxL] || idC != idMap[idxU]);

    // --- 2. DETECCIÓN POR PROFUNDIDAD (Bordes de silueta) ---
    // Calculamos la diferencia relativa de profundidad
    float diffDL = fabsf(dC - depthMap[idxL]) / dC;
    float diffDU = fabsf(dC - depthMap[idxU]) / dC;
    bool isDepthEdge = (diffDL > depthThresh || diffDU > depthThresh);

    // --- 3. DETECCIÓN POR NORMALES (Aristas y pliegues internos) ---
    // Calculamos el producto punto (1.0 = misma dirección, 0.0 = perpendicular)
    float dotL = dot(nC, normalMap[idxL]);
    float dotU = dot(nC, normalMap[idxU]);
    // Si la diferencia (1.0 - dot) supera el umbral, es un borde
    bool isNormalEdge = ((1.0f - dotL) > normalThresh || (1.0f - dotU) > normalThresh);

    // RESULTADO: Si cumple CUALQUIERA de las tres condiciones, dibujamos negro (0.0)
    if (isIdEdge || isDepthEdge || isNormalEdge) {
        linesMap[idxC] = 0.0f;
    } else {
        linesMap[idxC] = 1.0f;
    }
}

// ======================================================================================
// WRAPPERS (Llamados desde C++)
// ======================================================================================

void LaunchAllMapsKernel(
    const float3* d_vertices, int vertexCount, const int* d_triangles, 
    const float3* d_normals, const int* d_elementIds, const int* d_categoryIds,
    const float3* d_categoryColors, int triangleCount, int categoryCount,
    const CameraData& camera, const RenderConfig& config,
    float* d_depth, float3* d_normal, int* d_idPixel, float4* d_segment, 
    cudaStream_t stream) 
{
    dim3 blockSize(16, 16);
    dim3 gridSize((config.width + 15) / 16, (config.height + 15) / 16);
    
    RenderAllMapsCombinedKernel<<<gridSize, blockSize, 0, stream>>>(
        d_vertices, vertexCount, d_triangles, d_normals, d_elementIds, d_categoryIds,
        d_categoryColors, triangleCount, categoryCount, camera, config, 
        d_depth, d_normal, d_idPixel, d_segment);
}

void LaunchLinesKernel(
    const float3* d_normalMap, 
    const float* d_depthMap, 
    const int* d_idPixelMap,
    const RenderConfig& config, // <--- Aquí vienen los valores del JSON
    float* d_linesMap, 
    cudaStream_t stream) 
{
    dim3 blockSize(16, 16);
    dim3 gridSize((config.width + 15) / 16, (config.height + 15) / 16);
    
    CombinedEdgeDetectionKernel<<<gridSize, blockSize, 0, stream>>>(
        d_normalMap, 
        d_depthMap, 
        d_idPixelMap, 
        d_linesMap, 
        config.width, 
        config.height, 
        config.normalThreshold, // <--- Valor del JSON pasado al kernel
        config.depthThreshold   // <--- Valor del JSON pasado al kernel
    );
}