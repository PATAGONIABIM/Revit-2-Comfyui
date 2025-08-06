#include "WabiSabiKernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>  // Para FLT_MAX

#define MAX_CATEGORIES 256

// Funciones helper para operaciones con float3
__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ float3 normalize(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 0) {
        float invLen = 1.0f / len;
        return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
    }
    return v;
}

// Implementación del algoritmo de Möller-Trumbore para intersección rayo-triángulo
__device__ float rayTriangleIntersect(
    const float3& origin,
    const float3& direction,
    const float3& v0,
    const float3& v1,
    const float3& v2)
{
    const float EPSILON = 0.0000001f;
    float3 edge1 = v1 - v0;
    float3 edge2 = v2 - v0;
    float3 h = cross(direction, edge2);
    float a = dot(edge1, h);
    
    if (a > -EPSILON && a < EPSILON)
        return -1.0f;
    
    float f = 1.0f / a;
    float3 s = origin - v0;
    float u = f * dot(s, h);
    
    if (u < 0.0f || u > 1.0f)
        return -1.0f;
    
    float3 q = cross(s, edge1);
    float v = f * dot(direction, q);
    
    if (v < 0.0f || u + v > 1.0f)
        return -1.0f;
    
    float t = f * dot(edge2, q);
    
    if (t > EPSILON)
        return t;
    else
        return -1.0f;
}

// Kernel para renderizar mapa de profundidad
__global__ void RenderDepthMap(
    const float3* vertices, int vertexCount, const int* triangles, int triangleCount,
    const CameraData camera, const RenderConfig config, float* depthMap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= config.width || y >= config.height) return;

    float u = (float)x / (config.width - 1);
    float v = (float)y / (config.height - 1);

    float3 rayOrigin = camera.eyePosition;
    float3 rayDir = normalize(camera.lower_left_corner
                             + u * camera.horizontal_vec
                             + v * camera.vertical_vec
                             - rayOrigin);

    float closestT = FLT_MAX;

    for (int i = 0; i < triangleCount; i++) {
        int i0 = triangles[i * 3]; int i1 = triangles[i * 3 + 1]; int i2 = triangles[i * 3 + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;
        
        float3 v0 = vertices[i0]; float3 v1 = vertices[i1]; float3 v2 = vertices[i2];
        
        float t = rayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2);
        if (t > 0.0f && t < closestT) {
            closestT = t;
        }
    }

    // --- LÓGICA DE PROFUNDIDAD SIMPLIFICADA ---
    // Simplemente escribimos la distancia real. Si no hay hit, se queda como FLT_MAX.
    // La normalización ahora es responsabilidad del código C++.
    depthMap[y * config.width + x] = closestT;
}
// Kernel para renderizar mapa de normales
__global__ void RenderNormalMap(
    const float3* vertices, int vertexCount, const int* triangles, const float3* normals, int triangleCount,
    const CameraData camera, const RenderConfig config, float3* normalMap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= config.width || y >= config.height) return;
    
    float u = (float)x / (config.width - 1);
    // --- INICIO DE LA CORRECCIÓN ---
    // Se elimina la inversión de 'v' para que coincida con el kernel de profundidad.
    // La imagen se generará de abajo hacia arriba, lo cual es esperado por la
    // función de guardado que usa stbi_flip_vertically_on_write(1).
    float v = (float)y / (config.height - 1); 
    // --- FIN DE LA CORRECCIÓN ---

    float3 rayOrigin = camera.eyePosition;
    
    float3 rayDir = normalize(camera.lower_left_corner 
                             + u * camera.horizontal_vec 
                             + v * camera.vertical_vec 
                             - rayOrigin);

    float closestT = FLT_MAX;
    int hitTriangle = -1;
    

    for (int i = 0; i < triangleCount; i++) {
        int i0 = triangles[i * 3]; int i1 = triangles[i * 3 + 1]; int i2 = triangles[i * 3 + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;
        float3 v0 = vertices[i0]; float3 v1 = vertices[i1]; float3 v2 = vertices[i2];
        float t = rayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2);
        if (t > 0.0f && t < closestT) { closestT = t; hitTriangle = i; }
    }

    float3 normal = make_float3(0.5f, 0.5f, 1.0f);
    if (hitTriangle >= 0) {
        int i0 = triangles[hitTriangle * 3];
        // Aquí también validamos por si acaso, aunque ya debería estar cubierto
        if (i0 < vertexCount) {
             normal = normalize(normals[i0]);
        }
    }
    normalMap[y * config.width + x] = normal;
}

// --------------------------------------------------------------------------------------
// Kernel de Detección de Bordes por Normales
// Entrada: Mapa de Normales
// Salida:  Mapa de Líneas
// --------------------------------------------------------------------------------------
__global__ void AdvancedNormalEdgeKernel(
    const float3* normalMap,
    float* linesMap,
    int width, int height,
    float normalThreshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
        if (x < width && y < height) {
            linesMap[y * width + x] = 1.0f; // Fondo blanco
        }
        return;
    }

    // Muestreo de las normales de los 8 píxeles vecinos
    const float3 n00 = normalMap[(y - 1) * width + (x - 1)]; const float3 n01 = normalMap[(y - 1) * width + x]; const float3 n02 = normalMap[(y - 1) * width + (x + 1)];
    const float3 n10 = normalMap[y * width + (x - 1)];       const float3 n11 = normalMap[y * width + x];       const float3 n12 = normalMap[y * width + (x + 1)];
    const float3 n20 = normalMap[(y + 1) * width + (x - 1)]; const float3 n21 = normalMap[(y + 1) * width + x]; const float3 n22 = normalMap[(y + 1) * width + (x + 1)];

    // Si el píxel central es fondo, no hay línea que dibujar
    if (n11.x == 0.5f && n11.y == 0.5f && n11.z == 1.0f) {
        linesMap[y * width + x] = 1.0f;
        return;
    }

    // Aplicar Sobel para el eje X
    float3 gx = (n02 + 2.0f * n12 + n22) - (n00 + 2.0f * n10 + n20);

    // Aplicar Sobel para el eje Y
    float3 gy = (n20 + 2.0f * n21 + n22) - (n00 + 2.0f * n01 + n02);

    // --- LÍNEA CORREGIDA ---
    // Calcular la magnitud del gradiente (cambio de ángulo) usando la fórmula correcta
    float gx_mag = sqrtf(gx.x * gx.x + gx.y * gx.y + gx.z * gx.z);
    float gy_mag = sqrtf(gy.x * gy.x + gy.y * gy.y + gy.z * gy.z);
    float magnitude = gx_mag + gy_mag;

    // Si la magnitud supera el umbral, es una línea (negra)
    if (magnitude > normalThreshold) {
        linesMap[y * width + x] = 0.0f;
    } else {
        linesMap[y * width + x] = 1.0f;
    }
}

// Kernel para renderizar mapa de líneas (detección de bordes)
__global__ void SobelFilterKernel(const float* depthMap, float* linesMap, int width, int height, float threshold)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x == 0 || x >= width - 1 || y == 0 || y >= height - 1) {
        if (x < width && y < height) {
            linesMap[y * width + x] = 1.0f; // Fondo blanco en los bordes de la imagen
        }
        return;
    }

    // Muestreo de píxeles vecinos del mapa de profundidad
    float p00 = depthMap[(y-1) * width + (x-1)]; float p01 = depthMap[(y-1) * width + x]; float p02 = depthMap[(y-1) * width + (x+1)];
    float p10 = depthMap[y * width + (x-1)];                                             float p12 = depthMap[y * width + (x+1)];
    float p20 = depthMap[(y+1) * width + (x-1)]; float p21 = depthMap[(y+1) * width + x]; float p22 = depthMap[(y+1) * width + (x+1)];

    // Operadores de Sobel para detectar cambios en la profundidad
    float gx = (p02 + 2.0f * p12 + p22) - (p00 + 2.0f * p10 + p20);
    float gy = (p20 + 2.0f * p21 + p22) - (p00 + 2.0f * p01 + p02);

    float magnitude = sqrtf(gx * gx + gy * gy);

    // Si el cambio de profundidad es grande (un borde), pinta negro (0.0). Si no, pinta blanco (1.0).
    linesMap[y * width + x] = (magnitude > threshold) ? 0.0f : 1.0f;
}


// --------------------------------------------------------------------------------------
// Kernel de Líneas (SIMPLIFICADO)
// ¡Este kernel ya no existe! Su lógica se mueve a la función de lanzamiento.
// Lo mantenemos vacío por si alguna referencia antigua lo necesita, pero no se usa.
// --------------------------------------------------------------------------------------
__global__ void RenderLinesMap() { }
// Kernel para renderizar mapa de segmentación
__global__ void RenderSegmentationMap(
    const float3* vertices, int vertexCount, const int* triangles, const int* categoryIds,
    const float3* categoryColors, int triangleCount, int categoryCount,
    const CameraData camera, const RenderConfig config, float4* segmentationMap)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= config.width || y >= config.height) return;

    float u = (float)x / (config.width - 1);
    float v = (float)y / (config.height - 1);

    float3 rayOrigin = camera.eyePosition;
    float3 rayDir = normalize(camera.lower_left_corner
                             + u * camera.horizontal_vec
                             + v * camera.vertical_vec
                             - rayOrigin);

    float closestT = FLT_MAX;
    int hitCategoryId = -1;

    // Bucle sobre TODOS los triángulos.
    for (int i = 0; i < triangleCount; i++) {
        int i0 = triangles[i * 3]; int i1 = triangles[i * 3 + 1]; int i2 = triangles[i * 3 + 2];
        if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount) continue;
        
        float3 v0 = vertices[i0]; float3 v1 = vertices[i1]; float3 v2 = vertices[i2];
        
        float t = rayTriangleIntersect(rayOrigin, rayDir, v0, v1, v2);
        if (t > 0.0f && t < closestT) {
            closestT = t;
            hitCategoryId = categoryIds[i0];
        }
    }

    // Color de fondo por defecto: GRIS CLARO, OPACA.
    float4 finalColor = make_float4(0.8f, 0.8f, 0.8f, 1.0f);

    // Si el rayo golpeó un objeto y la categoría es válida...
    if (hitCategoryId != -1 && hitCategoryId >= 0 && hitCategoryId < categoryCount) {
        // ...obtenemos el color correcto.
        float3 catColor = categoryColors[hitCategoryId];
        finalColor = make_float4(catColor.x, catColor.y, catColor.z, 1.0f);
    }

    segmentationMap[y * config.width + x] = finalColor;
}


// Implementación de las funciones de lanzamiento
void LaunchDepthKernel(
    const float3* d_vertices, int vertexCount, const int* d_triangles, int triangleCount,
    const CameraData& camera, const RenderConfig& config, float* d_depthMap, cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((config.width + blockSize.x - 1) / blockSize.x, (config.height + blockSize.y - 1) / blockSize.y);
    RenderDepthMap<<<gridSize, blockSize, 0, stream>>>(d_vertices, vertexCount, d_triangles, triangleCount, camera, config, d_depthMap);
}

void LaunchNormalKernel(
    const float3* d_vertices, int vertexCount, const int* d_triangles, const float3* d_normals, int triangleCount,
    const CameraData& camera, const RenderConfig& config, float3* d_normalMap, cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((config.width + blockSize.x - 1) / blockSize.x, (config.height + blockSize.y - 1) / blockSize.y);
    RenderNormalMap<<<gridSize, blockSize, 0, stream>>>(d_vertices, vertexCount, d_triangles, d_normals, triangleCount, camera, config, d_normalMap);
}

void LaunchLinesKernel(
    const float3* d_normalMap,
    const RenderConfig& config,
    float* d_linesMap,
    cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((config.width + blockSize.x - 1) / blockSize.x, (config.height + blockSize.y - 1) / blockSize.y);

    AdvancedNormalEdgeKernel<<<gridSize, blockSize, 0, stream>>>(
        d_normalMap,
        d_linesMap,
        config.width, config.height,
        config.normalThreshold
    );
}

void LaunchSegmentationKernel(
    const float3* d_vertices, int vertexCount, const int* d_triangles, const int* d_categoryIds, // <-- AÑADIR d_categoryIds
    const float3* d_categoryColors, int triangleCount, int categoryCount, // <-- AÑADIR categoryCount
    const CameraData& camera, const RenderConfig& config, float4* d_segmentationMap, cudaStream_t stream)
{
    dim3 blockSize(16, 16);
    dim3 gridSize((config.width + blockSize.x - 1) / blockSize.x, (config.height + blockSize.y - 1) / blockSize.y);
    
    // --- INICIO DE LA CORRECCIÓN: PASAR LOS NUEVOS PARÁMETROS ---
    RenderSegmentationMap<<<gridSize, blockSize, 0, stream>>>(
        d_vertices, vertexCount, d_triangles, d_categoryIds, d_categoryColors,
        triangleCount, categoryCount, 
        camera, config, d_segmentationMap);
    // --- FIN DE LA CORRECCIÓN ---
}