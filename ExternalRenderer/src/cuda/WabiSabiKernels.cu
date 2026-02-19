#include "WabiSabiKernels.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <float.h>
#include <math.h>

// ======================================================================================
// OPERADORES MATEMÁTICOS PARA float3
// ======================================================================================

__device__ inline float3 operator-(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 operator+(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ inline float3 operator*(float s, const float3 &v) {
  return make_float3(s * v.x, s * v.y, s * v.z);
}

__device__ inline float3 operator*(const float3 &v, float s) {
  return make_float3(v.x * s, v.y * s, v.z * s);
}

__device__ inline float3 operator/(const float3 &v, float s) {
  float inv = 1.0f / s;
  return v * inv;
}

__device__ inline float dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 cross(const float3 &a, const float3 &b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ inline float3 normalize(const float3 &v) {
  float len = sqrtf(dot(v, v));
  if (len > 0.000001f) {
    return v / len;
  }
  return v;
}

// ======================================================================================
// HELPER: ATOMIC MIN FOR NON-NATIVE TYPES
// ======================================================================================

__device__ unsigned long long atomicMinULL(unsigned long long *addr,
                                           unsigned long long val) {
  return atomicMin(addr, val);
}

// ======================================================================================
// HELPER: PROJECTION (World -> Screen UV)
// Resuelve: Eye + t*(P-Eye) = LL + u*H + v*V
// ======================================================================================
__device__ bool ProjectToScreen(float3 P, CameraData cam, float &u, float &v,
                                float &t) {
  float3 eye = cam.eyePosition;
  float3 D = P - eye; // Ray direction (unnormalized)
  float3 H = cam.horizontal_vec;
  float3 V = cam.vertical_vec;
  float3 R = cam.lower_left_corner - eye;

  // Sistema: t*D - u*H - v*V = R
  // Matrix M = [D, -H, -V]

  // Cramer's Rule
  // Det(M) = D . ( (-H) x (-V) ) = D . (H x V)
  float3 HxV = cross(H, V);
  float detM = dot(D, HxV);

  const float EPSILON = 1e-6f;
  if (fabsf(detM) < EPSILON)
    return false;

  float invDet = 1.0f / detM;

  // Solve t: Det([R, -H, -V]) = R . (H x V)
  t = dot(R, HxV) * invDet;
  if (t < EPSILON)
    return false; // Detrás de la cámara

  // Solve u: Det([D, R, -V]) = D . (R x (-V)) = D . (V x R)
  u = dot(D, cross(V, R)) * invDet;

  // Solve v: Det([D, -H, R]) = D . ((-H) x R) = D . (R x H)
  v = dot(D, cross(R, H)) * invDet;

  return true;
}

// ======================================================================================
// KERNEL 1: RASTERIZE (Triangle -> Z-Buffer)
// ======================================================================================
__global__ void RasterizeTrianglesKernel(const float3 *vertices,
                                         int vertexCount, const int *triangles,
                                         int triangleCount,
                                         const CameraData camera, int width,
                                         int height,
                                         unsigned long long *zBuffer) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= triangleCount)
    return;

  int i0 = triangles[idx * 3];
  int i1 = triangles[idx * 3 + 1];
  int i2 = triangles[idx * 3 + 2];

  if (i0 >= vertexCount || i1 >= vertexCount || i2 >= vertexCount)
    return;

  float3 v0 = vertices[i0];
  float3 v1 = vertices[i1];
  float3 v2 = vertices[i2];

  float u0, v0_uv, t0;
  float u1, v1_uv, t1;
  float u2, v2_uv, t2;

  if (!ProjectToScreen(v0, camera, u0, v0_uv, t0))
    return;
  if (!ProjectToScreen(v1, camera, u1, v1_uv, t1))
    return;
  if (!ProjectToScreen(v2, camera, u2, v2_uv, t2))
    return;

  float x0 = u0 * (width - 1);
  float y0 = v0_uv * (height - 1);
  float x1 = u1 * (width - 1);
  float y1 = v1_uv * (height - 1);
  float x2 = u2 * (width - 1);
  float y2 = v2_uv * (height - 1);

  int minX = max(0, (int)floorf(min(x0, min(x1, x2))));
  int maxX = min(width - 1, (int)ceilf(max(x0, max(x1, x2))));
  int minY = max(0, (int)floorf(min(y0, min(y1, y2))));
  int maxY = min(height - 1, (int)ceilf(max(y0, max(y1, y2))));

  if (maxX < minX || maxY < minY)
    return;

  float area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
  // Backface Culling simple: si area <= 0, es backfacing (asumiendo CCW
  // winding) if (area <= 0) return; // Uncomment to enable backface culling
  if (fabsf(area) < 1e-4f)
    return;
  float invArea = 1.0f / area;

  for (int py = minY; py <= maxY; py++) {
    for (int px = minX; px <= maxX; px++) {
      float fx = (float)px;
      float fy = (float)py;

      float w0 = ((x1 - fx) * (y2 - fy) - (y1 - fy) * (x2 - fx)) * invArea;
      float w1 = ((x2 - fx) * (y0 - fy) - (y2 - fy) * (x0 - fx)) * invArea;
      float w2 = 1.0f - w0 - w1;

      if (w0 >= 0 && w1 >= 0 && w2 >= 0) {
        // Correct Perspective Interpolation:
        // Interpolate 't' (which is ~ 1/Depth) linearly in screen space.
        float t = w0 * t0 + w1 * t1 + w2 * t2;

        // Use atomicMax because larger t means closer directly proportional to
        // 1/Z.
        unsigned int tInt = __float_as_int(t);
        unsigned long long packed =
            ((unsigned long long)tInt << 32) | (unsigned long long)idx;

        // atomicMax for unsigned long long
        // We assume t > 0.
        atomicMax((unsigned long long *)&zBuffer[py * width + px], packed);
      }
    }
  }
}

// ======================================================================================
// KERNEL 2: RESOLVE (Z-Buffer -> Maps)
// ======================================================================================
__global__ void ResolveKernel(const float3 *vertices, const int *triangles,
                              const float3 *normals, const int *elementIds,
                              const int *categoryIds,
                              const float3 *categoryColors, int width,
                              int height,
                              const CameraData camera, // <--- Added Camera
                              const unsigned long long *zBuffer,
                              float *depthMap, float3 *normalMap,
                              int *idPixelMap, float4 *segmentationMap) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int idx = y * width + x;

  if (x >= width || y >= height)
    return;

  unsigned long long packed = zBuffer[idx];

  // Background (Initialized to 0)
  if (packed == 0) {
    depthMap[idx] = FLT_MAX;
    normalMap[idx] = make_float3(0.5f, 0.5f, 1.0f);
    idPixelMap[idx] = -1;
    segmentationMap[idx] = make_float4(0.1f, 0.1f, 0.1f, 1.0f);
    return;
  }

  int triIdx = (int)(packed & 0xFFFFFFFF);
  float t = __int_as_float((int)(packed >> 32));

  // Calculate Real Depth
  // t = RayLength_Plane / RayLength_Vertex
  // Depth = RayLength_Plane / t;
  float u = (float)x / (width - 1);
  float v = (float)y / (height - 1);
  float3 planePos = camera.lower_left_corner + (u * camera.horizontal_vec) +
                    (v * camera.vertical_vec);
  float rayLenPlane =
      sqrtf(dot(planePos - camera.eyePosition, planePos - camera.eyePosition));

  // Evitar div by zero
  if (t < 1e-6f)
    t = 1e-6f;
  float realDepth = rayLenPlane / t;

  depthMap[idx] = realDepth;

  // Flat Shading Attributes
  int i0 = triangles[triIdx * 3];
  normalMap[idx] = normalize(normals[i0]);
  idPixelMap[idx] = elementIds[i0];

  int catId = categoryIds[i0];
  if (catId >= 0 && catId < 256) {
    float3 c = categoryColors[catId];
    segmentationMap[idx] = make_float4(c.x, c.y, c.z, 1.0f);
  } else {
    segmentationMap[idx] = make_float4(0.5f, 0.5f, 0.5f, 1.0f);
  }
}

// ======================================================================================
// KERNEL: CombinedEdgeDetection (Post-procesado de líneas)
// ======================================================================================

__global__ void CombinedEdgeDetectionKernel(const float3 *normalMap,
                                            const float *depthMap,
                                            const int *idMap, float *linesMap,
                                            int w, int h, float normalThresh,
                                            float depthThresh) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < 1 || x >= w - 1 || y < 1 || y >= h - 1)
    return;

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
  bool isNormalEdge =
      ((1.0f - dotL) > normalThresh || (1.0f - dotU) > normalThresh);

  // RESULTADO: Si cumple CUALQUIERA de las tres condiciones, dibujamos negro
  // (0.0)
  if (isIdEdge || isDepthEdge || isNormalEdge) {
    linesMap[idxC] = 0.0f;
  } else {
    linesMap[idxC] = 1.0f;
  }
}

// ======================================================================================
// WRAPPERS (Llamados desde C++)
// ======================================================================================

void LaunchAllMapsKernel(const float3 *d_vertices, int vertexCount,
                         const int *d_triangles, const float3 *d_normals,
                         const int *d_elementIds, const int *d_categoryIds,
                         const float3 *d_categoryColors, int triangleCount,
                         int categoryCount, const CameraData &camera,
                         const RenderConfig &config, float *d_depth,
                         float3 *d_normal, int *d_idPixel, float4 *d_segment,
                         unsigned long long *d_zBuffer, cudaStream_t stream) {
  // PASO 0: INIT Z-BUFFER (0 = Farthest for atomicMax logic)
  cudaMemsetAsync(d_zBuffer, 0,
                  config.width * config.height * sizeof(unsigned long long),
                  stream);

  // PASO 1: RASTERIZE
  int threadsPerBlock = 256;
  int blocks = (triangleCount + threadsPerBlock - 1) / threadsPerBlock;
  if (blocks == 0)
    blocks = 1;

  RasterizeTrianglesKernel<<<blocks, threadsPerBlock, 0, stream>>>(
      d_vertices, vertexCount, d_triangles, triangleCount, camera, config.width,
      config.height, d_zBuffer);

  // PASO 2: RESOLVE
  dim3 blockSize(16, 16);
  dim3 gridSize((config.width + 15) / 16, (config.height + 15) / 16);

  ResolveKernel<<<gridSize, blockSize, 0, stream>>>(
      d_vertices, d_triangles, d_normals, d_elementIds, d_categoryIds,
      d_categoryColors, config.width, config.height, camera, d_zBuffer, d_depth,
      d_normal, d_idPixel, d_segment);
}

void LaunchLinesKernel(const float3 *d_normalMap, const float *d_depthMap,
                       const int *d_idPixelMap, const RenderConfig &config,
                       float *d_linesMap, cudaStream_t stream) {
  dim3 blockSize(16, 16);
  dim3 gridSize((config.width + 15) / 16, (config.height + 15) / 16);

  CombinedEdgeDetectionKernel<<<gridSize, blockSize, 0, stream>>>(
      d_normalMap, d_depthMap, d_idPixelMap, d_linesMap, config.width,
      config.height, config.normalThreshold, config.depthThreshold);
}