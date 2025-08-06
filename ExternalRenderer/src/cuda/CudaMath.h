// ExternalRenderer/src/cuda/CudaMath.h

#pragma once
#include <cuda_runtime.h>
#include <cmath>

// --- INICIO DE LA SOLUCIÓN QUIRÚRGICA ---
// Marcamos los operadores como 'static inline' para resolver los errores del enlazador LNK2005.
// Esto asegura que cada archivo fuente obtenga su propia copia privada de la función.

static inline float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static inline float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static inline float3 operator*(float s, const float3& v) {
    return make_float3(s * v.x, s * v.y, s * v.z);
}
// --- FIN DE LA SOLUCIÓN QUIRÚRGICA ---


// Las funciones helper específicas pueden permanecer dentro del namespace.
namespace CudaMath {
    inline float3 normalize(const float3& v) {
        float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
        if (len > 0) {
            float invLen = 1.0f / len;
            return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
        }
        return v;
    }

    inline float3 cross(const float3& a, const float3& b) {
        return make_float3(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }
}