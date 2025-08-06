// ======================================================================================
// ARCHIVO: CudaHelpers.h - VERSIÓN FINAL-FINAL-FINAL-CORREGIDA
// CONTIENE TODAS LAS FUNCIONES HELPER DECLARADAS CORRECTAMENTE CON 'inline'
// ======================================================================================

#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <string>

// Macro para verificar errores CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                     << " - " << cudaGetErrorString(error) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Macro para verificar el último error
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())

// Helper para obtener información del dispositivo
inline void printCudaDeviceInfo(int device = 0) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    
    std::cout << "=== CUDA Device Info ===" << std::endl;
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "========================" << std::endl;
}

// Helper para calcular grid size óptimo
inline dim3 calculateGridSize(int width, int height, dim3 blockSize) {
    return dim3(
        (width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );
}

// Helper para medir tiempo de kernel
class CudaTimer {
private:
    cudaEvent_t start, stop;
    std::string name;
    
public:
    CudaTimer(const std::string& name = "Kernel") : name(name) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
    }
    
    ~CudaTimer() {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
    
    void Start() {
        CUDA_CHECK(cudaEventRecord(start));
    }
    
    float Stop() {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        
        std::cout << name << " took " << milliseconds << " ms" << std::endl;
        return milliseconds;
    }
};


// --------------------------------------------------------------------------------------
// HELPERS DE CONVERSIÓN CORREGIDOS CON 'inline'
// --------------------------------------------------------------------------------------

// Helper para convertir imagen float a uchar para PNG
inline void convertFloatToUchar(const float* input, unsigned char* output, int size, bool normalize = true) {
    for (int i = 0; i < size; i++) {
        float value = input[i];
        if (normalize) {
            value = fmaxf(0.0f, fminf(1.0f, value));
        }
        output[i] = static_cast<unsigned char>(value * 255.0f);
    }
}

// Helper para convertir float3 a RGB
inline void convertFloat3ToRGB(const float3* input, unsigned char* output, int pixelCount) {
    for (int i = 0; i < pixelCount; i++) {
        // Mapear de [-1, 1] a [0, 255]
        output[i * 3 + 0] = static_cast<unsigned char>((fmaxf(-1.0f, fminf(1.0f, input[i].x)) * 0.5f + 0.5f) * 255.0f);
        output[i * 3 + 1] = static_cast<unsigned char>((fmaxf(-1.0f, fminf(1.0f, input[i].y)) * 0.5f + 0.5f) * 255.0f);
        output[i * 3 + 2] = static_cast<unsigned char>((fmaxf(-1.0f, fminf(1.0f, input[i].z)) * 0.5f + 0.5f) * 255.0f);
    }
}

// Helper para convertir float4 a RGBA
inline void convertFloat4ToRGBA(const float4* input, unsigned char* output, int pixelCount) {
    for (int i = 0; i < pixelCount; i++) {
        output[i * 4 + 0] = static_cast<unsigned char>(fmaxf(0.0f, fminf(1.0f, input[i].x)) * 255.0f);
        output[i * 4 + 1] = static_cast<unsigned char>(fmaxf(0.0f, fminf(1.0f, input[i].y)) * 255.0f);
        output[i * 4 + 2] = static_cast<unsigned char>(fmaxf(0.0f, fminf(1.0f, input[i].z)) * 255.0f);
        output[i * 4 + 3] = static_cast<unsigned char>(fmaxf(0.0f, fminf(1.0f, input[i].w)) * 255.0f);
    }
}