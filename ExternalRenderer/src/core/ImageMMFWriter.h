#pragma once

#include <string>
#include <vector>
#include <windows.h>
#include "SharedMemory.h"

class ImageMMFWriter {
public:
    ImageMMFWriter();
    ~ImageMMFWriter();

    bool Initialize(int width, int height);
    void WriteFrame(int64_t seqNum, 
                    const std::vector<float>& depthData, 
                    const std::vector<float>& normalData, // float3 flattened
                    const std::vector<float>& linesData, 
                    const std::vector<float>& segData);     // float4 flattened

private:
    HANDLE hMapFile;
    void* pBuf;
    ImageStreamHeader* header;
    
    int width;
    int height;
    size_t pixelCount;
    
    // Punteros a las secciones de datos dentro del MMF
    float* pDepth;
    float* pNormal;
    float* pLines;
    float* pSeg;
};
