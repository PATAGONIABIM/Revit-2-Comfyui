// ExternalRenderer/src/utils/CSVReader.h
#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <cuda_runtime.h>

class CSVReader {
public:
    struct ColorEntry {
        std::string category;
        float3 color; // RGB normalizado 0-1
    };

    static std::vector<ColorEntry> ReadCategoryColors(const std::string& filename);
    static std::unordered_map<int, float3> MapCategoriesToIds(
        const std::unordered_map<std::string, float3>& colorMap,
        const std::unordered_map<int, std::string>& categoryNames
    );
    
private:
    static float3 HexToFloat3(const std::string& hex);
    static std::vector<std::string> SplitString(const std::string& str, char delimiter);
};