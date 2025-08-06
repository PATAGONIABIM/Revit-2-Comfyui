// ExternalRenderer/src/utils/CSVReader.cpp
#include "CSVReader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>

std::vector<CSVReader::ColorEntry> CSVReader::ReadCategoryColors(const std::string& filename) {
    std::vector<ColorEntry> colorList;
    
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: No se pudo abrir " << filename << std::endl;
        return colorList;
    }
    
    std::string line;
    std::getline(file, line); // Saltar header
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        auto parts = SplitString(line, ',');
        if (parts.size() >= 2) {
            std::string category = parts[0];
            std::string colorHex = parts[1];
            
            category.erase(remove_if(category.begin(), category.end(), isspace), category.end());
            colorHex.erase(remove_if(colorHex.begin(), colorHex.end(), isspace), colorHex.end());
            
            float3 color = HexToFloat3(colorHex);
            // - Antes:
            // colorMap[category] = color;
            // + Después:
            colorList.push_back({category, color});
            
            std::cout << "Categoría: " << category << " -> Color: (" 
                     << color.x << ", " << color.y << ", " << color.z << ")" << std::endl;
        }
    }
    
    file.close();
    std::cout << "Cargadas " << colorList.size() << " categorías con colores" << std::endl;
    return colorList;
}

std::unordered_map<int, float3> CSVReader::MapCategoriesToIds(
    const std::unordered_map<std::string, float3>& colorMap,
    const std::unordered_map<int, std::string>& categoryNames) {
    
    std::unordered_map<int, float3> idColorMap;
    
    for (const auto& [id, name] : categoryNames) {
        auto it = colorMap.find(name);
        if (it != colorMap.end()) {
            idColorMap[id] = it->second;
        } else {
            // Color por defecto si no se encuentra
            idColorMap[id] = make_float3(0.5f, 0.5f, 0.5f);
            std::cout << "Advertencia: No se encontró color para categoría '" 
                     << name << "' (ID: " << id << ")" << std::endl;
        }
    }
    
    return idColorMap;
}

float3 CSVReader::HexToFloat3(const std::string& hex) {
    std::string cleanHex = hex;
    
    // Remover # si existe
    if (cleanHex[0] == '#') {
        cleanHex = cleanHex.substr(1);
    }
    
    // Convertir de hex a RGB
    if (cleanHex.length() == 6) {
        int r = std::stoi(cleanHex.substr(0, 2), nullptr, 16);
        int g = std::stoi(cleanHex.substr(2, 2), nullptr, 16);
        int b = std::stoi(cleanHex.substr(4, 2), nullptr, 16);
        
        return make_float3(r / 255.0f, g / 255.0f, b / 255.0f);
    }
    
    // Color por defecto si hay error
    return make_float3(0.5f, 0.5f, 0.5f);
}

std::vector<std::string> CSVReader::SplitString(const std::string& str, char delimiter) {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    
    return tokens;
}