// ExternalRenderer/src/utils/JournalParser.cpp
#include "JournalParser.h"
#include <Windows.h>
#include <iostream>
#include <filesystem>
#include <chrono>
#include <sstream>

std::atomic<int> JournalParser::globalSequenceNumber{0};

JournalParser::JournalParser() 
    : lastPosition(0), isWatching(false) {    
    cameraRegex = std::regex(
        R"(Jrn\.Directive[\s\S]*?"AutoCamCamera"[\s\S]*?,\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*_\s*,\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*_\s*,\s*([-.\dE]+)\s*,\s*([-.\dE]+)\s*,\s*([-.\dE]+))"
    );
}

JournalParser::~JournalParser() {
    StopWatching();
    if (journalFile.is_open()) {
        journalFile.close();
    }
}

bool JournalParser::Initialize(const std::string& revitVersion) {
    journalPath = FindLatestJournalFile(revitVersion);
    
    if (journalPath.empty()) {
        std::cerr << "No se encontró archivo journal" << std::endl;
        return false;
    }
    
    std::cout << "Monitoreando journal: " << journalPath << std::endl;
    
    // Abrir archivo y posicionarse al final
    journalFile.open(journalPath, std::ios::in);
    if (!journalFile.is_open()) {
        std::cerr << "Error abriendo journal" << std::endl;
        return false;
    }
    
    journalFile.seekg(0, std::ios::end);
    lastPosition = journalFile.tellg();
    
    return true;
}

void JournalParser::StartWatching(std::function<void(const JournalCameraData&)> onCameraUpdate) {
    if (isWatching) return;
    
    callback = onCameraUpdate;
    isWatching = true;
    watchThread = std::thread(&JournalParser::WatchLoop, this);
}

void JournalParser::StopWatching() {
    isWatching = false;
    if (watchThread.joinable()) {
        watchThread.join();
    }
}

void JournalParser::WatchLoop() {
    std::string contentBuffer;
    
    while (isWatching) {
        std::string newContent = ReadNewContent();
        
        if (!newContent.empty()) {
            contentBuffer += newContent;
            
            // Buscar directivas de cámara
            JournalCameraData cameraData;
            if (ParseCameraDirective(contentBuffer, cameraData)) {
                // Limpiar buffer hasta la última coincidencia procesada
                size_t lastMatch = contentBuffer.rfind("AutoCamCamera");
                if (lastMatch != std::string::npos) {
                    contentBuffer = contentBuffer.substr(lastMatch);
                }
                
                // Invocar callback
                if (callback) {
                    callback(cameraData);
                }
            }
            
            // Limitar tamaño del buffer
            if (contentBuffer.size() > 15000) {
                contentBuffer = contentBuffer.substr(contentBuffer.size() - 10000);
            }
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

std::string JournalParser::ReadNewContent() {
    if (!journalFile.is_open()) return "";
    
    journalFile.clear();
    journalFile.seekg(0, std::ios::end);
    std::streampos currentEnd = journalFile.tellg();
    
    if (currentEnd > lastPosition) {
        journalFile.seekg(lastPosition);
        
        std::string content;
        content.resize(static_cast<size_t>(currentEnd - lastPosition));
        journalFile.read(&content[0], content.size());
        
        lastPosition = currentEnd;
        return content;
    }
    
    return "";
}

bool JournalParser::ParseCameraDirective(const std::string& content, JournalCameraData& cameraData) {
    std::smatch match;
    
    if (std::regex_search(content, match, cameraRegex)) {
        try {
            // --- SIN CAMBIOS EN ESTA LÓGICA DE COORDENADAS ---
            float eye_revit_x = std::stof(match[3].str());
            float eye_revit_y = std::stof(match[1].str());
            float eye_revit_z = std::stof(match[2].str());

            float target_revit_x = std::stof(match[6].str());
            float target_revit_y = std::stof(match[4].str());
            float target_revit_z = std::stof(match[5].str());

            float up_revit_x = std::stof(match[9].str());
            float up_revit_y = std::stof(match[7].str());
            float up_revit_z = std::stof(match[8].str());

            cameraData.eyePosition = make_float3(eye_revit_x, eye_revit_y, eye_revit_z);
            cameraData.targetPosition = make_float3(target_revit_x, target_revit_y, target_revit_z);
            cameraData.upDirection = make_float3(up_revit_x, up_revit_y, up_revit_z);
            // --- FIN DE LA LÓGICA DE COORDENADAS ---
            
            cameraData.timestamp = std::chrono::high_resolution_clock::now().time_since_epoch().count();
            cameraData.sequenceNumber = globalSequenceNumber.fetch_add(1);
            
            return true;
        }
        catch (const std::exception& e) {
            std::cerr << "Error parseando datos de cámara: " << e.what() << std::endl;
        }
    }
    
    return false;
}

std::string JournalParser::FindLatestJournalFile(const std::string& revitVersion) {
    try {
        std::string journalDir = std::string(getenv("LOCALAPPDATA")) + 
            "\\Autodesk\\Revit\\Autodesk Revit " + revitVersion + "\\Journals";
        
        if (!std::filesystem::exists(journalDir)) {
            std::cerr << "Directorio de journals no existe: " << journalDir << std::endl;
            return "";
        }
        
        std::filesystem::path latestFile;
        std::filesystem::file_time_type latestTime;
        
        for (const auto& entry : std::filesystem::directory_iterator(journalDir)) {
            if (entry.path().extension() == ".txt" && 
                entry.path().filename().string().find("journal") != std::string::npos) {
                
                auto writeTime = std::filesystem::last_write_time(entry);
                if (latestFile.empty() || writeTime > latestTime) {
                    latestFile = entry.path();
                    latestTime = writeTime;
                }
            }
        }
        
        return latestFile.string();
    }
    catch (const std::exception& e) {
        std::cerr << "Error buscando journal: " << e.what() << std::endl;
        return "";
    }
}

void JournalParser::ProcessInitialCamera(std::function<void(const JournalCameraData&)> onCameraUpdate) {
    if (!journalFile.is_open() || !onCameraUpdate) {
        return;
    }

    std::cout << "[JOURNAL] Buscando posición de cámara inicial..." << std::endl;

    std::streampos endPosition = journalFile.tellg();
    journalFile.seekg(0, std::ios::beg);
    std::string content(
        (std::istreambuf_iterator<char>(journalFile)),
        std::istreambuf_iterator<char>()
    );

    // --- INICIO DE LA LÓGICA CORREGIDA ---
    // En lugar de usar sregex_iterator en todo el archivo (lo que causa el error de complejidad),
    // usamos una búsqueda de texto rápida para encontrar la última directiva y la parseamos.
    
    bool foundCamera = false;
    // 1. Buscar la última aparición de la clave "Jrn.Directive".
    size_t lastDirectivePos = content.rfind("Jrn.Directive");

    while (lastDirectivePos != std::string::npos) {
        // 2. Extraer el fragmento de texto desde esa posición hasta el final.
        std::string snippet = content.substr(lastDirectivePos);
        
        JournalCameraData tempCamera;
        // 3. Intentar parsear este fragmento. La expresión regular ahora opera en un texto pequeño.
        if (ParseCameraDirective(snippet, tempCamera)) {
            // Si tiene éxito, hemos encontrado la última cámara válida.
            std::cout << "[JOURNAL] Última cámara válida encontrada y procesada." << std::endl;
            onCameraUpdate(tempCamera);
            foundCamera = true;
            break; // Salimos del bucle.
        }
        
        // 4. Si falla (quizás era otra directiva), buscamos la aparición anterior.
        if (lastDirectivePos == 0) break; // Evitar bucle infinito
        lastDirectivePos = content.rfind("Jrn.Directive", lastDirectivePos - 1);
    }
    
    if (!foundCamera) {
        std::cout << "[JOURNAL] ADVERTENCIA: No se encontró ninguna directiva de cámara válida en el journal." << std::endl;
        std::cout << "[JOURNAL] El renderizador esperará a que se mueva la cámara en Revit." << std::endl;
    }
    // --- FIN DE LA LÓGICA CORREGIDA ---

    journalFile.clear(); 
    journalFile.seekg(endPosition);
    lastPosition = journalFile.tellg();
}