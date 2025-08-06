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
            // --- INICIO DE LA CORRECCIÓN ---
            // Paso 1: Leer las coordenadas del journal en el sistema de Revit (Z-up)
            // El journal las escribe en orden Y, Z, X
            float eye_revit_x = std::stof(match[3].str());
            float eye_revit_y = std::stof(match[1].str());
            float eye_revit_z = std::stof(match[2].str());

            float target_revit_x = std::stof(match[6].str());
            float target_revit_y = std::stof(match[4].str());
            float target_revit_z = std::stof(match[5].str());

            float up_revit_x = std::stof(match[9].str());
            float up_revit_y = std::stof(match[7].str());
            float up_revit_z = std::stof(match[8].str());

            // Paso 2: Transformar de sistema Z-up a Y-up para el renderizador
            // new_x = old_x
            // new_y = old_z
            // new_z = -old_y
            cameraData.eyePosition = make_float3(eye_revit_x, eye_revit_y, eye_revit_z);
            cameraData.targetPosition = make_float3(target_revit_x, target_revit_y, target_revit_z); // <-- Usando el miembro nuevo y correcto
            cameraData.upDirection = make_float3(up_revit_x, up_revit_y, up_revit_z);
            // --- FIN DE LA CORRECCIÓN ---
            
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

    // Guardar la posición final actual para restaurarla después
    std::streampos endPosition = journalFile.tellg();

    // Ir al principio del archivo
    journalFile.seekg(0, std::ios::beg);

    // Leer todo el contenido del archivo
    std::string content(
        (std::istreambuf_iterator<char>(journalFile)),
        std::istreambuf_iterator<char>()
    );

    // --- LÓGICA MEJORADA PARA ENCONTRAR LA ÚLTIMA CÁMARA ---
    // Usamos un iterador de regex para encontrar todas las coincidencias
    auto matches_begin = std::sregex_iterator(content.begin(), content.end(), cameraRegex);
    auto matches_end = std::sregex_iterator();

    JournalCameraData lastValidCamera;
    bool foundCamera = false;

    // Iteramos a través de todas las coincidencias y nos quedamos con la última
    for (std::sregex_iterator i = matches_begin; i != matches_end; ++i) {
        JournalCameraData tempCamera;
        if (ParseCameraDirective((*i).str(), tempCamera)) {
            lastValidCamera = tempCamera;
            foundCamera = true;
        }
    }

    if (foundCamera) {
        std::cout << "[JOURNAL] Última cámara válida encontrada y procesada." << std::endl;
        onCameraUpdate(lastValidCamera);
    } else {
        std::cout << "[JOURNAL] ADVERTENCIA: No se encontró ninguna directiva de cámara válida en el journal." << std::endl;
        std::cout << "[JOURNAL] El renderizador esperará a que se mueva la cámara en Revit." << std::endl;
    }
    // --- FIN DE LA LÓGICA MEJORADA ---

    // Restaurar el puntero del archivo al final para que WatchLoop funcione correctamente
    journalFile.clear(); // Limpiar flags de eof
    journalFile.seekg(endPosition);
    lastPosition = journalFile.tellg();
}