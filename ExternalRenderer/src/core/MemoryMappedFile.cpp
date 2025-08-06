// ExternalRenderer/src/core/MemoryMappedFile.cpp
#include "MemoryMappedFile.h"
#include <stdexcept>
#include <iostream>

MemoryMappedFile::MemoryMappedFile() 
    : hFile(INVALID_HANDLE_VALUE), hMapFile(NULL), pMapView(nullptr), fileSize(0), isReadOnly(true) {
}

MemoryMappedFile::~MemoryMappedFile() {
    Close();
}

bool MemoryMappedFile::Open(const std::string& filename, bool readOnly) {
    Close();
    
    this->isReadOnly = readOnly;
    
    // Abrir archivo
    hFile = CreateFileA(
        filename.c_str(),
        readOnly ? GENERIC_READ : (GENERIC_READ | GENERIC_WRITE),
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    
    if (hFile == INVALID_HANDLE_VALUE) {
        std::cerr << "Error abriendo archivo: " << GetLastError() << std::endl;
        return false;
    }
    
    // Obtener tamaño del archivo
    LARGE_INTEGER liSize;
    if (!GetFileSizeEx(hFile, &liSize)) {
        CloseHandle(hFile);
        return false;
    }
    fileSize = static_cast<size_t>(liSize.QuadPart);
    
    // Crear mapping
    hMapFile = CreateFileMappingA(
        hFile,
        NULL,
        readOnly ? PAGE_READONLY : PAGE_READWRITE,
        0, 0,
        NULL
    );
    
    if (hMapFile == NULL) {
        CloseHandle(hFile);
        return false;
    }
    
    // Mapear vista
    pMapView = MapViewOfFile(
        hMapFile,
        readOnly ? FILE_MAP_READ : FILE_MAP_ALL_ACCESS,
        0, 0, 0
    );
    
    if (pMapView == NULL) {
        CloseHandle(hMapFile);
        CloseHandle(hFile);
        return false;
    }
    
    return true;
}

bool MemoryMappedFile::OpenShared(const std::string& name, size_t size) {
    Close();
    
    // Abrir MMF compartido existente
    hMapFile = OpenFileMappingA(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        name.c_str()
    );
    
    if (hMapFile == NULL) {
        // Si no existe y tenemos tamaño, intentar crear
        if (size > 0) {
            hMapFile = CreateFileMappingA(
                INVALID_HANDLE_VALUE,
                NULL,
                PAGE_READWRITE,
                0,
                static_cast<DWORD>(size),
                name.c_str()
            );
        }
        
        if (hMapFile == NULL) {
            return false;
        }
    }
    
    // Mapear vista
    pMapView = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0, 0, 0
    );
    
    if (pMapView == NULL) {
        CloseHandle(hMapFile);
        return false;
    }
    
    // Si no conocemos el tamaño, usar VirtualQuery
    if (size == 0) {
        MEMORY_BASIC_INFORMATION mbi;
        if (VirtualQuery(pMapView, &mbi, sizeof(mbi))) {
            fileSize = mbi.RegionSize;
        }
    } else {
        fileSize = size;
    }
    
    return true;
}

void MemoryMappedFile::Close() {
    if (pMapView) {
        UnmapViewOfFile(pMapView);
        pMapView = nullptr;
    }
    
    if (hMapFile) {
        CloseHandle(hMapFile);
        hMapFile = NULL;
    }
    
    if (hFile != INVALID_HANDLE_VALUE) {
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
    }
    
    fileSize = 0;
}