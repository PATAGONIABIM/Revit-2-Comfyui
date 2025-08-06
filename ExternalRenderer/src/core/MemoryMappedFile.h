// ExternalRenderer/src/core/MemoryMappedFile.h
#pragma once
#include <string>
#include <vector>
#include <Windows.h>

class MemoryMappedFile {
private:
    HANDLE hFile;
    HANDLE hMapFile;
    LPVOID pMapView;
    size_t fileSize;
    bool isReadOnly;

public:
    MemoryMappedFile();
    ~MemoryMappedFile();

    bool Open(const std::string& filename, bool readOnly = true);
    bool OpenShared(const std::string& name, size_t size = 0);
    void Close();

    void* GetData() const { return pMapView; }
    size_t GetSize() const { return fileSize; }
    bool IsOpen() const { return pMapView != nullptr; }

    template<typename T>
    T* GetDataAs() { return static_cast<T*>(pMapView); }

    template<typename T>
    T Read(size_t offset) {
        if (offset + sizeof(T) > fileSize) throw std::out_of_range("Read beyond file size");
        return *reinterpret_cast<T*>(static_cast<char*>(pMapView) + offset);
    }
};