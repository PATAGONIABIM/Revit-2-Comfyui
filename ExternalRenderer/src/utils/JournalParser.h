// ExternalRenderer/src/utils/JournalParser.h
#pragma once
#include <string>
#include <cuda_runtime.h>
#include <regex>
#include <fstream>
#include <atomic>
#include <thread>
#include <functional>

struct JournalCameraData {
    float3 eyePosition;
    float3 targetPosition; 
    float3 upDirection;
    uint64_t timestamp;
    int sequenceNumber;
};

class JournalParser {
private:
    std::string journalPath;
    std::ifstream journalFile;
    std::streampos lastPosition;
    std::regex cameraRegex;
    std::atomic<bool> isWatching;
    std::thread watchThread;
    std::function<void(const JournalCameraData&)> callback;
    
    static std::atomic<int> globalSequenceNumber;

public:
    JournalParser();
    ~JournalParser();
    
    bool Initialize(const std::string& revitVersion = "2026");
    void ProcessInitialCamera(std::function<void(const JournalCameraData&)> onCameraUpdate);
    void StartWatching(std::function<void(const JournalCameraData&)> onCameraUpdate);
    void StopWatching();
    
    static std::string FindLatestJournalFile(const std::string& revitVersion);
    
private:
    void WatchLoop();
    bool ParseCameraDirective(const std::string& content, JournalCameraData& cameraData);
    std::string ReadNewContent();
};