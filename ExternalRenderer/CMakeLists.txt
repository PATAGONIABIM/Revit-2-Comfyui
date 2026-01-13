cmake_minimum_required(VERSION 3.18)

# --------------------------------------------------
# Project
# --------------------------------------------------

project(WabiSabiExternalRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --------------------------------------------------
# Global compile definitions
# --------------------------------------------------

add_compile_definitions(
    ASIO_ENABLE_DEPRECATED
    ASIO_STANDALONE
    _WEBSOCKETPP_CPP11_STL_
    _WEBSOCKETPP_CPP11_RANDOM_DEVICE_
)

# --------------------------------------------------
# Dependencies
# --------------------------------------------------

find_package(CUDAToolkit REQUIRED)
find_package(PNG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(asio CONFIG REQUIRED)

# websocketpp (header-only)
set(WEBSOCKETPP_ROOT "C:/websocketpp")

# --------------------------------------------------
# CUDA static library
# --------------------------------------------------

add_library(wabisabi_cuda STATIC
    src/cuda/WabiSabiKernels.cu
)

set_target_properties(wabisabi_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Include CUDA headers
target_include_directories(wabisabi_cuda PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)

# --------------------------------------------------
# Main executable
# --------------------------------------------------

add_executable(WabiSabiRenderer
    src/main.cpp
    src/core/WabiSabiRenderer.cpp
    src/core/ImageMMFWriter.cpp
    src/core/MemoryMappedFile.cpp
    src/utils/CSVReader.cpp
    src/utils/JournalParser.cpp
)

# Include directories for the executable
target_include_directories(WabiSabiRenderer PRIVATE
    "${WEBSOCKETPP_ROOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# --------------------------------------------------
# Link libraries
# --------------------------------------------------

target_link_libraries(WabiSabiRenderer PRIVATE
    wabisabi_cuda
    ws2_32
    CUDA::cudart
    PNG::PNG
    jsoncpp_lib
    asio::asio
)

# --------------------------------------------------
# MSVC specific options
# --------------------------------------------------

if (MSVC)
    set_target_properties(WabiSabiRenderer PROPERTIES
        LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    )
endif()

# --------------------------------------------------
# CUDA properties for executable
# --------------------------------------------------

set_target_properties(WabiSabiRenderer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# --------------------------------------------------
# Copy runtime config files
# --------------------------------------------------

file(COPY
    "${CMAKE_CURRENT_SOURCE_DIR}/config/"
    DESTINATION "${CMAKE_BINARY_DIR}/Release/"
)
