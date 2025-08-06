cmake_minimum_required(VERSION 3.18)

project(WabiSabiExternalRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# --- Dependencias (usando el método clásico que nos funciona) ---
find_package(CUDAToolkit REQUIRED)
find_package(PNG REQUIRED)
find_package(jsoncpp REQUIRED)

message(STATUS "Usando CUDA Toolkit en: ${CUDAToolkit_ROOT}")
message(STATUS "PNG encontrada en: ${PNG_INCLUDE_DIRS}")
message(STATUS "jsoncpp encontrado en: ${jsoncpp_INCLUDE_DIRS}")

set(WEBSOCKETPP_ROOT "C:/websocketpp")

# --- Librería estática para el código CUDA ---
add_library(wabisabi_cuda STATIC
    src/cuda/WabiSabiKernels.cu
)

set_target_properties(wabisabi_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"
)

target_include_directories(wabisabi_cuda PUBLIC
    ${CUDAToolkit_INCLUDE_DIRS}
)
# Resolver símbolos de dispositivo CUDA
set_target_properties(wabisabi_cuda PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
# --- Ejecutable Principal (solo con código C++) ---
add_executable(WabiSabiRenderer
    src/main.cpp
    src/core/WabiSabiRenderer.cpp
    src/core/MemoryMappedFile.cpp
    src/utils/CSVReader.cpp
    src/utils/JournalParser.cpp
)

# --- Inclusión de Directorios para el Ejecutable ---
target_include_directories(WabiSabiRenderer PRIVATE
    "${WEBSOCKETPP_ROOT}"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

# --- Enlace de Librerías ---
target_link_libraries(WabiSabiRenderer PRIVATE
    wabisabi_cuda
    ws2_32
    ${CUDAToolkit_LIBRARIES}
    PNG::PNG
    JsonCpp::JsonCpp
)

# --- SOLUCIÓN PARA EL WARNING LIBCMT ---
# Se le dice al enlazador de MSVC que ignore la librería estática en conflicto.
if(MSVC)
    set_target_properties(WabiSabiRenderer PROPERTIES 
        LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    )
endif()
# Propiedades adicionales para el ejecutable
set_target_properties(WabiSabiRenderer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# --- Copia de Archivos ---
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/config/category_colors.csv")
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/config/category_colors.csv"
        "${CMAKE_CURRENT_BINARY_DIR}/category_colors.csv"
        COPYONLY
    )
endif()