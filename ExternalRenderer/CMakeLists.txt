cmake_minimum_required(VERSION 3.18)

project(WabiSabiExternalRenderer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)

# --- INICIO DE LA CORRECCIÓN FINAL ---
# Forzar las definiciones de compilación a nivel GLOBAL.
# Esto es más contundente y se aplica antes de que Vcpkg pueda interferir.
add_compile_definitions(
    ASIO_ENABLE_DEPRECATED
    ASIO_STANDALONE
    _WEBSOCKETPP_CPP11_STL_
    _WEBSOCKETPP_CPP11_RANDOM_DEVICE_
)
# --- FIN DE LA CORRECCIÓN FINAL ---

# --- Dependencias ---
find_package(CUDAToolkit REQUIRED)
find_package(PNG REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(asio CONFIG REQUIRED)

# Ruta manual para la librería que Vcpkg NO tiene
set(WEBSOCKETPP_ROOT "C:/websocketpp")

# --- Librería estática para el código CUDA ---
add_library(wabisabi_cuda STATIC
    src/cuda/WabiSabiKernels.cu
)
set_target_properties(wabisabi_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86"
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_include_directories(wabisabi_cuda PUBLIC ${CUDAToolkit_INCLUDE_DIRS})

# --- Ejecutable Principal ---
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

# --- EL BLOQUE target_compile_definitions SE HA ELIMINADO DE AQUÍ ---

# --- Enlace de Librerías ---
target_link_libraries(WabiSabiRenderer PRIVATE
    wabisabi_cuda
    ws2_32
    ${CUDAToolkit_LIBRARIES}
    PNG::PNG
    jsoncpp_lib
    asio::asio
)

# --- El resto del archivo permanece igual ---
if(MSVC)
    set_target_properties(WabiSabiRenderer PROPERTIES 
        LINK_FLAGS "/NODEFAULTLIB:LIBCMT"
    )
endif()

set_target_properties(WabiSabiRenderer PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/config/" DESTINATION "${CMAKE_BINARY_DIR}/Release/")