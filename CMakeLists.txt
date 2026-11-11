cmake_minimum_required(VERSION 3.24)
set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures")
project(OOPS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

add_subdirectory(external/owl EXCLUDE_FROM_ALL)
find_package(CUDAToolkit REQUIRED)

add_library(oops_common src/Scene.cpp src/Reference.cpp)
target_include_directories(oops_common PUBLIC src external/owl/include)

embed_ptx(OUTPUT_TARGET mochi_ptx
    PTX_LINK_LIBRARIES owl::owl
    SOURCES src/OptixPrograms.cu)

add_library(oops_gpu src/OptixBackend.cpp src/Granular.cu src/Simulation.cpp src/Render.cpp)
target_link_libraries(oops_gpu PUBLIC oops_common owl::owl CUDA::cudart PRIVATE mochi_ptx stb_image)

add_executable(mochi src/main.cpp)
target_link_libraries(mochi PRIVATE oops_gpu)

include(CTest)
if(BUILD_TESTING)
    add_executable(mochi_tests tests/validation.cpp)
    target_link_libraries(mochi_tests PRIVATE oops_gpu)
    add_test(NAME mochi_cpu COMMAND mochi_tests --cpu-only)
    add_test(NAME mochi_gpu COMMAND mochi_tests)
endif()

option(MOCHI_BUILD_UPSTREAM "Build a harness around the authors' original kernels" OFF)
if(MOCHI_BUILD_UPSTREAM)
    set(MOCHI_UPSTREAM_DIR "${CMAKE_SOURCE_DIR}/build/reference/mochi" CACHE PATH "Fetched Mochi source")
    include(tools/upstream.cmake)
endif()
