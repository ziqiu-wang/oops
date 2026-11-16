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

include(FetchContent)
find_package(OpenGL REQUIRED)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG 3.4)

set(ONLY_LIBS ON CACHE BOOL "" FORCE)
set(glew-cmake_BUILD_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glew
    GIT_REPOSITORY https://github.com/Perlmint/glew-cmake.git
    GIT_TAG glew-cmake-2.2.0)

FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.5)

FetchContent_MakeAvailable(glfw glew imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

add_library(oops_common src/Scene.cpp src/Reference.cpp)
target_include_directories(oops_common PUBLIC src external/owl/include)

embed_ptx(OUTPUT_TARGET mochi_ptx
    PTX_LINK_LIBRARIES owl::owl
    SOURCES src/OptixPrograms.cu)

add_library(oops_gpu src/OptixBackend.cpp src/Granular.cu src/Simulation.cpp src/Render.cpp)
target_link_libraries(oops_gpu PUBLIC oops_common owl::owl CUDA::cudart PRIVATE mochi_ptx stb_image)

add_library(oops_window src/Window.cpp)
target_include_directories(oops_window PUBLIC src)
target_link_libraries(oops_window PUBLIC glfw libglew_static OpenGL::GL CUDA::cudart imgui)

add_executable(test_window tests/test_window.cpp)
target_link_libraries(test_window PRIVATE oops_window)

add_executable(mochi src/main.cpp)
target_link_libraries(mochi PRIVATE oops_gpu oops_window)

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
