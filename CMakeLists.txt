cmake_minimum_required(VERSION 3.20)
project(lfStreamer LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -pedantic -fmodules-ts")
set(${PROJECT_NAME}_VERSION_MAJOR 0)
set(${PROJECT_NAME}_VERSION_MINOR 0)


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMakeModules")

set(SOURCES 
src/decoder.cpp
src/muxing.cpp
src/renderer.cpp
src/lfStreamerApp.cpp
src/videoDecoder.cpp
src/interpolator.cpp
src/cudaGLInterop.cu
src/exporter.cu
src/kernels.cu
src/libs/loadingBar/loadingbar.hpp
src/libs/arguments/arguments.hpp
)

set(SOURCES_ENCODER
src/muxing.cpp
src/lfEncoderApp.cpp
src/encoder.cpp
src/libs/loadingBar/loadingbar.hpp
src/libs/arguments/arguments.hpp
)

find_package(CUDAToolkit)
find_package(glm REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(FFMPEG REQUIRED)
add_library(ffmpeg INTERFACE)
target_link_libraries(ffmpeg INTERFACE ${FFMPEG_LIBRARIES})
target_include_directories(ffmpeg INTERFACE $<BUILD_INTERFACE:${FFMPEG_INCLUDE_DIRS})

set(ENCODER lfEncoder)
add_executable(${ENCODER} ${SOURCES_ENCODER})

add_executable(${PROJECT_NAME} ${SOURCES})
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/src/libs)
include_directories(${CUDAToolkit_INCLUDE_DIRS}) 

target_link_libraries(${PROJECT_NAME} glm glfw GLEW::GLEW ${OPENGL_LIBRARIES} ${CUDA_LIBRARIES} nvcuvid cuda)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored>)

target_link_libraries(${ENCODER} ffmpeg swscale)

