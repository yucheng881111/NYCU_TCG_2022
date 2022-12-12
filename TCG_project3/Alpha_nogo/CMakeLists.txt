cmake_minimum_required(VERSION 3.0.0)
project(nogo VERSION 0.1.0)

set(CUDNN_INCLUDE_DIR "/usr/lib/cuda")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -std=c++2a -O3 -fopenmp ${TORCH_CXX_FLAGS}")

#file(GLOB NEURAL_SRC "neural/*.cpp")

add_executable("${PROJECT_NAME}" nogo.cpp)
target_link_libraries("${PROJECT_NAME}" PUBLIC "${TORCH_LIBRARIES}")
set_property(TARGET "${PROJECT_NAME}" PROPERTY CXX_STANDARD 20)
