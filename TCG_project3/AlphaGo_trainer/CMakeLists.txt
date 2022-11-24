cmake_minimum_required(VERSION 3.0.0)
project(my_alpha_zero VERSION 0.1.0)

set(CUDNN_INCLUDE_DIR "/usr/lib/cuda")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare -std=c++2a ${TORCH_CXX_FLAGS}")

#file(GLOB NEURAL_SRC "neural/*.cpp")

#add_executable("${PROJECT_NAME}" main.cpp ${NEURAL_SRC} ${TTT_SRC})
#add_executable("${PROJECT_NAME}" main.cpp)
add_executable("${PROJECT_NAME}" evaluate_model.cpp)
target_link_libraries("${PROJECT_NAME}" PUBLIC "${TORCH_LIBRARIES}")
set_property(TARGET "${PROJECT_NAME}" PROPERTY CXX_STANDARD 20)
