cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(cu_llama_app LANGUAGES CXX C CUDA)

# Set the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 50)
set(CMAKE_BUILD_TYPE Release)

# Include the directories where the CUDA headers are located
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Add the shared library from CUDA sources
add_library(cu_llama SHARED cu-llama.cu cu-ops.cu ../utils.c ../llama.c ../ops.c)
set_target_properties(cu_llama PROPERTIES LINKER_LANGUAGE CXX)
# Enable separable compilation for the CUDA library
set_target_properties(cu_llama PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# Set C++ standard for the CUDA library
set_target_properties(cu_llama PROPERTIES
                      CXX_STANDARD 14
                      CXX_STANDARD_REQUIRED ON)


# Add an executable that will use the shared library
add_executable(cu_app main.cc ../utils.c ../llama.c ../ops.c)

# Explicitly set the linker language for the executable to C++

#Link the executable with the shared CUDA library
target_link_libraries(cu_app PRIVATE cu_llama)
