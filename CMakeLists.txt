cmake_minimum_required(VERSION 3.22)
project(cu_llama_app LANGUAGES CXX C)

# Enable export of compile commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

include_directories(${CMAKE_SOURCE_DIR})

# Find the OpenMP package
find_package(OpenMP REQUIRED)

# Function to setup target properties
function(configure_target target_name)
    target_compile_options(${target_name} PUBLIC
        -Wall -Wextra -Wpedantic
        -fdiagnostics-color=always
        -march=native -ffast-math -fPIC
    )
    target_link_libraries(${target_name} PUBLIC
        OpenMP::OpenMP_C m
    )
endfunction()

# Llama library
add_library(llama SHARED lib/llama.c lib/utils.c lib/ops.c)
configure_target(llama)

# Quantize library
add_library(quantize SHARED lib/quantize.c lib/utils.c lib/ops.c)
configure_target(quantize)
target_link_libraries(quantize PRIVATE llama)

# Main executable
add_executable(test lib/test.c lib/utils.c lib/ops.c)
configure_target(test)
target_link_libraries(test PRIVATE llama quantize)

# Option to compile CUDA library
option(CUDA "Compile CUDA library" OFF)

if(CUDA)
    enable_language(CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    add_library(cu_llama SHARED lib/cuda/cu-llama.cu lib/cuda/cu-ops.cu lib/utils.c lib/llama.c lib/ops.c)
    set_target_properties(cu_llama PROPERTIES
                        CUDA_SEPARABLE_COMPILATION ON)

    set_target_properties(cu_llama PROPERTIES
                        CXX_STANDARD 14
                        CXX_STANDARD_REQUIRED ON)

    target_compile_options(cu_llama PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           --use_fast_math
                           >)

    target_link_libraries(test PRIVATE cu_llama)
endif()
