cmake_minimum_required(VERSION 3.22)
project(simple-llama)

# Enable export of compile commands and set C++ standard
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_BUILD_TYPE Release)

# Find the OpenMP package
find_package(OpenMP REQUIRED)

# Function to setup target properties
function(configure_target target_name)
    target_compile_options(${target_name} PUBLIC
        -Wall -Wextra -g -Wpedantic
        -fdiagnostics-color=always
        -march=native -ffast-math -fPIC
    )
    target_link_libraries(${target_name} PUBLIC
        OpenMP::OpenMP_CXX
    )
endfunction()

# Llama library
add_library(llama SHARED llama.cc)
configure_target(llama)

# Quantize library
add_library(quantize SHARED quantize.cc)
configure_target(quantize)

# Main executable
add_executable(main main.cc)
configure_target(main)
target_link_libraries(main PRIVATE llama quantize)
