cmake_minimum_required(VERSION 3.22)
project(simple-llama)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE Debug)
#find_package(OpenMP REQUIRED)


add_executable(llama llama.cc)
target_compile_options(llama PUBLIC -Wall -fopenmp -Wextra -g -Wpedantic -fdiagnostics-color=always  -Ofast -march=native -ffast-math )
target_link_libraries(llama)
target_link_options(llama PUBLIC -fopenmp)


add_executable(quantize quantize.cc)
target_compile_options(quantize PUBLIC -Wall -Wextra -fopenmp -g -Wpedantic -fdiagnostics-color=always  -Ofast -march=native -ffast-math)
target_link_libraries(quantize)
target_link_options(quantize PUBLIC -fopenmp)
