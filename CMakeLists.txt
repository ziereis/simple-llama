cmake_minimum_required(VERSION 3.22)
project(simple-llama)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_BUILD_TYPE Release)
find_package(OpenMP)


add_executable(llama llama.cc)
target_compile_options(llama PUBLIC -Wall -Wextra -Wpedantic -fdiagnostics-color=always  -Ofast -march=native -ffast-math )
target_link_libraries(llama OpenMP::OpenMP_CXX)
