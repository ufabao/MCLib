cmake_minimum_required(VERSION 3.27)
project(MCLib)
set(CMAKE_CXX_STANDARD 23)

find_package(Catch2 3 REQUIRED)
add_executable(tests tests.cpp newthreadpool.cpp)
target_link_libraries(tests PRIVATE Catch2::Catch2WithMain)

find_package(benchmark REQUIRED)
add_executable(benchmarks benchmarks.cpp newthreadpool.cpp)
target_link_libraries(benchmarks benchmark::benchmark)

add_library(MCLib MCLib.cpp)