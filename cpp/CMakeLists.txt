cmake_minimum_required(VERSION 3.14)
project(portfolio_ga CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

add_compile_options(-O3 -march=native -DNDEBUG)

# Header-only dependencies live in the repo root
include_directories(${CMAKE_SOURCE_DIR}/../eigen)
include_directories(${CMAKE_SOURCE_DIR}/../csv-parser/single_include)

add_executable(optimisation optimisation.cpp)
target_link_libraries(optimisation PRIVATE pthread)
