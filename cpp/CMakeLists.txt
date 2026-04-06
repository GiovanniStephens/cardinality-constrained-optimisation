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

# Platform-specific linking
if(APPLE)
    # Apple Accelerate framework provides optimised BLAS (cblas_dgemm) for ARM64/AMX.
    # EIGEN_USE_BLAS makes Eigen delegate matrix ops to Accelerate transparently.
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    target_link_libraries(optimisation PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(optimisation PRIVATE EIGEN_USE_BLAS)
else()
    target_link_libraries(optimisation PRIVATE pthread)
endif()
