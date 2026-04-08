cmake_minimum_required(VERSION 3.14)
project(portfolio_ga CXX OBJCXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

add_compile_options(-O3 -march=native -DNDEBUG -ffast-math -fno-finite-math-only)

# Header-only dependencies: prefer repo submodules, fall back to system Eigen
if(EXISTS "${CMAKE_SOURCE_DIR}/../eigen/Eigen")
    include_directories(${CMAKE_SOURCE_DIR}/../eigen)
else()
    find_package(Eigen3 REQUIRED)
    include_directories(${EIGEN3_INCLUDE_DIR})
endif()
include_directories(${CMAKE_SOURCE_DIR}/../csv-parser/single_include)

set(SOURCES optimisation.cpp)

# Platform-specific linking
if(APPLE)
    # Apple Accelerate framework provides optimised BLAS (cblas_dgemm) for ARM64/AMX.
    # EIGEN_USE_BLAS makes Eigen delegate matrix ops to Accelerate transparently.
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        list(APPEND SOURCES metal_fitness.mm)
        set_source_files_properties(metal_fitness.mm PROPERTIES
            COMPILE_FLAGS "-fobjc-arc"
            LANGUAGE OBJCXX)
    endif()

    add_executable(optimisation ${SOURCES})
    target_link_libraries(optimisation PRIVATE ${ACCELERATE_FRAMEWORK})
    target_compile_definitions(optimisation PRIVATE EIGEN_USE_BLAS)

    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK)
        target_link_libraries(optimisation PRIVATE ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK})
        target_compile_definitions(optimisation PRIVATE HAS_METAL)
    endif()
else()
    add_executable(optimisation ${SOURCES})
    target_link_libraries(optimisation PRIVATE pthread)
endif()
