cmake_minimum_required(VERSION 3.0)
project(rgbd-dense-visual-odometry)

# Set default build type to Release
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# Check C++17
include(CheckCXXCompilerFlag)
message("Using C++17")
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
    message(
        FATAL_ERROR
            "${CMAKE_CXX_COMPILER} はC++17をサポートしてません。C++17に対応したコンパイラを指定してください。\n"
        )
endif()
set(CMAKE_CXX_STANDARD 17)
message("Compiler:\n\t${CMAKE_CXX_COMPILER}")

# Set warning flags
set(CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wconversion
    -Wswitch-default
    -Wdisabled-optimization
    -Wformat
    -Winit-self
    -Woverloaded-virtual
    -Wfloat-equal
    -Wno-old-style-cast
    -Wno-pragmas)
foreach(FLAG IN LISTS CXX_WARNING_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()
message("Build type:\n\t${CMAKE_BUILD_TYPE}")

# Optimize flag
if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(OPTIMIZE_FLAGS "-mtune=native -march=native -mfpmath=both -Ofast")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OPTIMIZE_FLAGS}")
    message("Optimization flags:\n\t${OPTIMIZE_FLAGS}")
else()
    message("Not Optimized:\n\tBecause build type is ${CMAKE_BUILD_TYPE}")
endif()

# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
message("OpenMP found:\n\t${OpenMP_FOUND}")

# OpenCV
find_package(OpenCV REQUIRED)
message("OpenCV version:\n\t${OpenCV_VERSION}")

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIRS})

# matplotlib
#find_package(PythonLibs)

# camera-calibration
add_subdirectory(external/camera-calibration)
include_directories(
    ${CMAKE_CURRENT_LIST_DIR}/external/camera-calibration/include)

# camera-calibration
add_subdirectory(external/glfw-drawer)
include_directories(${CMAKE_CURRENT_LIST_DIR}/external/glfw-drawer/include)

# header & source
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB SOURCES src/*/*.cpp)

# library
add_library(vo-core STATIC ${SOURCES})
target_link_libraries(vo-core ${OpenCV_LIBS})

# cmake-format: off
# =============================
# add_executable(test-se3 test/se3.cpp)
# target_link_libraries(test-se3 ${OpenCV_LIBS} vo-core)
# add_executable(test-warp test/warp.cpp)
# target_link_libraries(test-warp ${OpenCV_LIBS} vo-core)
# add_executable(test-load test/load.cpp )
# target_link_libraries(test-load ${OpenCV_LIBS} vo-core)
# add_executable(test-track test/track.cpp)
# target_link_libraries(test-track ${OpenCV_LIBS} ${PYTHON_LIBRARIES} vo-core)
# target_include_directories(test-track SYSTEM PRIVATE external/matplotlib-cpp ${PYTHON_INCLUDE_DIRS})
# add_executable(test-seq test/sequence.cpp)
# target_link_libraries(test-seq ${OpenCV_LIBS} vo-core)
# add_executable(test-propagate test/propagate.cpp)
# target_link_libraries(test-propagate ${OpenCV_LIBS} vo-core)
# add_executable(test-update test/update.cpp)
# target_link_libraries(test-update ${OpenCV_LIBS} vo-core)
# add_executable(test-regular test/regularize.cpp)
# target_link_libraries(test-regular ${OpenCV_LIBS} vo-core)
# add_executable(test-record test/record.cpp)
# target_link_libraries(test-record ${OpenCV_LIBS} stdc++fs)
add_executable(test-step test/step.cpp)
target_link_libraries(test-step ${OpenCV_LIBS} vo-core glfw-drawer)
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS} vo-core glfw-drawer)
# add_executable(test-kinect test/kinect-vo.cpp)
# target_link_libraries(test-kinect ${OpenCV_LIBS} vo-core)
# add_executable(test-omp test/omp.cpp)
# target_link_libraries(test-omp ${OpenCV_LIBS} )
# =============================
# cmake-format: on
