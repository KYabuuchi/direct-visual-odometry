cmake_minimum_required(VERSION 3.0)
project(rgbd-dense-visual-odometry)

# Set default build type to Release
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV
find_package(OpenCV 3 REQUIRED)
message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# Eigen
find_package(Eigen3 REQUIRED)

# コンパイラーがC++17に対応しているかチェック
include(CheckCXXCompilerFlag)
message("Using C++17")
enable_language(CXX)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
  message(
    FATAL_ERROR
      "${CMAKE_CXX_COMPILER} はC++17をサポートしてません。C++17に対応したコンパイラを指定してください。\n")
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

include_directories(${CMAKE_CURRENT_LIST_DIR}/include)

file(GLOB SOURCES src/*.cpp)
add_executable(main main.cpp)
target_link_libraries(main ${OpenCV_LIBS})

add_executable(test-se3 test/se3.cpp src/math.cpp)
target_link_libraries(test-se3 ${OpenCV_LIBS})

add_executable(test-map test/map_depth_to_gray.cpp ${SOURCES})
target_link_libraries(test-map ${OpenCV_LIBS})

add_executable(test-kinect test/kinect_record.cpp)
target_link_libraries(test-kinect ${OpenCV_LIBS} freenect2)

add_executable(test-load test/load.cpp)
target_link_libraries(test-load ${OpenCV_LIBS})

add_executable(test-generator test/generator.cpp)
target_link_libraries(test-generator ${OpenCV_LIBS})

# matplotlib
find_package(PythonLibs 3.6)

add_executable(tracking test/tracking.cpp ${SOURCES})
target_link_libraries(tracking ${OpenCV_LIBS} ${PYTHON_LIBRARIES})
target_include_directories(
  tracking SYSTEM
  PRIVATE external/matplotlib-cpp ${PYTHON_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
