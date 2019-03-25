cmake_minimum_required(VERSION 3.0)
project(rgbd-dense-visual-odometry)

# Set default build type to Release
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
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

# OpenCV
find_package(OpenCV 3.0 REQUIRED)
message("OpenCV version:\n\t${OpenCV_VERSION}")

# matplotlib
find_package(PythonLibs)

# camera-calibration
add_subdirectory(camera-calibration)
include_directories(${CMAKE_CURRENT_LIST_DIR}/camera-calibration/include)

# header & source
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
file(GLOB CORE_SOURCES src/core/*.cpp)
file(GLOB TRACK_SOURCES src/track/*.cpp)

# cmake-format: off
# =============================

# add_executable(main main.cpp) 
# target_link_libraries(main ${OpenCV_LIBS})

add_executable(test-se3 test/se3.cpp src/core/math.cpp)
target_link_libraries(test-se3 ${OpenCV_LIBS})

add_executable(test-map test/map.cpp ${CORE_SOURCES})
target_link_libraries(test-map ${OpenCV_LIBS})

add_executable(test-load test/load.cpp src/core/params.cpp)
target_link_libraries(test-load ${OpenCV_LIBS} camera-calibration)

add_executable(test-gradient test/gradient.cpp ${CORE_SOURCES} )
target_link_libraries(test-gradient ${OpenCV_LIBS})

add_executable(test-axis test/axis_map.cpp ${CORE_SOURCES})
target_link_libraries(test-axis ${OpenCV_LIBS})

add_executable(test-tracking test/tracking.cpp ${CORE_SOURCES} ${TRACK_SOURCES})
target_link_libraries(test-tracking ${OpenCV_LIBS} ${PYTHON_LIBRARIES})
target_include_directories(test-tracking PRIVATE SYSTEM external/matplotlib-cpp ${PYTHON_INCLUDE_DIRS})

# =============================
# cmake-format: on
