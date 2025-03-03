cmake_minimum_required(VERSION 3.15)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "SKBUILD_PROJECT_NAME: ${SKBUILD_PROJECT_NAME}")

# Common warnings and safety flags
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wshadow
    -Wnull-dereference
    -Wnon-virtual-dtor
    -Woverloaded-virtual
    -Wformat=2
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=2
)
# Additional flags for Debug mode
if (CMAKE_BUILD_TYPE MATCHES "Debug")
    add_compile_options(
        -fsanitize=address
        -fsanitize=undefined
        -fsanitize=leak
        -fsanitize=thread
        -fno-omit-frame-pointer
        -g
    )
    add_link_options(
        -fsanitize=address
        -fsanitize=undefined
        -fsanitize=leak
        -fsanitize=thread
    )
else()
    # Optimizations for Release
    add_compile_options(-O2 -DNDEBUG)
endif()

# Detect Pixi environment
set(PIXI_ENV_PATH .pixi/envs/$ENV{PIXI_ENVIRONMENT_NAME})

message(STATUS "PIXI_ENV_PATH: ${PIXI_ENV_PATH}")

# Tell CMake where to find dependencies
set(CMAKE_PREFIX_PATH ${PIXI_ENV_PATH})
set(CMAKE_INCLUDE_PATH ${PIXI_ENV_PATH}/include)
set(CMAKE_LIBRARY_PATH ${PIXI_ENV_PATH}/lib)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN_INCLUDE_DIR})
include_directories(src/include)

# If SKBUILD_PROJECT_NAME is not set, we are building the library and not the Python bindings

file(GLOB_RECURSE SRC_FILES src/src/*.cpp)
add_library(navtoolbox SHARED ${SRC_FILES})
target_link_libraries(navtoolbox PRIVATE Eigen3::Eigen)
add_executable(main src/src/main.cpp)
target_link_libraries(main PRIVATE navtoolbox)
target_link_libraries(main PRIVATE Eigen3::Eigen)
    # install to lib directory
    # install(TARGETS navtoolbox DESTINATION ./bin)
    # return()
#else()
if (SKBUILD_PROJECT_NAME)
    message(STATUS "Building Python bindings")
    # If SKBUILD_PROJECT_NAME is set, we are building the Python bindings
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    # Make sure to add the C++ source files for the Python module bindings
    pybind11_add_module(earth MODULE src/earth_module.cpp src/src/earth.cpp)
    target_link_libraries(earth PRIVATE Eigen3::Eigen)
    #pybind11_add_module(_core MODULE src/main.cpp)
    install(TARGETS earth DESTINATION ${SKBUILD_PROJECT_NAME})
endif()