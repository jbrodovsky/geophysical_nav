cmake_minimum_required(VERSION 3.26)
# project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
project(navtoolbox LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PYBIND11_FINDPYTHON ON)

find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

if (Eigen3_FOUND)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Eigen3 not found")
endif()

include_directories(${CMAKE_SOURCE_DIR}/src/navtoolbox/include)

add_library(navtoolbox SHARED 
    ${CMAKE_SOURCE_DIR}/src/navtoolbox/src/navtoolbox.cpp
    ${CMAKE_SOURCE_DIR}/src/navtoolbox/src/earth.cpp
    ${CMAKE_SOURCE_DIR}/src/navtoolbox/src/transform.cpp
    #${CMAKE_SOURCE_DIR}/src/navtoolbox/src/util.cpp
    )

target_link_libraries(navtoolbox PRIVATE Eigen3::Eigen pybind11::module)

# Set the output directory for the Python module
set_target_properties(navtoolbox PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    PREFIX ""
    SUFFIX ".so"
)

#install(TARGETS navtoolbox LIBRARY DESTINATION ./src/)#