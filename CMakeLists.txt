cmake_minimum_required(VERSION 3.26)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX)
# project(navtoolbox LANGUAGES CXX)
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

pybind11_add_module(navtoolbox MODULE ${CMAKE_SOURCE_DIR}/src/navtoolbox/src/navtoolbox.cpp)
set_target_properties(navtoolbox PROPERTIES PREFIX "" OUTPUT_NAME "navtoolbox")

pybind11_add_module(earth MODULE ${CMAKE_SOURCE_DIR}/src/navtoolbox/src/earth.cpp)
target_link_libraries(earth PRIVATE Eigen3::Eigen) 
set_target_properties(earth PROPERTIES PREFIX "" OUTPUT_NAME "earth")

install(TARGETS navtoolbox earth DESTINATION ${SKBUILD_PROJECT_NAME})
# install(TARGETS navtoolbox earth LIBRARY DESTINATION .)