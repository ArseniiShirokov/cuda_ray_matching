project(NAIR-V)

cmake_minimum_required(VERSION 3.8)


set(CMAKE_CXX_STANDARD             17)
set(CMAKE_MODULE_PATH              "${CMAKE_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_EXPORT_COMPILE_COMMANDS  ON)

find_package(PNG)
find_package(JPEG)
find_package(Catch REQUIRED)

include(cmake/TestSolution.cmake)
include(cmake/BuildFlags.cmake)

if (${PNG_FOUND} AND ${JPEG_FOUND})
  add_subdirectory(ray_matching_cpu)
  add_subdirectory(ray_matching_gpu)
elseif()
  message(STATUS "ray_matching disabled. PNG:${PNG_FOUND}")
endif()
