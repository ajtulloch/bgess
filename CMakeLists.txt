cmake_minimum_required(VERSION 2.8.9)
project(bgess)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mavx2")

INCLUDE(ExternalProject)

# Set default ExternalProject root directory
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)

# Add BENCHMARK
# http://stackoverflow.com/questions/9689183/cmake-benchmark
ExternalProject_Add(
  googlebenchmark
  GIT_REPOSITORY https://github.com/google/benchmark
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/googlebenchmark
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=RELEASE
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  UPDATE_COMMAND ""
  )

ExternalProject_Get_Property(googlebenchmark install_dir)
include_directories(${install_dir}/include)

find_library(ACCELERATE Accelerate)
if (NOT ACCELERATE)
    message(FATAL_ERROR "Accelerate not found")
endif()


add_executable(bgess main.cpp)
add_dependencies(bgess googlebenchmark)
target_link_libraries(bgess ${install_dir}/lib/libbenchmark.a)
target_link_libraries(bgess ${ACCELERATE})
