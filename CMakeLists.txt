cmake_minimum_required(VERSION 2.8.9)
project(bgess)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mavx2 -march=native -Wno-c++11-narrowing")

INCLUDE(ExternalProject)

# Set default ExternalProject root directory
SET_DIRECTORY_PROPERTIES(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/third_party)

set(eigen_VERSION "3.3.3" )

ExternalProject_Add( eigen
  URL "http://bitbucket.org/eigen/eigen/get/${eigen_VERSION}.tar.gz"
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/eigen
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=RELEASE
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  UPDATE_COMMAND ""
)

ExternalProject_Add(
  gflags
  GIT_REPOSITORY https://github.com/gflags/gflags
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gflags
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=RELEASE
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  UPDATE_COMMAND ""
  )

ExternalProject_Add(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/gtest
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=RELEASE
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  UPDATE_COMMAND ""
  )

ExternalProject_Add(
  glog
  GIT_REPOSITORY https://github.com/google/glog
  PREFIX ${CMAKE_CURRENT_BINARY_DIR}/glog
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR> -DCMAKE_BUILD_TYPE=RELEASE
  LOG_DOWNLOAD ON
  LOG_CONFIGURE ON
  LOG_BUILD ON
  UPDATE_COMMAND ""
  )

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


find_library(ACCELERATE Accelerate)
if (NOT ACCELERATE)
    message(FATAL_ERROR "Accelerate not found")
endif()


add_executable(bgess main.cpp ulp.cc ulp_test.cc)
add_dependencies(bgess googlebenchmark gflags gtest eigen)

ExternalProject_Get_Property(gflags install_dir)
include_directories(${install_dir}/include)
target_link_libraries(bgess ${install_dir}/lib/libgflags.a)

ExternalProject_Get_Property(googlebenchmark install_dir)
include_directories(${install_dir}/include)
target_link_libraries(bgess ${install_dir}/lib/libbenchmark.a)

ExternalProject_Get_Property(gtest install_dir)
include_directories(${install_dir}/include)
target_link_libraries(bgess ${install_dir}/lib/libgtest.a)

ExternalProject_Get_Property(glog install_dir)
include_directories(${install_dir}/include)
target_link_libraries(bgess ${install_dir}/lib/libglog.a)

ExternalProject_Get_Property(eigen install_dir)
include_directories(${install_dir}/include)

target_link_libraries(bgess ${ACCELERATE})
