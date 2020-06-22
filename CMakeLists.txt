# Set minimum required CMake version
cmake_minimum_required(VERSION 3.10)

project(bachelor_xvigra_convolution LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
if(MSVC)
	message(STATUS "MSVC build")
    set(CMAKE_EXE_LINKER_FLAGS /MANIFEST:NO)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4996 /wd4267 /wd4244")
else()
	message(STATUS "G++ build")
    set(CMAKE_CXX_FLAGS "-Wall -Wextra")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(XTENSOR_USE_XSIMD 0)

# Include directories
include_directories("include")
include_directories("src")

# Find installed libraries

find_package(doctest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(Threads REQUIRED)

add_definitions(-DHAVE_CBLAS=1)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(clapack CONFIG REQUIRED)
set(BLAS_LIBRARIES ${CMAKE_INSTALL_PREFIX}${OpenBLAS_LIBRARIES})

find_package(OpenImageIO CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)
find_package(xtl CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-io CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)

message(STATUS "")

message(STATUS "                             Test libs")
message(STATUS "──────────────────────────────────────────────────────────────────────")
message(STATUS "doctest_VERSION: ${doctest_VERSION}")
message(STATUS "")

message(STATUS "                           Benchmark libs")
message(STATUS "──────────────────────────────────────────────────────────────────────")
message(STATUS "benchmark_VERSION: ${benchmark_VERSION}")
message(STATUS "")

message(STATUS "                            Xtensor libs")
message(STATUS "──────────────────────────────────────────────────────────────────────")
message(STATUS "xtl_VERSION         : " ${xtl_VERSION})
message(STATUS "xtensor_VERSION     : " ${xtensor_VERSION})
message(STATUS "xtensor-io_VERSION  : " ${xtensor-io_VERSION})
message(STATUS "xtensor-blas_VERSION: " ${xtensor-blas_VERSION})
message(STATUS "")

message(STATUS "                        Xtensor dependencies")
message(STATUS "──────────────────────────────────────────────────────────────────────")
message(STATUS "clapack_VERSION    : " ${clapack_VERSION})
message(STATUS "OpenBLAS_VERSION   : " ${OpenBLAS_VERSION})
message(STATUS "OpenImageIO_VERSION: " ${OpenImageIO_VERSION})
message(STATUS "")

enable_testing()
add_subdirectory(tests)

add_subdirectory(benchmarks)

add_subdirectory(src)