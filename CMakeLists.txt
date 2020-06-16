# Set minimum required CMake version
cmake_minimum_required(VERSION 3.10)

project(bachelor_xvigra_convolution LANGUAGES CXX)

# Force compiler to run in C++17 mode, with all warnings and optimization 3
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
message(STATUS "BLAS VENDOR:    " ${BLA_VENDOR})
message(STATUS "BLAS LIBRARIES: " ${BLAS_LIBRARIES})

find_package(OpenImageIO CONFIG REQUIRED)
find_package(xsimd CONFIG REQUIRED)
find_package(xtl CONFIG REQUIRED)
find_package(xtensor CONFIG REQUIRED)
find_package(xtensor-io CONFIG REQUIRED)
find_package(xtensor-blas CONFIG REQUIRED)

message(STATUS "xtensor_VERSION: " ${xtensor_VERSION})

configure_file("original.png" "original.png" COPYONLY)
configure_file("relief-filter_0x0_1x1_1x1.png" "relief-filter_0x0_1x1_1x1.png" COPYONLY)
configure_file("relief-filter_5x4_1x1_1x1.png" "relief-filter_5x4_1x1_1x1.png" COPYONLY)
configure_file("relief-filter_0x0_3x2_1x1.png" "relief-filter_0x0_3x2_1x1.png" COPYONLY)
configure_file("relief-filter_0x0_1x1_3x4.png" "relief-filter_0x0_1x1_3x4.png" COPYONLY)
configure_file("relief-filter_5x4_3x2_3x4.png" "relief-filter_5x4_3x2_3x4.png" COPYONLY)


add_subdirectory(src)

add_subdirectory(benchmarks)

enable_testing()
add_subdirectory(tests)