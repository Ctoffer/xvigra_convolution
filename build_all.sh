clear;
cd build-linux; 
cmake .. "-DCMAKE_TOOLCHAIN_FILE=/mnt/d/Subsystem_Ubuntu_20/vcpkg/scripts/buildsystems/vcpkg.cmake" -DLAPACK_lapack_WORKS=ON; 
cmake --build . --config Release;
cd ..;
#./build-linux/tests/test_convolution_util;
#./build-linux/tests/test_explicit_convolution;
#./build-linux/tests/test_image_io;
./build-linux/tests/test_separable_convolution;

