clear;
cd build-windows; 
cmake .. -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DLAPACK_lapack_WORKS=ON; 
cmake --build . --config Release;
cd ..;
#.\build-windows\tests\Release\test_image_io;
.\build-windows\tests\Release\test_explicit_convolution;
#.\build-windows\tests\Release\test_convolution_util;
