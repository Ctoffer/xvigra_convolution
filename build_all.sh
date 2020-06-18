clear
cd build-linux;

start_time=$(date +%s%3N)
cmake .. "-DCMAKE_TOOLCHAIN_FILE=/mnt/d/Subsystem_Ubuntu_20/vcpkg/scripts/buildsystems/vcpkg.cmake" -DLAPACK_lapack_WORKS=ON
end_time=$(date +%s%3N)
runtime=$((end_time-start_time))
printf 'Prepare-Time: %s ms\n\n\n' "$runtime"

start_time=$(date +%s%3N)
cmake --build . --config Release
end_time=$(date +%s%3N)
runtime=$((end_time-start_time))
printf 'Build-Time: %s ms\n\n\n' "$runtime"

cd ..

start_time=$(date +%s%3N)
#./build-linux/tests/test_convolution_util
#./build-linux/tests/test_explicit_convolution
#./build-linux/tests/test_image_io
./build-linux/tests/test_separable_convolution
end_time=$(date +%s%3N)
runtime=$((end_time-start_time))
printf 'Test-Time: %s ms\n\n\n' "$runtime"

