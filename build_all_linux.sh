if [ $# -eq 0 ]
  then
    clear
fi

cd build-linux

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
printf '────────────────────────────────────────────────────────────────────────────────\n'
printf '                                 Test ArrayView3D\n'
printf '────────────────────────────────────────────────────────────────────────────────\n'
./build-linux/tests/test_array_view_3d
printf '\n'

printf '────────────────────────────────────────────────────────────────────────────────\n'
printf '                                    Test Math\n'
printf '────────────────────────────────────────────────────────────────────────────────\n'
./build-linux/tests/test_math
printf '\n'

printf '────────────────────────────────────────────────────────────────────────────────\n'
printf '                                Test Kernel Util\n'
printf '────────────────────────────────────────────────────────────────────────────────\n'
./build-linux/tests/test_kernel_util
printf '\n'

printf '────────────────────────────────────────────────────────────────────────────────\n'
printf '                            Test Convolution Util\n'
printf '────────────────────────────────────────────────────────────────────────────────\n'
./build-linux/tests/test_convolution_util
printf '\n'

printf '────────────────────────────────────────────────────────────────────────────────\n'
printf '                                Test Image IO\n'
printf '────────────────────────────────────────────────────────────────────────────────\n'
./build-linux/tests/test_image_io
printf '\n'

printf '────────────────────────────────────────────────────────────────────────────────\n'
printf '                          Test Explicit Convolution\n'
printf '────────────────────────────────────────────────────────────────────────────────\n'
./build-linux/tests/test_explicit_convolution
printf '\n'

printf '────────────────────────────────────────────────────────────────────────────────\n'
printf '                         Test Separable Convolution\n'
printf '────────────────────────────────────────────────────────────────────────────────\n'
./build-linux/tests/test_separable_convolution
printf '\n'

end_time=$(date +%s%3N)
runtime=$((end_time-start_time))
printf 'Test-Time: %s ms\n\n\n' "$runtime"

