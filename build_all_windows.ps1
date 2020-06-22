#clear;
cd build-windows; 

$start_time = [Math]::Round((Get-Date).ToFileTime()/10000);
cmake .. -DCMAKE_TOOLCHAIN_FILE="D:/vcpkg/scripts/buildsystems/vcpkg.cmake" -DVCPKG_TARGET_TRIPLET=x64-windows -DLAPACK_lapack_WORKS=ON; 
$end_time = [Math]::Round((Get-Date).ToFileTime()/10000);
$runtime = $end_time - $start_time;
"Prepare-Time: {0} ms`n`n" -f $runtime;

$start_time = [Math]::Round((Get-Date).ToFileTime()/10000);
cmake --build . --config Release;
$end_time = [Math]::Round((Get-Date).ToFileTime()/10000);
$runtime = $end_time - $start_time;
"Build-Time: {0} ms`n`n" -f $runtime;

cd ..;

$start_time = [Math]::Round((Get-Date).ToFileTime()/10000);
"--------------------------------------------------------------------------------"
"                                  Test Image IO"
"--------------------------------------------------------------------------------"
.\build-windows\tests\Release\test_image_io.exe;
"`n"

"--------------------------------------------------------------------------------"
"                             Test Convolution Util"
"--------------------------------------------------------------------------------"
.\build-windows\tests\Release\test_convolution_util.exe;
"`n"

"--------------------------------------------------------------------------------"
"                          Test Explicit Convolution"
"--------------------------------------------------------------------------------"
.\build-windows\tests\Release\test_explicit_convolution.exe;
"`n"

"--------------------------------------------------------------------------------"
"                         Test Separable Convolution"
"--------------------------------------------------------------------------------"
.\build-windows\tests\Release\test_separable_convolution.exe;
"`n"

$end_time = [Math]::Round((Get-Date).ToFileTime()/10000);
$runtime = $end_time - $start_time;
"Test-Time: {0} ms`n`n" -f $runtime;
