Repository containing the code for my Bachelor Thesis 'Redesign of VIGRA Based on Xtensor'

This project is build on CMake with the vcpkg toolchain.

Project structure:
  - benchmarks: C++ code of the performed benchmarks
  - benchmarks-result: Contains JSON files with the recorded benchmark data and rendered tables / graphs used in the thesis
  - include
    - raw: utility header to access flattened arrays as strided n-dimensional view for benchmarking
    - xvigra: production code of this work
    - xvigra_legacy: contains different variants of the convolution implementation which are used for the benchmarks
  - py_impl: The Python reference implementation of the explicit convolution
  - resources: Image files needed for demos and tests
  - scripts: Python scripts to generate test data, plot benchmark data and increase convience of the build process
  - src: Some small demos to better understand the xtensor / xvigra API
  - tests: Unit tests used to ensure the correct behaviour of the Explicit and Separable Convolution

  different build files for Windows and Linux with a high-level dual build based on the WSL.