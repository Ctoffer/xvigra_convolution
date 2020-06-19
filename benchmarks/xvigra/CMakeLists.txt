set (TARGETS 
    benchmark_convolve1D_inputSize 
    benchmark_separableConvolve1D_inputSize
    benchmark_separableConvolve2D_inputSize
)

FOREACH(TARGET ${TARGETS})
    add_executable(xvigra_${TARGET} ${TARGET}.cpp)
    target_link_libraries(xvigra_${TARGET} benchmark::benchmark ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor Threads::Threads)
ENDFOREACH(TARGET)
