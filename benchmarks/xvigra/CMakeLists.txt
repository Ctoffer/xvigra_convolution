set (TARGETS 
    benchmark_convolve1D_inputSize_channelFirst
    benchmark_convolve1D_inputSize_channelLast
    benchmark_convolve2D_inputSize_channelFirst
    benchmark_convolve2D_inputSize_channelLast
    benchmark_separableConvolve1D_inputSize
    benchmark_separableConvolve2D_inputSize
)

FOREACH(TARGET ${TARGETS})
    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} benchmark::benchmark ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor Threads::Threads)
ENDFOREACH(TARGET)
