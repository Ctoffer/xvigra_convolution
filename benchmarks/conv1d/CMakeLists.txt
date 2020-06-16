set (TARGETS measure_access_time measure_channels measure_dilation measure_input_size measure_padding measure_stride)

FOREACH(TARGET ${TARGETS})
    add_executable(conv1d_${TARGET} ${TARGET}.cpp)
    target_link_libraries(conv1d_${TARGET} benchmark::benchmark ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor Threads::Threads)
ENDFOREACH(TARGET)
