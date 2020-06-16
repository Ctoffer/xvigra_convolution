set (TARGETS measure_access_time measure_channels measure_dilation measure_input_size measure_padding measure_stride)

FOREACH(TARGET ${TARGETS})
    add_executable(conv2d_${TARGET} ${TARGET}.cpp)
    target_link_libraries(conv2d_${TARGET} PUBLIC xtensor PRIVATE benchmark::benchmark ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} Threads::Threads)
	# target_compile_options(conv2d_${TARGET} PRIVATE /arch:AVX2)
ENDFOREACH(TARGET)
