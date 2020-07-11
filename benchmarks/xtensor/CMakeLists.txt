set (TARGETS 
    benchmark_strided-view_copy_complete_X
    benchmark_strided-view_copy_complete_Y
    benchmark_strided-view_copy_complete_Z
    benchmark_strided-view_copy_paddingStride_X
    benchmark_strided-view_copy_paddingStride_Y
    benchmark_strided-view_copy_paddingStride_Z
    benchmark_view_copy_complete_X
    benchmark_view_copy_complete_Y
    benchmark_view_copy_complete_Z
    benchmark_view_copy_paddingStride_X
    benchmark_view_copy_paddingStride_Y
    benchmark_view_copy_paddingStride_Z
)

FOREACH(TARGET ${TARGETS})
    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} benchmark::benchmark ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor Threads::Threads)
ENDFOREACH(TARGET)
