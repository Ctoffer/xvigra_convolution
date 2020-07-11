set (TARGETS 
    benchmark_xstrided-view_copy_complete_X
)

FOREACH(TARGET ${TARGETS})
    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} benchmark::benchmark ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor Threads::Threads)
ENDFOREACH(TARGET)
