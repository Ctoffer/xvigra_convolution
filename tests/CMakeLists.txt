set(TARGETS
    test_image_io
    test_explicit_convolution
    test_convolution_util
)

FOREACH(TARGET ${TARGETS})
    add_executable(${TARGET} main.cpp ${TARGET}.cpp)
    target_link_libraries(${TARGET} PRIVATE doctest::doctest Threads::Threads ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor OpenImageIO::OpenImageIO OpenImageIO::OpenImageIO_Util)
    add_test(NAME ${TARGET} COMMAND ${TARGET})
ENDFOREACH(TARGET)


