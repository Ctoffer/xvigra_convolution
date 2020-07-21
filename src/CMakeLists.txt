set(TARGETS
    demo_image_io
    demo_explicit_convolution
    demo_separable
    demo_xexpression
    demo_convolution
)

set (ALL_LIBS
    Threads::Threads
    ${BLAS_LIBRARIES}
    ${LAPACK_LIBRARIES} 
    xtensor
    xtensor-blas
    xtensor-io
    OpenImageIO::OpenImageIO
    OpenImageIO::OpenImageIO_Util
)

FOREACH(TARGET ${TARGETS})
    add_executable(${TARGET} ${TARGET}.cpp)
    target_link_libraries(${TARGET} PRIVATE  ${ALL_LIBS})
    add_test(NAME ${TARGET} COMMAND ${TARGET})
ENDFOREACH(TARGET)
