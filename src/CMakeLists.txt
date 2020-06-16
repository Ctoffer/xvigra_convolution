#add_executable(image_demo image_demo.cpp)
#target_link_libraries(image_demo ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor xtensor::optimize xtensor::use_xsimd xtensor-io OpenImageIO::OpenImageIO #OpenImageIO::OpenImageIO_Util Threads::Threads)

#add_executable(image_io_demo image_io_demo.cpp)
#target_link_libraries(image_io_demo PUBLIC xtensor xtensor-io PRIVATE Threads::Threads OpenImageIO::OpenImageIO OpenImageIO::OpenImageIO_Util)


#add_executable(explicit_convolution_demo explicit_convolution_demo.cpp)
#target_link_libraries(explicit_convolution_demo PUBLIC xtensor PRIVATE Threads::Threads ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable(demo_separable demo_separable.cpp)
target_link_libraries(demo_separable Threads::Threads ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor xtensor-blas xtensor-io OpenImageIO::OpenImageIO OpenImageIO::OpenImageIO_Util)

add_executable(demo_xexpression demo_xexpression.cpp)
target_link_libraries(demo_xexpression Threads::Threads ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} xtensor xtensor-blas xtensor-io OpenImageIO::OpenImageIO OpenImageIO::OpenImageIO_Util)