#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"

#include "xvigra/kernel_util.hpp"
#include "xvigra/math.hpp"
#include "xvigra/io_util.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - begin                                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

#define TYPES \
    std::uint16_t, \
    int, \
    float, \
    double


#define ZERO_1D_SYM_FLOAT    {0.0f, 0.0f, 0.0f, 0.0f}
#define ZERO_1D_SYM_INT      {   0,    0,    0,    0}
#define ZERO_1D_ASYM_FLOAT   {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
#define ZERO_1D_ASYM_INT     {   0,    0,    0,    0,    0}

#define KERNEL_1D_SYM_FLOAT  {1.0f, 1.3f, 1.7f, 2.1f}
#define KERNEL_1D_SYM_INT    {  10,   13,   17,   21}
#define KERNEL_1D_ASYM_FLOAT {1.0f, 1.3f, 1.7f, 2.1f, 2.3f}
#define KERNEL_1D_ASYM_INT   {  10,   13,   17,   21,   23}


#define ZERO_2D_SYM_FLOAT {\
    ZERO_1D_SYM_FLOAT,\
    ZERO_1D_SYM_FLOAT,\
    ZERO_1D_SYM_FLOAT,\
    ZERO_1D_SYM_FLOAT\
}
#define ZERO_2D_SYM_INT {\
    ZERO_1D_SYM_INT,\
    ZERO_1D_SYM_INT,\
    ZERO_1D_SYM_INT,\
    ZERO_1D_SYM_INT\
}
#define ZERO_2D_ASYM_FLOAT {\
    ZERO_1D_ASYM_FLOAT,\
    ZERO_1D_ASYM_FLOAT,\
    ZERO_1D_ASYM_FLOAT,\
    ZERO_1D_ASYM_FLOAT,\
    ZERO_1D_ASYM_FLOAT\
}
#define ZERO_2D_ASYM_INT {\
    ZERO_1D_ASYM_INT,\
    ZERO_1D_ASYM_INT,\
    ZERO_1D_ASYM_INT,\
    ZERO_1D_ASYM_INT,\
    ZERO_1D_ASYM_INT\
}


#define KERNEL_2D_SYM_FLOAT {\
    {1.00f, 1.30f, 1.70f, 2.10f},\
    {1.30f, 1.69f, 2.21f, 2.73f},\
    {1.70f, 2.21f, 2.89f, 3.57f},\
    {2.10f, 2.73f, 3.57f, 4.41f}\
}
#define KERNEL_2D_SYM_INT {\
    {100, 130, 170, 210},\
    {130, 169, 221, 273},\
    {170, 221, 289, 357},\
    {210, 273, 357, 441}\
}
#define KERNEL_2D_ASYM_FLOAT {\
    {1.00f, 1.30f, 1.70f, 2.10f, 2.30f},\
    {1.30f, 1.69f, 2.21f, 2.73f, 2.99f},\
    {1.70f, 2.21f, 2.89f, 3.57f, 3.91f},\
    {2.10f, 2.73f, 3.57f, 4.41f, 4.83f},\
    {2.30f, 2.99f, 3.91f, 4.83f, 5.29f}\
}
#define KERNEL_2D_ASYM_INT {\
    {100, 130, 170, 210, 230},\
    {130, 169, 221, 273, 299},\
    {170, 221, 289, 357, 391},\
    {210, 273, 357, 441, 483},\
    {230, 299, 391, 483, 529}\
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - end                                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test promoteKernelToFull1D - begin                                                                               ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE("Test promoteKernelToFull1D", KernelType, TYPES) {
    xt::xtensor<KernelType, 3> expected;

    SUBCASE("Raw Kernel 1D") {
        xt::xtensor<KernelType, 1> kernel;

        SUBCASE("Symmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = KERNEL_1D_SYM_FLOAT;
                expected = {
                    {KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT}, 
                    {ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT}, 
                    {ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT}
                };
            } else {
                kernel = KERNEL_1D_SYM_INT;
                expected = {
                    {KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, ZERO_1D_SYM_INT}, 
                    {ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT}, 
                    {ZERO_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT}
                };
            }

            auto actual = xvigra::promoteKernelToFull1D(kernel, 3);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());
            CHECK(actual == expected);
        }
        
        SUBCASE("Asymmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = KERNEL_1D_ASYM_FLOAT;
                expected = {
                    {KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT}, 
                    {ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT}, 
                    {ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT}
                };
            } else {
                kernel = KERNEL_1D_ASYM_INT;
                expected = {
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT}, 
                    {ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT}, 
                    {ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT}
                };
            }

            auto actual = xvigra::promoteKernelToFull1D(kernel, 3);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());
            CHECK(actual == expected);
        }
    }

    SUBCASE("Raw Kernel 2D") {
        xt::xtensor<KernelType, 2> kernel;
        
        SUBCASE("Symmetric Kernel") {
            std::size_t outputChannels = 0;

            if constexpr (std::is_floating_point<KernelType>::value) {
                outputChannels = 3;
                kernel = {KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT};
                expected = {
                    {KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT}, 
                    {KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT}, 
                    {KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT}
                };
            } else {
                outputChannels = 2;
                kernel = {KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, ZERO_1D_SYM_INT};
                expected = {
                    {KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, ZERO_1D_SYM_INT}, 
                    {KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, ZERO_1D_SYM_INT}, 
                };
            }

            auto actual = xvigra::promoteKernelToFull1D(kernel, outputChannels);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());
            CHECK(actual == expected);
        }
        
        SUBCASE("Asymmetric Kernel") {
            std::size_t outputChannels = 0;

            if constexpr (std::is_floating_point<KernelType>::value) {
                outputChannels = 3;
                kernel = {KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT};
                expected = {
                    {KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT}, 
                    {KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT}, 
                    {KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT}
                };
            } else {
                outputChannels = 6;
                kernel = {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT};
                expected = {
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT}, 
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT}, 
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT}, 
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT}, 
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT}, 
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT}, 
                };
            }

            auto actual = xvigra::promoteKernelToFull1D(kernel, outputChannels);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());
            CHECK(actual == expected);
        }
    }

    SUBCASE("Raw Kernel 3D") {
        xt::xtensor<KernelType, 3> kernel;
        
        SUBCASE("Symmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = {
                    {KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT},
                    {ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT},
                    {KERNEL_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT},
                    {ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT},
                };
                expected = {
                    {KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT},
                    {ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT},
                    {KERNEL_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT},
                    {ZERO_1D_SYM_FLOAT, ZERO_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT, KERNEL_1D_SYM_FLOAT},
                };
            } else {
                kernel = {
                    {KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT},
                    {ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT},
                    {KERNEL_1D_SYM_INT, KERNEL_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, ZERO_1D_SYM_INT},
                    {ZERO_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, KERNEL_1D_SYM_INT, KERNEL_1D_SYM_INT},
                };
                expected = {
                    {KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT},
                    {ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT},
                    {KERNEL_1D_SYM_INT, KERNEL_1D_SYM_INT, KERNEL_1D_SYM_INT, ZERO_1D_SYM_INT, ZERO_1D_SYM_INT},
                    {ZERO_1D_SYM_INT, ZERO_1D_SYM_INT, KERNEL_1D_SYM_INT, KERNEL_1D_SYM_INT, KERNEL_1D_SYM_INT},
                };
            }

            auto actual = xvigra::promoteKernelToFull1D(kernel);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());
            CHECK(actual == expected);
        }
        
        SUBCASE("Asymmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = {
                    {KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT},
                    {ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT},
                    {KERNEL_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT},
                    {ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT},
                };
                expected = {
                    {KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT},
                    {ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT},
                    {KERNEL_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT},
                    {ZERO_1D_ASYM_FLOAT, ZERO_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT, KERNEL_1D_ASYM_FLOAT},
                };
            } else {
                kernel = {
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT},
                    {ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT},
                    {KERNEL_1D_ASYM_INT, KERNEL_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT},
                    {ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, KERNEL_1D_ASYM_INT, KERNEL_1D_ASYM_INT},
                };
                expected = {
                    {KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT},
                    {ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT},
                    {KERNEL_1D_ASYM_INT, KERNEL_1D_ASYM_INT, KERNEL_1D_ASYM_INT, ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT},
                    {ZERO_1D_ASYM_INT, ZERO_1D_ASYM_INT, KERNEL_1D_ASYM_INT, KERNEL_1D_ASYM_INT, KERNEL_1D_ASYM_INT},
                };
            }

            auto actual = xvigra::promoteKernelToFull1D(kernel);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());
            CHECK(actual == expected);
        }
    }
}   

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test promoteKernelToFull1D - end                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test promoteKernelToFull1D - begin                                                                               ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


TEST_CASE_TEMPLATE("Test promoteKernelToFull2D", KernelType, TYPES) {
    xt::xtensor<KernelType, 4> expected;

    SUBCASE("Raw Kernel 1D") {
        xt::xtensor<KernelType, 1> kernel;

        SUBCASE("Symmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = KERNEL_1D_SYM_FLOAT;
                expected = {
                    {KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT}, 
                    {ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT}, 
                    {ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT}
                };
            } else {
                kernel = KERNEL_1D_SYM_INT;
                expected = {
                    {KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, ZERO_2D_SYM_INT}, 
                    {ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT}, 
                    {ZERO_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT}
                };
            }

            auto actual = xvigra::promoteKernelToFull2D(kernel, 3);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());
            
            if constexpr (std::is_floating_point<KernelType>::value) {
                CHECK(xvigra::roundExpression(actual, 2) == xvigra::roundExpression(expected, 2));
            } else {
                CHECK(actual == expected);
            }
            
        }
        
        SUBCASE("Asymmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = KERNEL_1D_ASYM_FLOAT;
                expected = {
                    {KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT}, 
                    {ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT}, 
                    {ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT}
                };
            } else {
                kernel = KERNEL_1D_ASYM_INT;
                expected = {
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT}
                };
            }

            auto actual = xvigra::promoteKernelToFull2D(kernel, 3);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());

            if constexpr (std::is_floating_point<KernelType>::value) {
                CHECK(xvigra::roundExpression(actual, 2) == xvigra::roundExpression(expected, 2));
            } else {
                CHECK(actual == expected);
            }
        }
    }

    SUBCASE("Raw Kernel 2D") {
        xt::xtensor<KernelType, 2> kernel;

        SUBCASE("Symmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = KERNEL_2D_SYM_FLOAT;
                expected = {
                    {KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT}, 
                    {ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT}, 
                    {ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT}
                };
            } else {
                kernel = KERNEL_2D_SYM_INT;
                expected = {
                    {KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, ZERO_2D_SYM_INT}, 
                    {ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT}, 
                    {ZERO_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT}
                };
            }

            auto actual = xvigra::promoteKernelToFull2D(kernel, 3);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());

            if constexpr (std::is_floating_point<KernelType>::value) {
                CHECK(xvigra::roundExpression(actual, 2) == xvigra::roundExpression(expected, 2));
            } else {
                CHECK(actual == expected);
            }
        }
        
        SUBCASE("Asymmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = KERNEL_2D_ASYM_FLOAT;
                expected = {
                    {KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT}, 
                    {ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT}, 
                    {ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT}
                };
            } else {
                kernel = KERNEL_2D_ASYM_INT;
                expected = {
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT}
                };
            }

            auto actual = xvigra::promoteKernelToFull2D(kernel, 3);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());

            if constexpr (std::is_floating_point<KernelType>::value) {
                CHECK(xvigra::roundExpression(actual, 2) == xvigra::roundExpression(expected, 2));
            } else {
                CHECK(actual == expected);
            }
        }
    }

    SUBCASE("Raw Kernel 3D") {
        xt::xtensor<KernelType, 3> kernel;
        
        SUBCASE("Symmetric Kernel") {
            std::size_t outputChannels = 0;

            if constexpr (std::is_floating_point<KernelType>::value) {
                outputChannels = 3;
                kernel = {KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT};
                expected = {
                    {KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT}, 
                    {KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT}, 
                    {KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT}
                };
            } else {
                outputChannels = 2;
                kernel = {KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, ZERO_2D_SYM_INT};
                expected = {
                    {KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, ZERO_2D_SYM_INT}, 
                    {KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, ZERO_2D_SYM_INT}, 
                };
            }

            auto actual = xvigra::promoteKernelToFull2D(kernel, outputChannels);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());

            if constexpr (std::is_floating_point<KernelType>::value) {
                CHECK(xvigra::roundExpression(actual, 2) == xvigra::roundExpression(expected, 2));
            } else {
                CHECK(actual == expected);
            }
        }
        
        SUBCASE("Asymmetric Kernel") {
            std::size_t outputChannels = 0;

            if constexpr (std::is_floating_point<KernelType>::value) {
                outputChannels = 3;
                kernel = {KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT};
                expected = {
                    {KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT}, 
                    {KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT}, 
                    {KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT}
                };
            } else {
                outputChannels = 6;
                kernel = {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT};
                expected = {
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT}, 
                };
            }

            auto actual = xvigra::promoteKernelToFull2D(kernel, outputChannels);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());

            if constexpr (std::is_floating_point<KernelType>::value) {
                CHECK(xvigra::roundExpression(actual, 2) == xvigra::roundExpression(expected, 2));
            } else {
                CHECK(actual == expected);
            }
        }
    }

    SUBCASE("Raw Kernel 4D") {
        xt::xtensor<KernelType, 4> kernel;
        
        SUBCASE("Symmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = {
                    {KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT},
                    {ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT},
                    {KERNEL_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT},
                    {ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT},
                };
                expected = {
                    {KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT},
                    {ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT},
                    {KERNEL_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT},
                    {ZERO_2D_SYM_FLOAT, ZERO_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT, KERNEL_2D_SYM_FLOAT},
                };
            } else {
                kernel = {
                    {KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT},
                    {ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT},
                    {KERNEL_2D_SYM_INT, KERNEL_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, ZERO_2D_SYM_INT},
                    {ZERO_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, KERNEL_2D_SYM_INT, KERNEL_2D_SYM_INT},
                };
                expected = {
                    {KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT},
                    {ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT},
                    {KERNEL_2D_SYM_INT, KERNEL_2D_SYM_INT, KERNEL_2D_SYM_INT, ZERO_2D_SYM_INT, ZERO_2D_SYM_INT},
                    {ZERO_2D_SYM_INT, ZERO_2D_SYM_INT, KERNEL_2D_SYM_INT, KERNEL_2D_SYM_INT, KERNEL_2D_SYM_INT},
                };
            }

            auto actual = xvigra::promoteKernelToFull2D(kernel);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());

            if constexpr (std::is_floating_point<KernelType>::value) {
                CHECK(xvigra::roundExpression(actual, 2) == xvigra::roundExpression(expected, 2));
            } else {
                CHECK(actual == expected);
            }
        }
        
        SUBCASE("Asymmetric Kernel") {
            if constexpr (std::is_floating_point<KernelType>::value) {
                kernel = {
                    {KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT},
                    {ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT},
                    {KERNEL_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT},
                    {ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT},
                };
                expected = {
                    {KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT},
                    {ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT},
                    {KERNEL_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT},
                    {ZERO_2D_ASYM_FLOAT, ZERO_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT, KERNEL_2D_ASYM_FLOAT},
                };
            } else {
                kernel = {
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT},
                    {ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT},
                    {KERNEL_2D_ASYM_INT, KERNEL_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT},
                    {ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, KERNEL_2D_ASYM_INT, KERNEL_2D_ASYM_INT},
                };
                expected = {
                    {KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT},
                    {ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT},
                    {KERNEL_2D_ASYM_INT, KERNEL_2D_ASYM_INT, KERNEL_2D_ASYM_INT, ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT},
                    {ZERO_2D_ASYM_INT, ZERO_2D_ASYM_INT, KERNEL_2D_ASYM_INT, KERNEL_2D_ASYM_INT, KERNEL_2D_ASYM_INT},
                };
            }

            auto actual = xvigra::promoteKernelToFull2D(kernel);
            REQUIRE(actual.dimension() == expected.dimension());
            REQUIRE(actual.shape() == expected.shape());

            if constexpr (std::is_floating_point<KernelType>::value) {
                CHECK(xvigra::roundExpression(actual, 2) == xvigra::roundExpression(expected, 2));
            } else {
                CHECK(actual == expected);
            }
        }
    }
}   


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test promoteKernelToFull2D - end                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝