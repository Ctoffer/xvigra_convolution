#include <limits>
#include <stdexcept>
#include <type_traits>

#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xtensor.hpp"

#include "xvigra/explicit_convolution.hpp"

TEST_CASE("Test calculateOutputSize") {
    constexpr int inputSize = 5;
    constexpr int kernelSize = 3;
    xvigra::KernelOptions options;

    SUBCASE("(2, 1, 1)") {
        options.setPadding(2);
        options.stride = 1;
        options.dilation = 1;

        SUBCASE("(Constant, Constant)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(0));
            CHECK(calculateOutputSize(inputSize, kernelSize, options) == 7);
        }
        
        SUBCASE("(Constant, Avoid)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(0), xvigra::BorderTreatment::avoid());
            CHECK(calculateOutputSize(inputSize, kernelSize, options) == 5);
        }

        SUBCASE("(Avoid, Constant)") {
            options.setBorderTreatment(xvigra::BorderTreatment::avoid(), xvigra::BorderTreatment::constant(0));
            CHECK(calculateOutputSize(inputSize, kernelSize, options) == 5);
        }

        SUBCASE("(Avoid, Avoid)") {
            options.setBorderTreatment(xvigra::BorderTreatment::avoid(), xvigra::BorderTreatment::avoid());
            CHECK(calculateOutputSize(inputSize, kernelSize, options) == 3);
        }
    }
}