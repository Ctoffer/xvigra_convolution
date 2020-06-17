#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xarray.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xtensor.hpp"

#include "xvigra/convolution_util.hpp"
#include "xvigra/explicit_convolution.hpp"
#include "xvigra/separable_convolution.hpp"
#include "xvigra/image_io.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ constexpr - begin                                                                                                ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

constexpr double FLOAT_EPSILON = std::numeric_limits<float>::epsilon();

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ constexpr - end                                                                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ utility - begin                                                                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename T, typename O>
void checkExpressions(
    const xt::xexpression<T>& actualExpression, 
    const xt::xexpression<O>& expectedExpression,
    double epsilon = FLOAT_EPSILON
) {
    auto actual = actualExpression.derived_cast();
    auto expected = expectedExpression.derived_cast();

    CHECK(actual.dimension() == expected.dimension());
    
    std::vector<std::size_t> actualShape;
    for (const auto& value : actual.shape()) {
        actualShape.push_back(value);
    }

    std::vector<std::size_t> expectedShape;
    for (const auto& value : expected.shape()) {
        expectedShape.push_back(value);
    }
    CHECK(actualShape == expectedShape);
    
    auto iterActual = actual.begin();
    auto iterExpected = expected.begin();
    auto endExpected = expected.end();

    for (; iterExpected != endExpected; ++iterActual, ++iterExpected) {
        CHECK(*iterActual == doctest::Approx(*iterExpected).epsilon(epsilon));
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ utility - end                                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test separableConvolve1D - begin                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE("SeparableConvolve1D: Test Default Options", T, std::pair<int, double>) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = std::common_type_t<InputType, KernelType>;

    xvigra::KernelOptions options;

    // SUBCASE("Symmetric Kernel") {
    //     xt::xtensor<InputType, 2> input{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
    //     xt::xtensor<KernelType, 1> kernel{1.0f, 1.3f, 1.7f, 2.11f};
    //     xt::xtensor<KernelType, 3> fullKernel{{{1.0f, 1.3f, 1.7f, 2.11f}}};

    //     auto expected = xvigra::convolve1D(input, fullKernel, options);
    //     auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::LAST>(input, kernel, options);

    //     checkExpressions(expected, actual);
    // }

    SUBCASE("Asymmetric Kernel") {
        xt::xtensor<InputType, 2> input{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        xt::xtensor<KernelType, 1> kernel{1.0f, 1.3f, 1.7f};
        xt::xtensor<KernelType, 3> fullKernel{{{1.0f, 1.3f, 1.7f}}};

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::LAST>(input, kernel, options);

        checkExpressions(expected, actual);
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test separableConvolve1D - end                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝