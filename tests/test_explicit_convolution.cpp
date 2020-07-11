#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xtensor.hpp"

#include "xvigra/image_io.hpp"
#include "xvigra/explicit_convolution.hpp"
#include "xvigra/convolution_util.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - begin                                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

#define TYPE_PAIRS              \
    std::pair<short, float>,    \
    std::pair<short, double>,   \
    std::pair<int, float>,      \
    std::pair<int, double>

#define FLOATING_TYPE_PAIRS     \
    std::pair<double, float>,   \
    std::pair<double, double>

TYPE_TO_STRING(std::pair<short, float>);
TYPE_TO_STRING(std::pair<short, double>);
TYPE_TO_STRING(std::pair<int, float>);
TYPE_TO_STRING(std::pair<int, double>);
TYPE_TO_STRING(std::pair<double, float>);
TYPE_TO_STRING(std::pair<double, double>);

#define EXPECTED_UNPADDED_RESULT                     \
    8.7f, 12.7f, 16.7f, 20.7f, 24.7f, 28.7f, 32.7f

#define ZERO_KERNEL {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}
#define EDGE_KERNEL {{-1.0f, -1.0f, -1.0f}, {-1.0f,  9.0f, -1.0f}, {-1.0f, -1.0f, -1.0f}}
#define IDENTITY_KERNEL {{0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - end                                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ constexpr - begin                                                                                                ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

constexpr double FLOAT_EPSILON = std::numeric_limits<float>::epsilon();

constexpr auto TEST_IMAGE_PBM = "./resources/tests/Piercing-The-Ocean.pbm";
constexpr auto TEST_IMAGE_PGM = "./resources/tests/Piercing-The-Ocean.pgm";
constexpr auto TEST_IMAGE_PPM = "./resources/tests/Piercing-The-Ocean.ppm";
constexpr auto TEST_IMAGE_PNG = "./resources/tests/Piercing-The-Ocean.png";

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

    CHECK_EQ(actual.dimension(), expected.dimension());
    
    std::vector<std::size_t> actualShape;
    for (const auto& value : actual.shape()) {
        actualShape.push_back(value);
    }

    std::vector<std::size_t> expectedShape;
    for (const auto& value : expected.shape()) {
        expectedShape.push_back(value);
    }
    CHECK_EQ(actualShape, expectedShape);
    
    auto iterActual = actual.begin();
    auto iterExpected = expected.begin();
    auto endExpected = expected.end();

    for (; iterExpected != endExpected; ++iterActual, ++iterExpected) {
        CHECK_EQ(*iterActual, doctest::Approx(*iterExpected).epsilon(epsilon));
    }
}


template <typename I, typename K, typename R>
void checkConvolution1D(
    const xt::xexpression<I>& input,
    const xt::xexpression<K>& kernel,
    xvigra::KernelOptions& options,
    const xt::xexpression<R>& expectedExpression,
    double epsilon = FLOAT_EPSILON
) {
    auto expected = expectedExpression.derived_cast();
    auto actual = xvigra::convolve1D(input, kernel, options);

    checkExpressions(actual, expected, epsilon);
}


template <typename I, typename K, typename R>
void checkConvolution1DImplicit(
    const xt::xexpression<I>& inputExpression,
    const xt::xexpression<K>& kernelExpression,
    xvigra::KernelOptions& options,
    const xt::xexpression<R>& expectedExpression,
    double epsilon = FLOAT_EPSILON
) {
    auto input = inputExpression.derived_cast();
    auto kernel = kernelExpression.derived_cast();
    auto expected = expectedExpression.derived_cast();

    auto actual = xvigra::convolve1DImplicit(input, kernel, options);

    checkExpressions(actual, expected, epsilon);
}


template <typename I, typename K, typename R>
void checkConvolution2D(
    const xt::xexpression<I>& inputExpression,
    const xt::xexpression<K>& kernelExpression,
    const xvigra::KernelOptions2D& options,
    const xt::xexpression<R>& expectedExpression,
    double epsilon = FLOAT_EPSILON
) {
    auto input = inputExpression.derived_cast();
    auto kernel = kernelExpression.derived_cast();
    auto expected = expectedExpression.derived_cast();

    auto actual = xvigra::convolve2D(input, kernel, options);

    checkExpressions(actual, expected, epsilon);
}


template <typename I, typename K, typename R>
void checkConvolution2DImplicit(
    const xt::xexpression<I>& inputExpression,
    const xt::xexpression<K>& kernelExpression,
    xvigra::KernelOptions2D& options,
    const xt::xexpression<R>& expectedExpression,
    double epsilon = FLOAT_EPSILON
) {
    auto input = inputExpression.derived_cast();
    auto kernel = kernelExpression.derived_cast();
    auto expected = expectedExpression.derived_cast();

    auto actual = xvigra::convolve2DImplicit(input, kernel, options);

    checkExpressions(actual, expected, epsilon);
}


template <typename InputType, typename KernelType, typename ResultType>
void checkConvolution2DWithImage(
    const std::string& pathOfSource, 
    const xt::xtensor<KernelType, 4>& kernel,
    const xvigra::KernelOptions2D& options,
    const std::string& pathOfExpected 
) {
    xt::xtensor<InputType, 3> sourceImage = xvigra::loadImageAsXTensor<InputType>(pathOfSource);
    xt::xtensor<ResultType, 3> actualImage = xvigra::convolve2D(sourceImage, kernel, options);
    xt::xtensor<ResultType, 3> expectedImage = xvigra::loadImageAsXTensor<ResultType>(pathOfExpected);

   
    actualImage = xvigra::normalizeAfterConvolution<ResultType>(actualImage);
    // simulate saving from and reading from image file
    actualImage = xt::cast<ResultType>(xt::cast<double>(xt::cast<uint8_t>(actualImage * 255)) / 255.0);

    CHECK_EQ(actualImage.shape(), expectedImage.shape());
    CHECK_EQ(actualImage, expectedImage);
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ utility - end                                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test getBorderIndex - begin                                                                                      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// FIXME Test macros!
// TEST_CASE("Test getBorderIndex") {
//     constexpr int size = 5;

//     SUBCASE("Begin") {
//         constexpr bool isBegin = true;

//         SUBCASE("BorderTreatment::constant(0)") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(0), 0, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(0), -1, size), -1);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(0), -2, size), -1);
//         }

//         SUBCASE("BorderTreatment::constant(2)") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(2), 0, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(2), -1, size), -1);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(2), -2, size), -1);
//         }

//         SUBCASE("BorderTreatment::asymmetricReflect()") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::asymmetricReflect(), 0, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::asymmetricReflect(), -1, size), 1);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::asymmetricReflect(), -2, size), 2);
//         }

//         SUBCASE("BorderTreatment::avoid()") {
//            CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::avoid(), 0, size), 0);
//            CHECK_THROWS_WITH_AS(
//                xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::avoid(), -1, size),
//                "getBorderIndex(): Border treatment AVOID should not be used here!",
//                std::domain_error
//            );
//            CHECK_THROWS_WITH_AS(
//                xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::avoid(), -2, size),
//                "getBorderIndex(): Border treatment AVOID should not be used here!",
//                std::domain_error
//            );
//         }

//         SUBCASE("BorderTreatment::repeat()") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::repeat(), 0, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::repeat(), -1, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::repeat(), -2, size), 0);
//         }  

//         SUBCASE("BorderTreatment::symmetricReflect()") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::symmetricReflect(), 0, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::symmetricReflect(), -1, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::symmetricReflect(), -2, size), 1);
//         }

//         SUBCASE("BorderTreatment::wrap()") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::wrap(), 0, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::wrap(), -1, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::wrap(), -2, size), 3);
//         }
//     }

//     SUBCASE("End") {
//         constexpr bool isBegin = false;

//         SUBCASE("BorderTreatment::constant(0)") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(0), 4, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(0), 5, size), -1);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(0), 6, size), -1);
//         }

//         SUBCASE("BorderTreatment::constant(2)") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(2), 4, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(2), 5, size), -1);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::constant(2), 6, size), -1);
//         }

//         SUBCASE("BorderTreatment::asymmetricReflect()") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::asymmetricReflect(), 4, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::asymmetricReflect(), 5, size), 3);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::asymmetricReflect(), 6, size), 2);
//         }

//         SUBCASE("BorderTreatment::avoid()") {
//            CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::avoid(), 4, size), 4);
//            CHECK_THROWS_WITH_AS(
//                xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::avoid(), 5, size),
//                "getBorderIndex(): Border treatment AVOID should not be used here!",
//                std::domain_error
//            );
//            CHECK_THROWS_WITH_AS(
//                xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::avoid(), 6, size),
//                "getBorderIndex(): Border treatment AVOID should not be used here!",
//                std::domain_error
//            );
//         }

//         SUBCASE("BorderTreatment::repeat()") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::repeat(), 4, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::repeat(), 5, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::repeat(), 6, size), 4);
//         }  

//         SUBCASE("BorderTreatment::symmetricReflect()") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::symmetricReflect(), 4, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::symmetricReflect(), 5, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::symmetricReflect(), 6, size), 3);
//         }

//         SUBCASE("BorderTreatment::wrap()") {
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::wrap(), 4, size), 4);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::wrap(), 5, size), 0);
//             CHECK_EQ(xvigra::getBorderIndex<isBegin>(xvigra::BorderTreatment::wrap(), 6, size), 1);
//         }
//     }
// }

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test getBorderIndex - end                                                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test convolve1D - begin                                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE("Convolve1D: Test Default Options", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = std::common_type_t<InputType, KernelType>;

    xvigra::KernelOptions options;

    SUBCASE("Symmetric Kernel") {
        xt::xtensor<InputType, 2> input{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        xt::xtensor<KernelType, 3> kernel{{{1.0f, 1.3f, 1.7f, 2.11f}}};
        xt::xtensor<ResultType, 2> expected{{17.14f}, {23.25f}, {29.36f}, {35.47f}, {41.58f}, {47.69f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("Asymmetric Kernel") {
        xt::xtensor<InputType, 2> input{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        xt::xtensor<KernelType, 3> kernel{{{1.0f, 1.3f, 1.7f}}};
        xt::xtensor<ResultType, 2> expected{{8.7f}, {12.7f}, {16.7f}, {20.7f}, {24.7f}, {28.7f}, {32.7f}};

        checkConvolution1D(input, kernel, options, expected);
    }
}


TEST_CASE_TEMPLATE("Convolve1D: Test Channel Position", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    xt::xtensor<KernelType, 3> kernel{{{1.0f, 1.3f, 1.7f}}};
    xvigra::KernelOptions options;

    SUBCASE("Channel First") {
        options.channelPosition = xvigra::ChannelPosition::FIRST;   
        xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        xt::xtensor<ResultType, 2> expected{{8.7f, 12.7f, 16.7f, 20.7f, 24.7f, 28.7f, 32.7f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("Channel Last") {
        options.channelPosition = xvigra::ChannelPosition::LAST;
        xt::xtensor<InputType, 2> input{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        xt::xtensor<ResultType, 2> expected{{8.7f}, {12.7f}, {16.7f}, {20.7f}, {24.7f}, {28.7f}, {32.7f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("Channel Implicit") {
        options.channelPosition = xvigra::ChannelPosition::IMPLICIT;
        xt::xtensor<InputType, 1> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
        xt::xtensor<ResultType, 1> expected{8.7f, 12.7f, 16.7f, 20.7f, 24.7f, 28.7f, 32.7f};

        checkConvolution1DImplicit(input, kernel, options, expected);
    }
}


TEST_CASE_TEMPLATE("Convolve1D: Test Border Treatment", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    xt::xtensor<KernelType, 3> kernel{{{1.0f, 1.3f, 1.7f}}};

    xvigra::KernelOptions options;
    options.channelPosition = xvigra::ChannelPosition::FIRST;
    options.setPadding(1);
    

    SUBCASE("BorderTreatment::constant(0)") {
        xt::xtensor<ResultType, 2> expected{{4.7f, EXPECTED_UNPADDED_RESULT, 19.7f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::constant(2)") {
        options.setBorderTreatment(xvigra::BorderTreatment::constant(2));
        xt::xtensor<ResultType, 2> expected{{6.7f, EXPECTED_UNPADDED_RESULT, 23.1f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::asymmetricReflect()") {
        options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());
        xt::xtensor<ResultType, 2> expected{{6.7f, EXPECTED_UNPADDED_RESULT, 33.3f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::avoid()") {
        options.setBorderTreatment(xvigra::BorderTreatment::avoid());
        xt::xtensor<ResultType, 2> expected{{EXPECTED_UNPADDED_RESULT}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::repeat()") {
        options.setBorderTreatment(xvigra::BorderTreatment::repeat());
        xt::xtensor<ResultType, 2> expected{{5.7f, EXPECTED_UNPADDED_RESULT, 35.0f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::symmetricReflect()") {
        options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());
        xt::xtensor<ResultType, 2> expected{{5.7f, EXPECTED_UNPADDED_RESULT, 35.0f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::wrap()") {
        options.setBorderTreatment(xvigra::BorderTreatment::wrap());
        xt::xtensor<ResultType, 2> expected{{13.7f, EXPECTED_UNPADDED_RESULT, 21.4f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::avoid() & BorderTreatment::asymmetricReflect()") {
        options.setBorderTreatmentBegin(xvigra::BorderTreatment::avoid());
        options.setBorderTreatmentEnd(xvigra::BorderTreatment::asymmetricReflect());
        xt::xtensor<ResultType, 2> expected{{EXPECTED_UNPADDED_RESULT, 33.3f}};

        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::asymmetricReflect() & BorderTreatment::avoid()") {
        options.setBorderTreatmentBegin(xvigra::BorderTreatment::asymmetricReflect());
        options.setBorderTreatmentEnd(xvigra::BorderTreatment::avoid());
        xt::xtensor<ResultType, 2> expected{{6.7f, EXPECTED_UNPADDED_RESULT}};

        checkConvolution1D(input, kernel, options, expected);
    }
}


TEST_CASE_TEMPLATE("Convolve1D: Test Different Padding, Stride, Dilation", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    xt::xtensor<KernelType, 3> kernel{{{1.0f, 1.3f, 1.7f}}};

    xvigra::KernelOptions options;
    options.channelPosition = xvigra::ChannelPosition::FIRST;

    SUBCASE("Padding=1, Stride=1, Dilation=1") {
        options.setPadding(1);
        options.stride = 1;
        options.dilation = 1;

        xt::xtensor<ResultType, 2> expected{{4.7f, 8.7f, 12.7f, 16.7f, 20.7f, 24.7f, 28.7f, 32.7f, 19.7f}};
        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("Padding=0, Stride=2, Dilation=1") {
        options.setPadding(0);
        options.stride = 2;
        options.dilation = 1;

        xt::xtensor<ResultType, 2> expected{{8.7f, 16.7f, 24.7f, 32.7f}};
        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("Padding=0, Stride=1, Dilation=2") {
        options.setPadding(0);
        options.stride = 1;
        options.dilation = 2;

        xt::xtensor<ResultType, 2> expected{{13.4f, 17.4f, 21.4f, 25.4f, 29.4f}};
        checkConvolution1D(input, kernel, options, expected);
    }

    SUBCASE("Padding=1, Stride=2, Dilation=2") {
        options.setPadding(1);
        options.stride = 2;
        options.dilation = 2;

        xt::xtensor<ResultType, 2> expected{{9.4f, 17.4f, 25.4f, 16.4f}};
        checkConvolution1D(input, kernel, options, expected);
    }
}


TEST_CASE_TEMPLATE("Convolve1D: Test Invalid Configurations", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    SUBCASE("Input wrong dimension") {
        xt::xarray<float> inputTooSmall(std::vector<std::size_t>{7});
        xt::xarray<float> inputTooBig(std::vector<std::size_t>{7, 5, 3});

        xt::xarray<float> kernel(std::vector<std::size_t>{1, 1, 3});

        xvigra::KernelOptions options;
        options.channelPosition = xvigra::ChannelPosition::LAST;

        REQUIRE_EQ(inputTooSmall.dimension(), 1);
        REQUIRE_EQ(inputTooBig.dimension(), 3);
        REQUIRE_EQ(kernel.dimension(), 3);

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1D(inputTooSmall, kernel, options),
            "convolve1D(): Need 2 dimensional (W x C or C x W) input!",
            std::invalid_argument
        );

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1D(inputTooBig, kernel, options),
            "convolve1D(): Need 2 dimensional (W x C or C x W) input!",
            std::invalid_argument
        );
    }

    SUBCASE("Implicit: Input wrong dimension") {
        xt::xarray<float> inputTooSmall(std::vector<std::size_t>{});
        xt::xarray<float> inputTooBig(std::vector<std::size_t>{7, 3});

        xt::xarray<float> kernel(std::vector<std::size_t>{1, 1, 3});

        xvigra::KernelOptions options;
        options.channelPosition = xvigra::ChannelPosition::IMPLICIT;

        REQUIRE_EQ(inputTooSmall.dimension(), 0);
        REQUIRE_EQ(inputTooBig.dimension(), 2);
        REQUIRE_EQ(kernel.dimension(), 3);

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1DImplicit(inputTooSmall, kernel, options),
            "convolve1DImplicit(): Need 1 dimensional (W) input!",
            std::invalid_argument
        );

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1DImplicit(inputTooBig, kernel, options),
            "convolve1DImplicit(): Need 1 dimensional (W) input!",
            std::invalid_argument
        );
    }

    SUBCASE("Implicit Input, Explicit Option") {
        xt::xarray<InputType> input(std::vector<std::size_t>{9});
        xt::xarray<KernelType> kernel(std::vector<std::size_t>{1, 1, 3});

        xvigra::KernelOptions options;
        options.channelPosition = xvigra::ChannelPosition::FIRST;

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1DImplicit(input, kernel, options),
            "convolve1DImplicit(): Expected implicit channels in options!",
            std::domain_error
        );
    }

    SUBCASE("Explicit Input, Implicit Option") {
        xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        xt::xtensor<KernelType, 3> kernel{{{1.0f, 1.3f, 1.7f}}};

        xvigra::KernelOptions options;
        options.channelPosition = xvigra::ChannelPosition::IMPLICIT;

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1D(input, kernel, options),
            "convolve1D(): Implicit channel option is not supported for explicit channels in input!",
            std::invalid_argument
        );
    }

    SUBCASE("Input Channel Mismatch In Input And Kernel") {
        xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5}, {6, 7, 8, 9}};
        xt::xtensor<KernelType, 3> kernel{{{1.0f, 1.3f, 1.7f}}};

        xvigra::KernelOptions options;
        options.channelPosition = xvigra::ChannelPosition::FIRST;

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1D(input, kernel, options),
            "convolve1D(): Input channels of input and kernel do not align!",
            std::invalid_argument
        );
    }

    SUBCASE("Kernel Too Big") {
        xt::xtensor<InputType, 2> input{{1, 2, 3}};
        xt::xtensor<KernelType, 3> kernel{{{1.0f, 1.3f, 1.7f, 2.11f}}};

        xvigra::KernelOptions options;
        options.channelPosition = xvigra::ChannelPosition::FIRST;

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1D(input, kernel, options),
            "convolve1D(): Kernel width is greater than padded input width!",
            std::invalid_argument
        );
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test convolve1D - end                                                                                            ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test convolve2D - begin                                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE("Convolve2D: Test Default Options", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = std::common_type_t<InputType, KernelType>;

    xvigra::KernelOptions2D options;

    SUBCASE("Symmetric Kernel") {
        xt::xtensor<InputType, 3> input{
            {{ 1}, { 2}, { 3}, { 4}, { 5}}, 
            {{ 6}, { 7}, { 8}, { 9}, {10}}, 
            {{11}, {12}, {13}, {14}, {15}}, 
            {{16}, {17}, {18}, {19}, {20}}, 
            {{21}, {22}, {23}, {24}, {25}}
        };
        xt::xtensor<KernelType, 4> kernel{{{
            {1.00f, 1.30f, 1.70f, 2.10f},
            {1.30f, 1.69f, 2.21f, 2.73f},
            {1.70f, 2.21f, 2.89f, 3.57f},
            {2.10f, 2.73f, 3.57f, 4.41f}
        }}};
        xt::xtensor<ResultType, 3> expected{
            {{439.81f}, {477.02f}}, 
            {{625.86f}, {663.07f}}, 
        };

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("Asymmetric Kernel") {
        xt::xtensor<InputType, 3> input{
            {{ 1}, { 2}, { 3}, { 4}, { 5}}, 
            {{ 6}, { 7}, { 8}, { 9}, {10}}, 
            {{11}, {12}, {13}, {14}, {15}}, 
            {{16}, {17}, {18}, {19}, {20}}, 
            {{21}, {22}, {23}, {24}, {25}}
        };
        xt::xtensor<KernelType, 4> kernel{{{
            {1.00f, 1.30f, 1.70f},
            {1.30f, 1.69f, 2.21f},
            {1.70f, 2.21f, 2.89f}
        }}};
        xt::xtensor<ResultType, 3> expected{
            {{128.8f}, {144.8f}, {160.8f}}, 
            {{208.8f}, {224.8f}, {240.8f}}, 
            {{288.8f}, {304.8f}, {320.8f}}
        };

        checkConvolution2D(input, kernel, options, expected);
    }
}


TEST_CASE_TEMPLATE("Convolve2D: Test Channel Position", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    xt::xtensor<KernelType, 4> kernel{{{
            {1.00f, 1.30f, 1.70f},
            {1.30f, 1.69f, 2.21f},
            {1.70f, 2.21f, 2.89f}
    }}};
    xvigra::KernelOptions2D options;

    SUBCASE("Channel First") {
        options.setChannelPosition(xvigra::ChannelPosition::FIRST);   
        xt::xtensor<InputType, 3> input{{
            { 1,  2,  3,  4,  5}, 
            { 6,  7,  8,  9, 10}, 
            {11, 12, 13, 14, 15}, 
            {16, 17, 18, 19, 20}, 
            {21, 22, 23, 24, 25}
        }};
        xt::xtensor<ResultType, 3> expected{{
            {128.8f, 144.8f, 160.8f}, 
            {208.8f, 224.8f, 240.8f}, 
            {288.8f, 304.8f, 320.8f}
        }};

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("Channel Last") {
        options.setChannelPosition(xvigra::ChannelPosition::LAST);
        xt::xtensor<InputType, 3> input{
            {{ 1}, { 2}, { 3}, { 4}, { 5}}, 
            {{ 6}, { 7}, { 8}, { 9}, {10}}, 
            {{11}, {12}, {13}, {14}, {15}}, 
            {{16}, {17}, {18}, {19}, {20}}, 
            {{21}, {22}, {23}, {24}, {25}}
        };
        xt::xtensor<ResultType, 3> expected{
            {{128.8f}, {144.8f}, {160.8f}}, 
            {{208.8f}, {224.8f}, {240.8f}}, 
            {{288.8f}, {304.8f}, {320.8f}}
        };

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("Channel Implicit") {
        options.setChannelPosition(xvigra::ChannelPosition::IMPLICIT);
        xt::xtensor<InputType, 2> input{
            { 1,  2,  3,  4,  5}, 
            { 6,  7,  8,  9, 10}, 
            {11, 12, 13, 14, 15}, 
            {16, 17, 18, 19, 20}, 
            {21, 22, 23, 24, 25}
        };
        xt::xtensor<ResultType, 2> expected{
            {128.8f, 144.8f, 160.8f}, 
            {208.8f, 224.8f, 240.8f}, 
            {288.8f, 304.8f, 320.8f}
        };

        checkConvolution2DImplicit(input, kernel, options, expected);
    }
}


TEST_CASE_TEMPLATE("Convolve2D: Test Border Treatment", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    xt::xtensor<InputType, 3> input{{
        { 1,  2,  3,  4,  5}, 
        { 6,  7,  8,  9, 10}, 
        {11, 12, 13, 14, 15}, 
        {16, 17, 18, 19, 20}, 
        {21, 22, 23, 24, 25}
    }};
    xt::xtensor<KernelType, 4> kernel{{{
        {1.00f, 1.30f, 1.70f},
        {1.30f, 1.69f, 2.21f},
        {1.70f, 2.21f, 2.89f}
    }}};

    xvigra::KernelOptions2D options;
    options.setChannelPosition(xvigra::ChannelPosition::FIRST);
    options.setPadding(1);

    SUBCASE("BorderTreatment::constant(0)") {
        options.setBorderTreatment(xvigra::BorderTreatment::constant(0));
        xt::xtensor<ResultType, 3> expected{{
            { 39.60f,  60.10f,  72.10f,  84.10f,  51.05f},
            { 89.30f, 128.80f, 144.80f, 160.80f,  96.05f},
            {149.30f, 208.80f, 224.80f, 240.80f, 142.05f},
            {209.30f, 288.80f, 304.80f, 320.80f, 188.05f},
            {133.81f, 184.01f, 193.21f, 202.41f, 118.45f}
        }};

        checkConvolution2D(input, kernel, options, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::constant(2)") {
        options.setBorderTreatment(xvigra::BorderTreatment::constant(2));
        xt::xtensor<ResultType, 3> expected{{
            { 53.60f,  68.10f,  80.10f,  92.10f,  69.25f},
            { 97.30f, 128.80f, 144.80f, 160.80f, 109.65f},
            {157.30f, 208.80f, 224.80f, 240.80f, 155.65f},
            {217.30f, 288.80f, 304.80f, 320.80f, 201.65f},
            {152.01f, 197.61f, 206.81f, 216.01f, 139.87f}
        }};

        checkConvolution2D(input, kernel, options, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::asymmetricReflect()") {
        options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());
        xt::xtensor<ResultType, 3> expected{{
            { 80.80f,  88.80f, 104.80f, 120.80f, 123.20f},
            {120.80f, 128.80f, 144.80f, 160.80f, 163.20f},
            {200.80f, 208.80f, 224.80f, 240.80f, 243.20f},
            {280.80f, 288.80f, 304.80f, 320.80f, 323.20f},
            {292.80f, 300.80f, 316.80f, 332.80f, 335.20f}
        }};

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::avoid()") {
        options.setBorderTreatment(xvigra::BorderTreatment::avoid());
        xt::xtensor<ResultType, 3> expected{{
            {128.8f, 144.8f, 160.8f},
            {208.8f, 224.8f, 240.8f},
            {288.8f, 304.8f, 320.8f},
        }};

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::repeat()") {
        options.setBorderTreatment(xvigra::BorderTreatment::repeat());
        xt::xtensor<ResultType, 3> expected{{
            { 56.80f,  68.80f,  84.80f, 100.80f, 110.00f},
            {116.80f, 128.80f, 144.80f, 160.80f, 170.00f},
            {196.80f, 208.80f, 224.80f, 240.80f, 250.00f},
            {276.80f, 288.80f, 304.80f, 320.80f, 330.00f},
            {322.80f, 334.80f, 350.80f, 366.80f, 376.00f}
        }};

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::symmetricReflect()") {
        options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());
        xt::xtensor<ResultType, 3> expected{{
               { 56.80f,  68.80f,  84.80f, 100.80f, 110.00f},
               {116.80f, 128.80f, 144.80f, 160.80f, 170.00f},
               {196.80f, 208.80f, 224.80f, 240.80f, 250.00f},
               {276.80f, 288.80f, 304.80f, 320.80f, 330.00f},
               {322.80f, 334.80f, 350.80f, 366.80f, 376.00f}
        }};

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("BorderTreatment::wrap()") {
        options.setBorderTreatment(xvigra::BorderTreatment::wrap());
        xt::xtensor<ResultType, 3> expected{{
            {152.80f, 148.80f, 164.80f, 180.80f, 162.80f},
            {132.80f, 128.80f, 144.80f, 160.80f, 142.80f},
            {212.80f, 208.80f, 224.80f, 240.80f, 222.80f},
            {292.80f, 288.80f, 304.80f, 320.80f, 302.80f},
            {202.80f, 198.80f, 214.80f, 230.80f, 212.80f}
        }};

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("(asymmetricReflect, avoid) (avoid, avoid)") {
        options.setBorderTreatmentBegin(xvigra::BorderTreatment::asymmetricReflect(), xvigra::BorderTreatment::avoid());
        options.setBorderTreatmentEnd(xvigra::BorderTreatment::avoid());
        xt::xtensor<ResultType, 3> expected{{
            { 88.80f, 104.80f, 120.80f},
            {128.80f, 144.80f, 160.80f},
            {208.80f, 224.80f, 240.80f},
            {288.80f, 304.80f, 320.80f}
        }};

        checkConvolution2D(input, kernel, options, expected);
    }

    SUBCASE("(asymmetricReflect, avoid) (avoid, asymmetricReflect)") {
        options.setBorderTreatmentBegin(xvigra::BorderTreatment::asymmetricReflect(), xvigra::BorderTreatment::avoid());
        options.setBorderTreatmentEnd(xvigra::BorderTreatment::avoid(), xvigra::BorderTreatment::asymmetricReflect());
        xt::xtensor<ResultType, 3> expected{{
            { 88.80f, 104.80f, 120.80f, 123.20f},
            {128.80f, 144.80f, 160.80f, 163.20f},
            {208.80f, 224.80f, 240.80f, 243.20f},
            {288.80f, 304.80f, 320.80f, 323.20f}
        }};

        checkConvolution2D(input, kernel, options, expected);
    }
}


TEST_CASE_TEMPLATE("Convolve2D: Test With Images", T, FLOATING_TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    xvigra::KernelOptions2D options;

    SUBCASE("0x0-1x1-1x1") {
        options.setPadding(0, 0);
        options.setStride(1, 1);
        options.setDilation(1, 1);

        SUBCASE("BorderTreatment::asymmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/ASYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/ASYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::avoid()") {
            options.setBorderTreatment(xvigra::BorderTreatment::avoid());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/AVOID/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/AVOID/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(0)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(0));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/CONSTANT_0/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/CONSTANT_0/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(2)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(2));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/CONSTANT_2/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/CONSTANT_2/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::repeat()") {
            options.setBorderTreatment(xvigra::BorderTreatment::repeat());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/REPEAT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/REPEAT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::symmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/SYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/SYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::wrap()") {
            options.setBorderTreatment(xvigra::BorderTreatment::wrap());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/WRAP/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-1x1/WRAP/Piercing-The-Ocean.ppm"
                );
            }
        }
    }

    SUBCASE("3x2-1x1-1x1") {
        options.setPadding(3, 2);
        options.setStride(1, 1);
        options.setDilation(1, 1);

        SUBCASE("BorderTreatment::asymmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/ASYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/ASYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::avoid()") {
            options.setBorderTreatment(xvigra::BorderTreatment::avoid());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/AVOID/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/AVOID/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(0)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(0));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/CONSTANT_0/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/CONSTANT_0/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(2)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(2));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/CONSTANT_2/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/CONSTANT_2/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::repeat()") {
            options.setBorderTreatment(xvigra::BorderTreatment::repeat());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/REPEAT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/REPEAT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::symmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/SYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/SYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::wrap()") {
            options.setBorderTreatment(xvigra::BorderTreatment::wrap());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/WRAP/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-1x1-1x1/WRAP/Piercing-The-Ocean.ppm"
                );
            }
        }
    }

    SUBCASE("0x0-4x5-1x1") {
        options.setPadding(0, 0);
        options.setStride(4, 5);
        options.setDilation(1, 1);

        SUBCASE("BorderTreatment::asymmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/ASYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/ASYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::avoid()") {
            options.setBorderTreatment(xvigra::BorderTreatment::avoid());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/AVOID/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/AVOID/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(0)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(0));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/CONSTANT_0/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/CONSTANT_0/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(2)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(2));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/CONSTANT_2/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/CONSTANT_2/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::repeat()") {
            options.setBorderTreatment(xvigra::BorderTreatment::repeat());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/REPEAT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/REPEAT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::symmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/SYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/SYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::wrap()") {
            options.setBorderTreatment(xvigra::BorderTreatment::wrap());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/WRAP/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-4x5-1x1/WRAP/Piercing-The-Ocean.ppm"
                );
            }
        }
    }

    SUBCASE("0x0-1x1-2x3") {
        options.setPadding(0, 0);
        options.setStride(1, 1);
        options.setDilation(2, 3);

        SUBCASE("BorderTreatment::asymmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/ASYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/ASYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::avoid()") {
            options.setBorderTreatment(xvigra::BorderTreatment::avoid());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/AVOID/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/AVOID/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(0)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(0));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/CONSTANT_0/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/CONSTANT_0/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(2)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(2));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/CONSTANT_2/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/CONSTANT_2/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::repeat()") {
            options.setBorderTreatment(xvigra::BorderTreatment::repeat());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/REPEAT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/REPEAT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::symmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/SYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/SYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::wrap()") {
            options.setBorderTreatment(xvigra::BorderTreatment::wrap());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/WRAP/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/0x0-1x1-2x3/WRAP/Piercing-The-Ocean.ppm"
                );
            }
        }
    }

    SUBCASE("3x2-4x5-2x3") {
        options.setPadding(3, 2);
        options.setStride(4, 5);
        options.setDilation(2, 3);

        SUBCASE("BorderTreatment::asymmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/ASYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/ASYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::avoid()") {
            options.setBorderTreatment(xvigra::BorderTreatment::avoid());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/AVOID/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/AVOID/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(0)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(0));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/CONSTANT_0/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/CONSTANT_0/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::constant(2)") {
            options.setBorderTreatment(xvigra::BorderTreatment::constant(2));

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/CONSTANT_2/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/CONSTANT_2/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::repeat()") {
            options.setBorderTreatment(xvigra::BorderTreatment::repeat());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/REPEAT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/REPEAT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::symmetricReflect()") {
            options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/SYMMETRIC_REFLECT/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/SYMMETRIC_REFLECT/Piercing-The-Ocean.ppm"
                );
            }
        }

        SUBCASE("BorderTreatment::wrap()") {
            options.setBorderTreatment(xvigra::BorderTreatment::wrap());

            SUBCASE("PGM") {
                xt::xtensor<KernelType, 4> kernel{{EDGE_KERNEL}};

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PGM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/WRAP/Piercing-The-Ocean.pgm"
                );
            }

            SUBCASE("PPM") {
                xt::xtensor<KernelType, 4> kernel{
                    {EDGE_KERNEL, ZERO_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, EDGE_KERNEL, ZERO_KERNEL},
                    {ZERO_KERNEL, ZERO_KERNEL, EDGE_KERNEL}
                };

                checkConvolution2DWithImage<InputType, KernelType, ResultType>(
                    TEST_IMAGE_PPM,
                    kernel,
                    options, 
                    "./resources/tests/3x2-4x5-2x3/WRAP/Piercing-The-Ocean.ppm"
                );
            }
        }
    }
}


TEST_CASE_TEMPLATE("Convolve2D: Test Invalid Configurations", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    SUBCASE("Input wrong dimension") {
        xt::xarray<float> inputTooSmall(std::vector<std::size_t>{7, 6});
        xt::xarray<float> inputTooBig(std::vector<std::size_t>{7, 6, 5, 3});

        xt::xarray<float> kernel(std::vector<std::size_t>{1, 1, 1, 3});

        xvigra::KernelOptions2D options;
        options.setChannelPosition(xvigra::ChannelPosition::LAST);

        REQUIRE_EQ(inputTooSmall.dimension(), 2);
        REQUIRE_EQ(inputTooBig.dimension(), 4);
        REQUIRE_EQ(kernel.dimension(), 4);

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2D(inputTooSmall, kernel, options),
            "convolve2D(): Need 3 dimensional (H x W x C or C x H x W) input!",
            std::invalid_argument
        );

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2D(inputTooBig, kernel, options),
            "convolve2D(): Need 3 dimensional (H x W x C or C x H x W) input!",
            std::invalid_argument
        );
    }

    SUBCASE("Implicit: Input wrong dimension") {
        xt::xarray<float> inputTooSmall(std::vector<std::size_t>{});
        xt::xarray<float> inputTooBig(std::vector<std::size_t>{7, 3, 3});

        xt::xarray<float> kernel(std::vector<std::size_t>{1, 1, 1, 3});

        xvigra::KernelOptions2D options;
        options.setChannelPosition(xvigra::ChannelPosition::IMPLICIT);

        REQUIRE_EQ(inputTooSmall.dimension(), 0);
        REQUIRE_EQ(inputTooBig.dimension(), 3);
        REQUIRE_EQ(kernel.dimension(), 4);

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2DImplicit(inputTooSmall, kernel, options),
            "convolve2DImplicit(): Need 2 dimensional (H x W) input!",
            std::invalid_argument
        );

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2DImplicit(inputTooBig, kernel, options),
            "convolve2DImplicit(): Need 2 dimensional (H x W) input!",
            std::invalid_argument
        );
    }

    SUBCASE("Implicit Input, Explicit Option") {
        xt::xarray<InputType> input(std::vector<std::size_t>{9});
        xt::xarray<KernelType> kernel(std::vector<std::size_t>{1, 1, 3});

        xvigra::KernelOptions options;
        options.channelPosition = xvigra::ChannelPosition::FIRST;

        CHECK_THROWS_WITH_AS(
            xvigra::convolve1DImplicit(input, kernel, options),
            "convolve1DImplicit(): Expected implicit channels in options!",
            std::domain_error
        );
    }

    SUBCASE("Implicit Input, Explicit Option") {
        xt::xarray<InputType> input(std::vector<std::size_t>{5, 4});
        xt::xarray<KernelType> kernel(std::vector<std::size_t>{1, 1, 3, 3});
        xvigra::KernelOptions2D options;
        options.setChannelPosition(xvigra::ChannelPosition::FIRST);

        REQUIRE_EQ(input.dimension(), 2);
        REQUIRE_EQ(kernel.dimension(), 4);

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2DImplicit(input, kernel, options),
            "convolve2DImplicit(): Expected implicit channels in options!",
            std::domain_error
        );
    }

    SUBCASE("Explicit Input, Implicit Option") {
        xt::xarray<InputType> input(std::vector<std::size_t>{7, 5, 3});
        xt::xarray<KernelType> kernel(std::vector<std::size_t>{1, 1, 3, 3});

        xvigra::KernelOptions2D options;
        options.setChannelPosition(xvigra::ChannelPosition::IMPLICIT);

        REQUIRE_EQ(input.dimension(), 3);
        REQUIRE_EQ(kernel.dimension(), 4);

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2D(input, kernel, options),
            "convolve2D(): Implicit channel option is not supported for explicit channels in input!",
            std::invalid_argument
        );
    }

    SUBCASE("Input Channel Mismatch In Input And Kernel") {
        xt::xarray<InputType> input(std::vector<std::size_t>{7, 5, 3});
        xt::xarray<KernelType> kernel(std::vector<std::size_t>{1, 1, 3, 3});

        xvigra::KernelOptions2D options;
        options.setChannelPosition(xvigra::ChannelPosition::LAST);

        REQUIRE_EQ(input.dimension(), 3);
        REQUIRE_EQ(kernel.dimension(), 4);

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2D(input, kernel, options),
            "convolve2D(): Input channels of input and kernel do not align!",
            std::invalid_argument
        );
    }

    SUBCASE("Kernel Too Big") {
        xt::xarray<InputType> input(std::vector<std::size_t>{5, 5, 3});
        xt::xarray<KernelType> kernelTooTall(std::vector<std::size_t>{3, 3, 6, 4});
        xt::xarray<KernelType> kernelTooWide(std::vector<std::size_t>{3, 3, 4, 6});

        xvigra::KernelOptions2D options;
        options.setChannelPosition(xvigra::ChannelPosition::LAST);

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2D(input, kernelTooTall, options),
            "convolve2D(): Kernel height is greater than padded input height!",
            std::invalid_argument
        );

        CHECK_THROWS_WITH_AS(
            xvigra::convolve2D(input, kernelTooWide, options),
            "convolve2D(): Kernel width is greater than padded input width!",
            std::invalid_argument
        );
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test convolve2D - end                                                                                            ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
