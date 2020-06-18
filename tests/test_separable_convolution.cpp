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

#include "xtensor/xarray.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xtensor.hpp"

#include "xvigra/convolution_util.hpp"
#include "xvigra/explicit_convolution.hpp"
#include "xvigra/separable_convolution.hpp"
#include "xvigra/image_io.hpp"
#include "xvigra/io_util.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - begin                                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

#define TYPE_PAIRS \
    std::pair<short, float>, \
    std::pair<short, double>, \
    std::pair<int, float>, \
    std::pair<int, double>
    

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - end                                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


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
        CHECK_MESSAGE(
            *iterActual == doctest::Approx(*iterExpected).epsilon(epsilon), 
            "actual: " << *iterActual << ", expected: " << *iterExpected
        );
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ utility - end                                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test separableConvolve1D - begin                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE("SeparableConvolve1D: Test Default Options", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    xvigra::KernelOptions options;

    SUBCASE("Symmetric Kernel") {
        xt::xtensor<InputType, 2> input{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        xt::xtensor<KernelType, 1> kernel{1.0f, 1.3f, 1.7f, 2.11f};
        xt::xtensor<KernelType, 3> fullKernel{{{1.0f, 1.3f, 1.7f, 2.11f}}};

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::LAST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("Asymmetric Kernel") {
        xt::xtensor<InputType, 2> input{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        xt::xtensor<KernelType, 1> kernel{1.0f, 1.3f, 1.7f};
        xt::xtensor<KernelType, 3> fullKernel{{{1.0f, 1.3f, 1.7f}}};

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::LAST>(input, kernel, options);

        checkExpressions(actual, expected);
    }
}


TEST_CASE_TEMPLATE("SeparableConvolve1D: Test Channel Position", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    xt::xtensor<KernelType, 1> kernel{1.0f, 1.3f, 1.7f};
    xt::xtensor<KernelType, 3> fullKernel{{{1.0f, 1.3f, 1.7f}}};
    xvigra::KernelOptions options;

    SUBCASE("Channel First") {
        options.channelPosition = xvigra::ChannelPosition::FIRST;   
        xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5, 6, 7, 8, 9}};

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("Channel Last") {
        options.channelPosition = xvigra::ChannelPosition::LAST;
        xt::xtensor<InputType, 2> input{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::LAST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    // SUBCASE("Channel Implicit") {
    //     options.channelPosition = xvigra::ChannelPosition::IMPLICIT;
    //     xt::xtensor<InputType, 1> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
        
    //     auto expected = xvigra::convolve1D(input, fullKernel, options);
    //     auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::LAST>(input, kernel, options);

    //     checkExpressions(actual, expected);
    // }
}


TEST_CASE_TEMPLATE("SeparableConvolve1D: Test Border Treatment", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    xt::xtensor<KernelType, 1> kernel{1.0f, 1.3f, 1.7f};
    xt::xtensor<KernelType, 3> fullKernel{{{1.0f, 1.3f, 1.7f}}};

    xvigra::KernelOptions options;
    options.channelPosition = xvigra::ChannelPosition::FIRST;
    options.setPadding(2);
    

    SUBCASE("BorderTreatment::constant(0)") {
        options.setBorderTreatment(xvigra::BorderTreatment::constant(0));

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("BorderTreatment::constant(2)") {
        options.setBorderTreatment(xvigra::BorderTreatment::constant(2));
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("BorderTreatment::asymmetricReflect()") {
        options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("BorderTreatment::avoid()") {
        options.setBorderTreatment(xvigra::BorderTreatment::avoid());
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("BorderTreatment::repeat()") {
        options.setBorderTreatment(xvigra::BorderTreatment::repeat());
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("BorderTreatment::symmetricReflect()") {
        options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("BorderTreatment::wrap()") {
        options.setBorderTreatment(xvigra::BorderTreatment::wrap());
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("BorderTreatment::avoid() & BorderTreatment::asymmetricReflect()") {
        options.setBorderTreatmentBegin(xvigra::BorderTreatment::avoid());
        options.setBorderTreatmentEnd(xvigra::BorderTreatment::asymmetricReflect());
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("BorderTreatment::asymmetricReflect() & BorderTreatment::avoid()") {
        options.setBorderTreatmentBegin(xvigra::BorderTreatment::asymmetricReflect());
        options.setBorderTreatmentEnd(xvigra::BorderTreatment::avoid());
        
        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }
}


TEST_CASE_TEMPLATE("SeparableConvolve1D: Test Different Padding, Stride, Dilation", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    xt::xtensor<KernelType, 1> kernel{1.0f, 1.3f, 1.7f};
    xt::xtensor<KernelType, 3> fullKernel{{{1.0f, 1.3f, 1.7f}}};

    xvigra::KernelOptions options;
    options.channelPosition = xvigra::ChannelPosition::FIRST;

    SUBCASE("Padding=1, Stride=1, Dilation=1") {
        options.setPadding(1);
        options.setStride(1);
        options.setDilation(1);

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("Padding=0, Stride=2, Dilation=1") {
        options.setPadding(0);
        options.setStride(2);
        options.setDilation(1);

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("Padding=0, Stride=1, Dilation=2") {
        options.setPadding(0);
        options.setStride(1);
        options.setDilation(2);

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }

    SUBCASE("Padding=1, Stride=2, Dilation=2") {
        options.setPadding(1);
        options.setStride(2);
        options.setDilation(2);

        auto expected = xvigra::convolve1D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(input, kernel, options);

        checkExpressions(actual, expected);
    }
}


TEST_CASE_TEMPLATE("SeparableConvolve1D: Test Invalid Configurations", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    SUBCASE("Unsupported ChannelPosition") {
        xt::xtensor<InputType, 2> input{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        xt::xtensor<KernelType, 1> kernel{1.0f, 1.3f, 1.7f};
        xvigra::KernelOptions options;

        auto function = [&](){return xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::IMPLICIT>(input, kernel, options);};

        CHECK_THROWS_WITH_AS(
            function(),
            "separableConvolve1D(): ChannelPosition for input can't be IMPLICIT.",
            std::invalid_argument
        );
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test separableConvolve1D - end                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test separableConvolve2D - begin                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE("SeparableConvolve2D: Test Default Options", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    xvigra::KernelOptions2D options2D;

    SUBCASE("Symmetric Kernel") {
        xt::xtensor<InputType, 3> input{
            {{ 1}, { 2}, { 3}, { 4}, { 5}}, 
            {{ 6}, { 7}, { 8}, { 9}, {10}}, 
            {{11}, {12}, {13}, {14}, {15}}, 
            {{16}, {17}, {18}, {19}, {20}}, 
            {{21}, {22}, {23}, {24}, {25}}
        };
        xt::xtensor<KernelType, 1> kernel{1.00f, 1.30f, 1.70f, 2.10f};
        xt::xtensor<KernelType, 4> fullKernel{{{
            {1.00f, 1.30f, 1.70f, 2.10f},
            {1.30f, 1.69f, 2.21f, 2.73f},
            {1.70f, 2.21f, 2.89f, 3.57f},
            {2.10f, 2.73f, 3.57f, 4.41f}
        }}};

        auto expected = xvigra::convolve2D(input, fullKernel, options2D);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::LAST>(
            input, 
            kernel, 
            options2D
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("Asymmetric Kernel") {
        xt::xtensor<InputType, 3> input{
            {{ 1}, { 2}, { 3}, { 4}, { 5}}, 
            {{ 6}, { 7}, { 8}, { 9}, {10}}, 
            {{11}, {12}, {13}, {14}, {15}}, 
            {{16}, {17}, {18}, {19}, {20}}, 
            {{21}, {22}, {23}, {24}, {25}}
        };
        xt::xtensor<KernelType, 1> kernel{1.00f, 1.30f, 1.70f};
        xt::xtensor<KernelType, 4> fullKernel{{{
            {1.00f, 1.30f, 1.70f},
            {1.30f, 1.69f, 2.21f},
            {1.70f, 2.21f, 2.89f}
        }}};

        auto expected = xvigra::convolve2D(input, fullKernel, options2D);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::LAST>(
            input, 
            kernel, 
            options2D
        );

        checkExpressions(actual, expected, 1e-5);
    }
}


TEST_CASE_TEMPLATE("SeparableConvolve2D: Test Channel Position", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    xt::xtensor<KernelType, 1> kernel{1.00f, 1.30f, 1.70f};
    xt::xtensor<KernelType, 4> fullKernel{{{
        {1.00f, 1.30f, 1.70f},
        {1.30f, 1.69f, 2.21f},
        {1.70f, 2.21f, 2.89f}
    }}};
    xvigra::KernelOptions2D options2D;

    SUBCASE("Channel First") {
        options2D.setChannelPosition(xvigra::ChannelPosition::FIRST);   
        xt::xtensor<InputType, 3> input{{
            { 1,  2,  3,  4,  5}, 
            { 6,  7,  8,  9, 10}, 
            {11, 12, 13, 14, 15}, 
            {16, 17, 18, 19, 20}, 
            {21, 22, 23, 24, 25}
        }};

        auto expected = xvigra::convolve2D(input, fullKernel, options2D);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options2D
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("Channel Last") {
        options2D.setChannelPosition(xvigra::ChannelPosition::LAST);
        xt::xtensor<InputType, 3> input{
            {{ 1}, { 2}, { 3}, { 4}, { 5}}, 
            {{ 6}, { 7}, { 8}, { 9}, {10}}, 
            {{11}, {12}, {13}, {14}, {15}}, 
            {{16}, {17}, {18}, {19}, {20}}, 
            {{21}, {22}, {23}, {24}, {25}}
        };
        
        auto expected = xvigra::convolve2D(input, fullKernel, options2D);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::LAST>(
            input, 
            kernel, 
            options2D
        );

        checkExpressions(actual, expected, 1e-5);
    }

    // SUBCASE("Channel Implicit") {
    //     options.channelPosition = xvigra::ChannelPosition::IMPLICIT;
    //     xt::xtensor<InputType, 1> input{1, 2, 3, 4, 5, 6, 7, 8, 9};
        
    //     auto expected = xvigra::convolve1D(input, fullKernel, options);
    //     auto actual = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::LAST>(input, kernel, options);

    //     checkExpressions(actual, expected);
    // }
}


TEST_CASE_TEMPLATE("SeparableConvolve1D: Test Border Treatment", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    xt::xtensor<InputType, 3> input{{
        { 1,  2,  3,  4,  5}, 
        { 6,  7,  8,  9, 10}, 
        {11, 12, 13, 14, 15}, 
        {16, 17, 18, 19, 20}, 
        {21, 22, 23, 24, 25}
    }};
    xt::xtensor<KernelType, 1> kernel{1.00f, 1.30f, 1.70f};
    xt::xtensor<KernelType, 4> fullKernel{{{
        {1.00f, 1.30f, 1.70f},
        {1.30f, 1.69f, 2.21f},
        {1.70f, 2.21f, 2.89f}
    }}};

    xvigra::KernelOptions2D options;
    options.setChannelPosition(xvigra::ChannelPosition::FIRST);
    options.setPadding(2);
    

    SUBCASE("BorderTreatment::constant(0)") {
        options.setBorderTreatment(xvigra::BorderTreatment::constant(0));

        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::constant(2)") {
        options.setBorderTreatment(xvigra::BorderTreatment::constant(2));
        
        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::asymmetricReflect()") {
        options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());
        
        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::avoid()") {
        options.setBorderTreatment(xvigra::BorderTreatment::avoid());
        
        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::repeat()") {
        options.setBorderTreatment(xvigra::BorderTreatment::repeat());
        
        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::symmetricReflect()") {
        options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());
        
        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::wrap()") {
        options.setBorderTreatment(xvigra::BorderTreatment::wrap());
        
        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::avoid() & BorderTreatment::asymmetricReflect()") {
        options.setBorderTreatmentBegin(xvigra::BorderTreatment::avoid());
        options.setBorderTreatmentEnd(xvigra::BorderTreatment::asymmetricReflect());
        
        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("BorderTreatment::asymmetricReflect() & BorderTreatment::avoid()") {
        options.setBorderTreatmentBegin(xvigra::BorderTreatment::asymmetricReflect());
        options.setBorderTreatmentEnd(xvigra::BorderTreatment::avoid());
        
        auto expected = xvigra::convolve2D(input, fullKernel, options);
        auto actual = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
            input, 
            kernel, 
            options
        );

        checkExpressions(actual, expected, 1e-5);
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test separableConvolve2D - end                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test separableConvolve<2> - begin                                                                                ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE("SeparableConvolve<2>: Test Default Options", T, TYPE_PAIRS) {
    using InputType = typename T::first_type;
    using KernelType = typename T::second_type;

    xt::xtensor<InputType, 3> input{
        {{ 1}, { 2}, { 3}, { 4}, { 5}}, 
        {{ 6}, { 7}, { 8}, { 9}, {10}}, 
        {{11}, {12}, {13}, {14}, {15}}, 
        {{16}, {17}, {18}, {19}, {20}}, 
        {{21}, {22}, {23}, {24}, {25}}
    };
    xvigra::KernelOptions2D options2D;

    SUBCASE("convolve2D: Symmetric Kernel") {
        xt::xtensor<KernelType, 1> kernel{1.00f, 1.30f, 1.70f, 2.10f};
        xt::xtensor<KernelType, 4> fullKernel{{{
            {1.00f, 1.30f, 1.70f, 2.10f},
            {1.30f, 1.69f, 2.21f, 2.73f},
            {1.70f, 2.21f, 2.89f, 3.57f},
            {2.10f, 2.73f, 3.57f, 4.41f}
        }}};

        auto expected = xvigra::convolve2D(input, fullKernel, options2D);
        auto actual = xvigra::separableConvolve<InputType, KernelType, xvigra::ChannelPosition::LAST, 2>(
            input, 
            {kernel, kernel}, 
            {options2D.optionsY, options2D.optionsX}
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("convolve2D: Asymmetric Kernel") {
        xt::xtensor<KernelType, 1> kernel{1.00f, 1.30f, 1.70f};
        xt::xtensor<KernelType, 4> fullKernel{{{
            {1.00f, 1.30f, 1.70f},
            {1.30f, 1.69f, 2.21f},
            {1.70f, 2.21f, 2.89f}
        }}};

        auto expected = xvigra::convolve2D(input, fullKernel, options2D);
        auto actual = xvigra::separableConvolve<InputType, KernelType, xvigra::ChannelPosition::LAST, 2>(
            input, 
            {kernel, kernel}, 
            {options2D.optionsY, options2D.optionsX}
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("separableConvolve2D: Symmetric Kernel") {
        xt::xtensor<KernelType, 1> kernel{1.00f, 1.30f, 1.70f, 2.10f};
        xt::xtensor<KernelType, 4> fullKernel{{{
            {1.00f, 1.30f, 1.70f, 2.10f},
            {1.30f, 1.69f, 2.21f, 2.73f},
            {1.70f, 2.21f, 2.89f, 3.57f},
            {2.10f, 2.73f, 3.57f, 4.41f}
        }}};

        auto expected = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::LAST>(
            input, 
            {kernel, kernel}, 
            {options2D.optionsY, options2D.optionsX}
        );
        auto actual = xvigra::separableConvolve<InputType, KernelType, xvigra::ChannelPosition::LAST, 2>(
            input, 
            {kernel, kernel}, 
            {options2D.optionsY, options2D.optionsX}
        );

        checkExpressions(actual, expected, 1e-5);
    }

    SUBCASE("separableConvolve2D: Asymmetric Kernel") {
        xt::xtensor<KernelType, 1> kernel{1.00f, 1.30f, 1.70f};
        xt::xtensor<KernelType, 4> fullKernel{{{
            {1.00f, 1.30f, 1.70f},
            {1.30f, 1.69f, 2.21f},
            {1.70f, 2.21f, 2.89f}
        }}};

        auto expected = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::LAST>(
            input, 
            {kernel, kernel}, 
            {options2D.optionsY, options2D.optionsX}
        );
        auto actual = xvigra::separableConvolve<InputType, KernelType, xvigra::ChannelPosition::LAST, 2>(
            input, 
            {kernel, kernel}, 
            {options2D.optionsY, options2D.optionsX}
        );

        checkExpressions(actual, expected, 1e-5);
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test separableConvolve<2> - end                                                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝