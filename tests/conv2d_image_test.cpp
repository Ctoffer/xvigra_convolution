#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#if defined(_WIN32) || defined(_WIN64)
    #define DETECTED_OS_IS_WINDOWS
#elif defined(__unix__)
    #define DETECTED_OS_IS_UNIX
#else
    #error Unknown os detected - assuming UNIX
    #define DETECTED_OS_IS_UNIX
#endif

#include <tuple>
#include <string>

#include "image-io.hpp"
#include "conv2d_v2.hpp"

#include "xtensor/xtensor.hpp"

xt::xtensor<int, 4> filters{{
		{{-2, -1, 0}, {-1,1,1}, {0, 1, 2}}, 
		{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, 
		{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
	}, {
		{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, 
		{{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}}, 
		{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}
	}, {
		{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, 
		{{0, 0, 0}, {0, 0, 0}, {0, 0, 0}}, 
		{{-2, -1, 0}, {-1, 1, 1}, {0, 1, 2}}
	}
};


TEST_CASE("Image -> Default padding, dilation, stride - > Image") {
	std::tuple<int, int> padding{0, 0};
	std::tuple<int, int> dilation{1, 1};
	std::tuple<int, int> stride{1, 1};

    #ifdef DETECTED_OS_IS_WINDOWS
	auto originalImage = loadImage(".\\original.png");
	auto expectedImage = loadImage(".\\relief-filter_0x0_1x1_1x1.png");
    #else
    auto originalImage = loadImage("./original.png");
    auto expectedImage = loadImage("./relief-filter_0x0_1x1_1x1.png");
    #endif
	Conv2D convolution(padding, dilation, stride);
	auto convolutedImage = convolution(originalImage, filters, false);
	auto actualImage = normalizeAfterConvolution(convolutedImage);

	CHECK(expectedImage.shape() == actualImage.shape());
	CHECK(expectedImage == actualImage);
}

TEST_CASE("Image -> Custom padding, Default dilation, stride - > Image") {
	std::tuple<int, int> padding{0, 0};
	std::tuple<int, int> dilation{1, 1};
	std::tuple<int, int> stride{3, 4};

    #ifdef DETECTED_OS_IS_WINDOWS
	auto originalImage = loadImage(".\\original.png");
	auto expectedImage = loadImage(".\\relief-filter_0x0_1x1_3x4.png");
    #else
    auto originalImage = loadImage("./original.png");
    auto expectedImage = loadImage("./relief-filter_0x0_1x1_3x4.png");
    #endif
	Conv2D convolution(padding, dilation, stride);
	auto convolutedImage = convolution(originalImage, filters, false);
	auto actualImage = normalizeAfterConvolution(convolutedImage);

	CHECK(expectedImage.shape() == actualImage.shape());
	CHECK(expectedImage == actualImage);
}

TEST_CASE("Image -> Custom dilation, Default padding, stride - > Image") {
	std::tuple<int, int> padding{0, 0};
	std::tuple<int, int> dilation{3, 2};
	std::tuple<int, int> stride{1, 1};

    #ifdef DETECTED_OS_IS_WINDOWS
	auto originalImage = loadImage(".\\original.png");
	auto expectedImage = loadImage(".\\relief-filter_0x0_3x2_1x1.png");
    #else
    auto originalImage = loadImage("./original.png");
    auto expectedImage = loadImage("./relief-filter_0x0_3x2_1x1.png");
    #endif
	Conv2D convolution(padding, dilation, stride);
	auto convolutedImage = convolution(originalImage, filters, false);
	auto actualImage = normalizeAfterConvolution(convolutedImage);

	CHECK(expectedImage.shape() == actualImage.shape());
	CHECK(expectedImage == actualImage);
}

TEST_CASE("Image -> Custom stride, Default padding, dilation - > Image") {
	std::tuple<int, int> padding{5, 4};
	std::tuple<int, int> dilation{1, 1};
	std::tuple<int, int> stride{1, 1};

    #ifdef DETECTED_OS_IS_WINDOWS
	auto originalImage = loadImage(".\\original.png");
	auto expectedImage = loadImage(".\\relief-filter_5x4_1x1_1x1.png");
    #else
    auto originalImage = loadImage("./original.png");
    auto expectedImage = loadImage("./relief-filter_5x4_1x1_1x1.png");
    #endif
	Conv2D convolution(padding, dilation, stride);
	auto convolutedImage = convolution(originalImage, filters, false);
	auto actualImage = normalizeAfterConvolution(convolutedImage);

	CHECK(expectedImage.shape() == actualImage.shape());
	CHECK(expectedImage == actualImage);
}

TEST_CASE("Image -> Custom padding, dilation, stride - > Image") {
	std::tuple<int, int> padding{5, 4};
	std::tuple<int, int> dilation{3, 2};
	std::tuple<int, int> stride{3, 4};

    #ifdef DETECTED_OS_IS_WINDOWS
	auto originalImage = loadImage(".\\original.png");
	auto expectedImage = loadImage(".\\relief-filter_5x4_3x2_3x4.png");
    #else
    auto originalImage = loadImage("./original.png");
    auto expectedImage = loadImage("./relief-filter_5x4_3x2_3x4.png");
    #endif
	Conv2D convolution(padding, dilation, stride);
	auto convolutedImage = convolution(originalImage, filters, false);
	auto actualImage = normalizeAfterConvolution(convolutedImage);

	CHECK(expectedImage.shape() == actualImage.shape());
	CHECK(expectedImage == actualImage);
}