#include <cmath>

#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#include "raw/array_view_3d.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xview.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - begin                                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

#define TYPES short, int, float, double
#define SIZE 15
#define CREATE_COPIED                                                   \
    OutputTensor stridedViewCopied = xt::ones<T>({outputY, outputX});   \
    OutputTensor rawViewCopied = xt::ones<T>({outputY, outputX});       \
    auto resData = rawViewCopied.data();

#define ZYX_SUBCASES {                                                                                                            \
        SUBCASE ("Constant Z") {                                                                                                  \
            int outputY = static_cast<int>(std::ceil(static_cast<double>((inputY - paddingStartY - paddingEndY)) / (strideY)));   \
            int outputX = static_cast<int>(std::ceil(static_cast<double>((inputX - paddingStartX - paddingEndX)) / (strideX)));   \
                                                                                                                                  \
            CREATE_COPIED                                                                                                         \
                                                                                                                                  \
            int z = inputZ / 2;                                                                                                   \
            auto view = xt::strided_view(input, {                                                                                 \
                z,                                                                                                                \
                xt::range(paddingStartY, inputY - paddingEndY, strideY),                                                          \
                xt::range(paddingStartX, inputX - paddingEndX, strideX)                                                           \
            });                                                                                                                   \
            raw::ArrayView3D rawView{                                                                                             \
                {static_cast<int>(inputZ), static_cast<int>(inputY), static_cast<int>(inputX)},                                   \
                {0, paddingStartY, paddingStartX},                                                                                \
                {1, strideY, strideX}                                                                                             \
            };                                                                                                                    \
            std::copy(view.begin(), view.end(), stridedViewCopied.begin());                                                       \
                                                                                                                                  \
                                                                                                                                  \
            for(int y = 0; y < outputY; ++y) {                                                                                    \
                for(int x = 0; x < outputX; ++x) {                                                                                \
                    *raw::access_direct(resData, outputX, y, x) = *rawView.access(rawData, z, y, x);                              \
                }                                                                                                                 \
            }                                                                                                                     \
                                                                                                                                  \
            CHECK_EQ(rawViewCopied, stridedViewCopied);                                                                           \
        }                                                                                                                         \
                                                                                                                                  \
        SUBCASE ("Constant Y") {                                                                                                  \
            int outputY = static_cast<int>(std::ceil(static_cast<double>((inputZ - paddingStartZ - paddingEndZ)) / (strideZ)));   \
            int outputX = static_cast<int>(std::ceil(static_cast<double>((inputX - paddingStartX - paddingEndX)) / (strideX)));   \
                                                                                                                                  \
            CREATE_COPIED                                                                                                         \
                                                                                                                                  \
            int y = inputY / 2;                                                                                                   \
            auto view = xt::strided_view(input, {                                                                                 \
                xt::range(paddingStartZ, inputZ - paddingEndZ, strideZ),                                                          \
                y,                                                                                                                \
                xt::range(paddingStartX, inputX - paddingEndX, strideX)                                                           \
            });                                                                                                                   \
            raw::ArrayView3D rawView{                                                                                             \
                {static_cast<int>(inputZ), static_cast<int>(inputY), static_cast<int>(inputX)},                                   \
                {paddingStartZ, 0, paddingStartX},                                                                                \
                {strideZ, 1, strideX}                                                                                             \
            };                                                                                                                    \
            std::copy(view.begin(), view.end(), stridedViewCopied.begin());                                                       \
                                                                                                                                  \
            for(int z = 0; z < outputY; ++z) {                                                                                    \
                for(int x = 0; x < outputX; ++x) {                                                                                \
                    *raw::access_direct(resData, outputX, z, x) = *rawView.access(rawData, z, y, x);                              \
                }                                                                                                                 \
            }                                                                                                                     \
                                                                                                                                  \
            CHECK_EQ(rawViewCopied, stridedViewCopied);                                                                           \
        }                                                                                                                         \
                                                                                                                                  \
        SUBCASE ("Constant X") {                                                                                                  \
            int outputY = static_cast<int>(std::ceil(static_cast<double>((inputZ - paddingStartZ - paddingEndZ)) / (strideZ)));   \
            int outputX = static_cast<int>(std::ceil(static_cast<double>((inputY - paddingStartY - paddingEndY)) / (strideY)));   \
                                                                                                                                  \
            CREATE_COPIED                                                                                                         \
                                                                                                                                  \
            int x = inputX / 2;                                                                                                   \
            auto view = xt::strided_view(input, {                                                                                 \
                xt::range(paddingStartZ, inputZ - paddingEndZ, strideZ),                                                          \
                xt::range(paddingStartY, inputY - paddingEndY, strideY),                                                          \
                x                                                                                                                 \
            });                                                                                                                   \
            raw::ArrayView3D rawView{                                                                                             \
                {static_cast<int>(inputZ), static_cast<int>(inputY), static_cast<int>(inputX)},                                   \
                {paddingStartZ, paddingStartY, 0},                                                                                \
                {strideZ, strideY, 1}                                                                                             \
            };                                                                                                                    \
            std::copy(view.begin(), view.end(), stridedViewCopied.begin());                                                       \
                                                                                                                                  \
            for(int z = 0; z < outputY; ++z) {                                                                                    \
                for(int y = 0; y < outputX; ++y) {                                                                                \
                    *raw::access_direct(resData, outputX, z, y) = *rawView.access(rawData, z, y, x);                              \
                }                                                                                                                 \
            }                                                                                                                     \
                                                                                                                                  \
            CHECK_EQ(rawViewCopied, stridedViewCopied);                                                                           \
        }                                                                                                                         \
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - end                                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test ArrayView3D - begin                                                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE ("Test ArrayView3D", T, TYPES) {
    using InputTensor = typename xt::xtensor<T, 3>;
    using OutputTensor = typename xt::xtensor<T, 2>;
    using ShapeType = typename InputTensor::shape_type;

    std::size_t inputZ = SIZE - 1;
	std::size_t inputY = SIZE;
	std::size_t inputX = SIZE + 1;

    ShapeType inputShape{inputZ, inputY, inputX};
    InputTensor input = xt::ones<T>(inputShape);
    int i = 0;
    for(auto iter = input.begin(); iter < input.end(); ++iter) {
        *iter = static_cast<T>(i++);
    }

    auto rawData = input.data();

    SUBCASE ("complete") {
	    int paddingStartZ = 0;
        int paddingEndZ = 0;
        int paddingStartY = 0;
        int paddingEndY = 0;
        int paddingStartX = 0;
        int paddingEndX = 0;

        int strideZ = 1;
        int strideY = 1;
        int strideX = 1;

	    ZYX_SUBCASES
	}

    SUBCASE ("padding") {
	    int paddingStartZ = 1;
        int paddingEndZ = 2;
        int paddingStartY = 3;
        int paddingEndY = 2;
        int paddingStartX = 2;
        int paddingEndX = 1;

        int strideZ = 1;
        int strideY = 1;
        int strideX = 1;

	    ZYX_SUBCASES
	}

    SUBCASE ("stride") {
	    int paddingStartZ = 0;
        int paddingEndZ = 0;
        int paddingStartY = 0;
        int paddingEndY = 0;
        int paddingStartX = 0;
        int paddingEndX = 0;

        int strideZ = 4;
        int strideY = 2;
        int strideX = 3;

	    ZYX_SUBCASES
	}

	SUBCASE ("padding & stride") {
	    int paddingStartZ = 1;
        int paddingEndZ = 2;
        int paddingStartY = 3;
        int paddingEndY = 2;
        int paddingStartX = 2;
        int paddingEndX = 1;

        int strideZ = 4;
        int strideY = 2;
        int strideX = 3;

	    ZYX_SUBCASES
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test ArrayView3D - end                                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
