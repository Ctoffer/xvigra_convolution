#include <stdexcept>

#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"

#include "xvigra/convolution.hpp"
#include "xvigra/image_io.hpp"


void runGaussianSmoothingDemo() {
    xt::xtensor<float, 3> image = xvigra::loadImageAsXTensor<float>(
        "./resources/src/Piercing-The-Ocean.pgm"
    );
    auto result = xvigra::gaussianSmoothing<2>(image, std::array<double, 2>{1.5, 1.5});
    xvigra::saveImage(
        "./resources/src/demo_gaussian_smoothing.pgm",
         xvigra::normalizeAfterConvolution<float>(result)
    );
}


void runGaussianSharpeningDemo() {
    xt::xtensor<float, 3> image = xvigra::loadImageAsXTensor<float>(
        "./resources/src/Piercing-The-Ocean.pgm"
    );
    auto result = xvigra::gaussianSharpening<2>(image, 0.4, 2.5);
    auto res = xvigra::normalizeAfterConvolution<float>(result);
    xvigra::saveImage(
        "./resources/src/demo_gaussian_sharpening.pgm",
        res
    );
}


int main() {
    runGaussianSmoothingDemo();
    runGaussianSharpeningDemo();

    return 0;
}