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
        "./resources/src/demo_small.pgm"
    );
    auto result = xvigra::gaussianSharpening<2>(image, 2.3, 0.5);
    xvigra::saveImage(
        "./resources/src/demo_gaussian_sharpening.pgm",
        xvigra::normalizeAfterConvolution<float>(result)
    );
}


int main() {
    // FIXME Something seems broken in the implementation, but after several hours of debugging I didn't find
    //       the issue. Unfortunately, the time frame of my thesis work is over
    runGaussianSmoothingDemo();
    //runGaussianSharpeningDemo();

    return 0;
}