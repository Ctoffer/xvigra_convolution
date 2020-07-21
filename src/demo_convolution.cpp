#include <stdexcept>

#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"

#include "xvigra/convolution.hpp"
#include "xvigra/image_io.hpp"


void runGaussianSharpeningDemo() {
    xt::xtensor<float, 3> image = xvigra::loadImageAsXTensor<float>(
        "./resources/tests/Piercing-The-Ocean_Small.ppm"
    );
    auto result = xvigra::gaussianSharpening<2>(image, 2.3, 0.5);
    xvigra::saveImage(
        "./resources/src/demo_gaussian_sharpening.ppm",
        xvigra::normalizeAfterConvolution<float>(result)
    );
}


void runGaussianSmoothingDemo() {
    xt::xtensor<float, 3> image = xvigra::loadImageAsXTensor<float>(
        "./resources/tests/Piercing-The-Ocean_Small.ppm"
    );
    auto result = xvigra::gaussianSmoothing<2>(image, std::array<double, 2>{0.1, 0.9});
    xvigra::saveImage(
        "./resources/src/demo_gaussian_smoothing.ppm",
         xvigra::normalizeAfterConvolution<float>(result)
    );
}


int main() {
    runGaussianSharpeningDemo();
    runGaussianSmoothingDemo();

    return 0;
}