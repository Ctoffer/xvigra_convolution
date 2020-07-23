#include <array>
#include <stdexcept>
#include <string>

#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"

#include "xvigra/convolution.hpp"
#include "xvigra/image_io.hpp"


void runGaussianSmoothingDemo(const std::string& fileExtension) {
    xt::xtensor<float, 3> image = xvigra::loadImageAsXTensor<float>(
        "./resources/src/Piercing-The-Ocean." + fileExtension
    );
    auto result = xvigra::gaussianSmoothing<2>(image, std::array<double, 2>{1.5, 1.5});
    xvigra::saveImage(
        "./resources/src/demo_gaussian_smoothing." + fileExtension,
         xvigra::normalizeAfterConvolution<float>(result)
    );
}


void runGaussianSharpeningDemo(const std::string& fileExtension) {
    xt::xtensor<float, 3> image = xvigra::loadImageAsXTensor<float>(
        "./resources/src/Piercing-The-Ocean." + fileExtension
    );
    auto result = xvigra::gaussianSharpening<2>(image, 0.4, 2.5);
    auto res = xvigra::normalizeAfterConvolution<float>(result);
    xvigra::saveImage(
        "./resources/src/demo_gaussian_sharpening." + fileExtension,
        res
    );
}


int main() {
    for (const auto& fileExtension : std::array<std::string, 2>{"pgm", "ppm"}) {
        runGaussianSmoothingDemo(fileExtension);
        runGaussianSharpeningDemo(fileExtension);
    }

    return 0;
}