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


void runGaussianGradientDemo(const std::string& fileExtension) {
    xt::xtensor<float, 3> image = xvigra::loadImageAsXTensor<float>(
        "./resources/src/Piercing-The-Ocean." + fileExtension
    );
    std::array<xt::xtensor<float, 3>, 2> result = xvigra::gaussianGradient<2>(image, 0.4);

    {
        // TODO need other normalization method
        auto tmp = result[1] - xt::amin(result[1])[0];
        auto res = xvigra::normalizeAfterConvolution<float>(xt::eval(tmp / xt::amax(tmp)[0]));
        xvigra::saveImage(
            "./resources/src/demo_gaussian_gradient_y." + fileExtension,
            res
        );
    }
    {
        auto tmp = result[0] - xt::amin(result[0])[0];
        auto res = xvigra::normalizeAfterConvolution<float>(xt::eval(tmp / xt::amax(tmp)[0]));
        xvigra::saveImage(
            "./resources/src/demo_gaussian_gradient_x." + fileExtension,
            res
        );
    }
}


int main() {
    for (const auto& fileExtension : std::array<std::string, 2>{"pgm", "ppm"}) {
        runGaussianSmoothingDemo(fileExtension);
        runGaussianSharpeningDemo(fileExtension);
        runGaussianGradientDemo(fileExtension);
    }

    return 0;
}