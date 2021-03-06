#ifndef XVIGRA_CONVOLUTION_HPP
#define XVIGRA_CONVOLUTION_HPP

#ifdef VOID
#undef VOID
#endif

#include <algorithm>
#include <array>
#include <iostream>

#include "xtensor/xarray.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xtensor.hpp"

#include "xvigra/convolution_util.hpp"
#include "xvigra/separable_convolution.hpp"
#include "xvigra/kernel_init.hpp"

namespace xvigra {
    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ gaussianSmoothing - begin                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <std::size_t N, typename T>
    auto gaussianSmoothing(const xt::xexpression<T>& sourceExpression,
                           std::array<double, N> scales) {
        using SourceContainerType = typename xt::xexpression<T>::derived_type;
        using SourceType = typename SourceContainerType::value_type;

        SourceContainerType source = sourceExpression.derived_cast();
        std::array<xt::xarray<SourceType>, N> gaussianKernels;
        std::transform(scales.begin(),
                     scales.end(),
                     gaussianKernels.begin(),
                     [](double scale) -> xt::xarray<SourceType> {return xvigra::initGaussian<SourceType>(scale);}
                     );
        std::array<xvigra::KernelOptions, N> options;
        for(std::size_t i = 0; i < N; ++i) {
            xvigra::KernelOptions option;
            option.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());
            option.setPadding(gaussianKernels[i].shape()[0] / 2);
            options[i] = option;
        }


        return xvigra::separableConvolve(source, gaussianKernels, options);
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ gaussianSmoothing - end                                                                                      ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ gaussianSharpening - begin                                                                                   ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <std::size_t N, typename T>
    auto gaussianSharpening(const xt::xexpression<T>& sourceExpression,
                            double sharpeningFactor,
                            double scale) {
        using SourceContainerType = typename xt::xexpression<T>::derived_type;

        SourceContainerType source = sourceExpression.derived_cast();
        std::array<double, N> scales;
        for (std::size_t i = 0; i < N; ++i) {
            scales[i] = scale;
        }

        auto smoothed = gaussianSmoothing<N>(source, scales);

        return xt::eval((1.0 + sharpeningFactor) * source - sharpeningFactor * smoothed);
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ gaussianSharpening - end                                                                                     ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ gaussianGradient - begin                                                                                     ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <std::size_t N, typename T>
    auto gaussianGradient(const xt::xexpression<T>& sourceExpression,
                                                        double scale) {
        using SourceContainerType = typename xt::xexpression<T>::derived_type;
        using V = typename SourceContainerType::value_type;

        SourceContainerType source = sourceExpression.derived_cast();
        auto smooth = xvigra::initGaussian<V>(scale);
        auto grad = xvigra::initGaussianDerivative<V>(scale, 1);

        std::array<xt::xarray<V>, N> kernels;
        std::array<xvigra::KernelOptions, N> options;

        for(std::size_t i = 0; i < N; ++i) {
            kernels[i] = smooth;

            xvigra::KernelOptions option;
            option.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());
            option.setPadding(kernels[i].shape()[0] / 2);
            options[i] = option;
        }

        std::array<xt::xarray<V>, N> specializedKernels;
        std::array<xvigra::KernelOptions, N> specializedOptions;
        std::array<xt::xtensor<V, N+1>, N> result;

        for (std::size_t i = 0; i < N; ++i) {
            std::copy(kernels.begin(), kernels.end(), specializedKernels.begin());
            std::copy(options.begin(), options.end(), specializedOptions.begin());

            specializedKernels[i] = grad;
            specializedOptions[i].setPadding(specializedKernels[i].shape()[0] / 2);

            result[i] = xvigra::separableConvolve(source, specializedKernels, specializedOptions);
        }

        return result;
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ gaussianGradient - end                                                                                       ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
}

#endif // XVIGRA_CONVOLUTION_HPP