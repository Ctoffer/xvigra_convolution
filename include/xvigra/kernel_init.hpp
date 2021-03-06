#ifndef XVIGRA_KERNEL_INIT_HPP
#define XVIGRA_KERNEL_INIT_HPP

#include <stdexcept>
#include <string>

#include "vigra/gaussians.hxx"

#include "xtensor/xbuilder.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xstrided_view.hpp"

namespace xvigra {
    template <class KernelValueType>
    xt::xarray<KernelValueType> initGaussian(double stdDev,
                                             double windowRatio=0.0) {
        if (stdDev > 0.0) {
            vigra::Gaussian<KernelValueType> gauss(static_cast<KernelValueType>(stdDev));

            int radius;
            if (windowRatio == 0.0) {
                radius = static_cast<int>(3.0 * stdDev + 0.5);
            } else {
                radius = static_cast<int>(windowRatio * stdDev + 0.5);
            }

            if (radius == 0) {
                radius = 1;
            }

            using ShapeType = typename xt::xarray<KernelValueType>::shape_type;
            xt::xarray<KernelValueType> array(ShapeType{static_cast<typename ShapeType::value_type>(radius*2 + 1)});

            std::size_t i = 0;
            for (int x = -radius; x <= radius; ++x, ++i) {
                array[i] = gauss(x);
            }

            return array;
        } else {
            xt::xarray<KernelValueType> array = {1.0};
            return array;
        }
    }

    template <class KernelValueType>
    xt::xarray<KernelValueType> initGaussianDerivative(double stdDev,
                                                       int order=0,
                                                       double windowRatio=0.0) {
        if (order == 0) {
            return initGaussian<KernelValueType>(stdDev, windowRatio);
        }

        vigra::Gaussian<KernelValueType> gauss(static_cast<KernelValueType>(stdDev), order);

        int radius;
        if (windowRatio == 0.0) {
            radius = static_cast<int>((3.0  + 0.5 * order) * stdDev + 0.5);
        } else {
            radius = static_cast<int>(windowRatio * stdDev + 0.5);
        }

        if (radius == 0) {
            radius = 1;
        }

        using ShapeType = typename xt::xarray<KernelValueType>::shape_type;
        xt::xarray<KernelValueType> array(ShapeType{static_cast<typename ShapeType::value_type>(radius*2 + 1)});

        std::size_t i = 0;
        for (int x = -radius; x <= radius; ++x, ++i) {
            array[i] = gauss(x);
        }

        return array;
    }
}

#endif // XVIGRA_KERNEL_INIT_HPP