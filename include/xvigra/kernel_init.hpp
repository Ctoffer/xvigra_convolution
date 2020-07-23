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
            xt::xarray<KernelValueType> array(ShapeType{radius*2 + 1});

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
}

#endif // XVIGRA_KERNEL_INIT_HPP