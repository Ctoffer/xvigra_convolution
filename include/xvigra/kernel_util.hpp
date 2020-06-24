#ifndef XVIGRA_KERNEL_UTIL_HPP
#define XVIGRA_KERNEL_UTIL_HPP

#include <stdexcept>
#include <string>

#include "xtensor/xbuilder.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xstrided_view.hpp"

namespace xvigra {
    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ kernel promote - begin                                                                                       ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T>
    auto promoteKernelToFull1D(const xt::xexpression<T>& kernelExpression, std::size_t outputChannels=0) {
        
        using KernelContainerType = typename xt::xexpression<T>::derived_type;
        using KernelType = typename KernelContainerType::value_type;

        KernelContainerType originalKernel = kernelExpression.derived_cast();

        switch(originalKernel.dimension()) {
            case 1: {
                if (outputChannels == 0) {
                    throw new std::invalid_argument("promoteKernelToFull1D(): Need atleast 1 output channel!");
                }


                std::size_t inputChannels = outputChannels;
                std::size_t kernelWidth = originalKernel.shape()[0];
                
                typename xt::xtensor<KernelType, 3>::shape_type kernelShape{outputChannels, inputChannels, kernelWidth};

                xt::xtensor<KernelType, 3> result(kernelShape);

                for (std::size_t outIndex = 0; outIndex < outputChannels; ++outIndex) {
                    for (std::size_t inIndex = 0; inIndex < inputChannels; ++inIndex) {
                        for (std::size_t w = 0; w < kernelWidth; ++w) {
                            result(outIndex, inIndex, w) = outIndex == inIndex ? originalKernel(w) : static_cast<KernelType>(0);
                        }
                    }
                }

                return result;
            }
            case 2: {
                if (outputChannels == 0) {
                    throw new std::invalid_argument("promoteKernelToFull1D(): Need atleast 1 output channel!");
                }

                std::size_t inputChannels = originalKernel.shape()[0];
                std::size_t kernelWidth = originalKernel.shape()[1];
                
                typename xt::xtensor<KernelType, 3>::shape_type kernelShape{outputChannels, inputChannels, kernelWidth};

                xt::xtensor<KernelType, 3> result(kernelShape);

                for (std::size_t outIndex = 0; outIndex < outputChannels; ++outIndex) {
                    for (std::size_t inIndex = 0; inIndex < inputChannels; ++inIndex) {
                        for (std::size_t w = 0; w < kernelWidth; ++w) {
                            result(outIndex, inIndex, w) = originalKernel(inIndex, w);
                        }
                    }
                }

                return result;
            }
            case 3: {
                return xt::xtensor<KernelType, 3>(originalKernel);
            }
            default: {
                throw new std::invalid_argument("promoteKernelToFull1D(): Can't promote " + std::to_string(originalKernel.dimension()) + " dimensional kernel!");
            }
        }
    }

    template <typename T>
    auto promoteKernelToFull2D(const xt::xexpression<T>& kernelExpression, std::size_t outputChannels=0) {
        
        using KernelContainerType = typename xt::xexpression<T>::derived_type;
        using KernelType = typename KernelContainerType::value_type;

        KernelContainerType originalKernel = kernelExpression.derived_cast();

        switch(originalKernel.dimension()) {
            case 1: {
                if (outputChannels == 0) {
                    throw new std::invalid_argument("promoteKernelToFull2D(): Need atleast 1 output channel!");
                }

                std::size_t inputChannels = outputChannels;
                std::size_t kernelWidth = originalKernel.shape()[0];
                std::size_t kernelHeight = kernelWidth;
                
                typename xt::xtensor<KernelType, 4>::shape_type kernelShape{outputChannels, inputChannels, kernelHeight, kernelWidth};

                xt::xtensor<KernelType, 4> result(kernelShape);

                for (std::size_t outIndex = 0; outIndex < outputChannels; ++outIndex) {
                    for (std::size_t inIndex = 0; inIndex < inputChannels; ++inIndex) {
                        for (std::size_t h = 0; h < kernelHeight; ++h) {
                            for (std::size_t w = 0; w < kernelWidth; ++w) {
                                result(outIndex, inIndex, h, w) = outIndex == inIndex ? originalKernel(h) * originalKernel(w) : static_cast<KernelType>(0);
                            }
                        }
                    }
                }

                return result;
            }
            case 2: {
                if (outputChannels == 0) {
                    throw new std::invalid_argument("promoteKernelToFull2D(): Need atleast 1 output channel!");
                }

                std::size_t inputChannels = outputChannels;
                std::size_t kernelHeight = originalKernel.shape()[0];
                std::size_t kernelWidth = originalKernel.shape()[1];
                
                typename xt::xtensor<KernelType, 4>::shape_type kernelShape{outputChannels, inputChannels, kernelHeight, kernelWidth};

                xt::xtensor<KernelType, 4> result(kernelShape);
                KernelContainerType zero = xt::zeros<KernelType>(originalKernel.shape());

                for (std::size_t outIndex = 0; outIndex < outputChannels; ++outIndex) {
                    for (std::size_t inIndex = 0; inIndex < inputChannels; ++inIndex) {
                        auto sliceVector = xt::xstrided_slice_vector{outIndex, inIndex, xt::all(), xt::all()};
                        xt::strided_view(result, sliceVector) = outIndex == inIndex ? originalKernel : zero;
                    }
                }

                return result;
            }
            case 3: {
                if (outputChannels == 0) {
                    throw new std::invalid_argument("promoteKernelToFull2D(): Need atleast 1 output channel!");
                }

                std::size_t inputChannels = originalKernel.shape()[0];
                std::size_t kernelHeight = originalKernel.shape()[1];
                std::size_t kernelWidth = originalKernel.shape()[2];
                
                typename xt::xtensor<KernelType, 4>::shape_type kernelShape{outputChannels, inputChannels, kernelHeight, kernelWidth};

                xt::xtensor<KernelType, 4> result(kernelShape);

                for (std::size_t outIndex = 0; outIndex < outputChannels; ++outIndex) {
                    xt::strided_view(result, xt::xstrided_slice_vector{outIndex, xt::all(), xt::all(), xt::all()}) = originalKernel;
                }

                return result;
            }
            case 4: {
                return xt::xtensor<KernelType, 4>(originalKernel);
            }
            default: {
                throw new std::invalid_argument("promoteKernelToFull2D(): Can't promote " + std::to_string(originalKernel.dimension()) + " dimensional kernel!");
            }
        }
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ kernel promote - end                                                                                         ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
}

#endif // XVIGRA_KERNEL_UTIL_HPP