#ifndef XVIGRA_SEPARABLE_CONVOLUTION_HPP
#define XVIGRA_SEPARABLE_CONVOLUTION_HPP

#include <array>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xvigra/explicit_convolution.hpp"
#include "xvigra/convolution_util.hpp"
#include "xvigra/kernel_util.hpp"

namespace xvigra {
    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ utility - begin                                                                                              ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    /*
     * <p>
     * Calculates the maximal index for a 'N'-dimensional shape for a flattened index.
     * Only axises between 'startAxis' and 'endAxis' without the 'currentAxis' are used.
     * </p>
     *
     * @tparam N number of dimensions in the shape.
     * @param currentAxis axis which should be ignored.
     * @param startAxis (inclusive) the first axis that should be used.
     * @param endAxis (exclusive) the last axis that should be used.
     * @return the product of all axis in the given range, which are not the 'currentAxis'.
     */
    template <std::size_t N>
    std::size_t calculateMaxIndex(
        const std::array<std::size_t, N>& shape, 
        std::size_t currentAxis,
        std::size_t startAxis, 
        std::size_t endAxis
    ) {
        std::size_t result = 1;

        for (std::size_t i = startAxis; i < endAxis; ++i) {
            if (currentAxis != i){
                result *= shape[i];
            }
        }

        return result;
    }


    template <std::size_t Dim>
    xt::xstrided_slice_vector decomposeIndex(
        std::size_t compoundIndex,
        const std::array<std::size_t, Dim>& shape, 
        std::size_t currentAxis,
        std::size_t startAxis, 
        std::size_t endAxis
    ) {
        xt::xstrided_slice_vector result(shape.size(), xt::all());

        for (std::size_t i = startAxis; i < endAxis; ++i) {
            auto maxIndex = xvigra::calculateMaxIndex<Dim>(shape, currentAxis, i + 1, endAxis);
            if (i != currentAxis) {
                std::size_t rest = compoundIndex % maxIndex;
                result[i] = (compoundIndex - rest) / maxIndex;
                compoundIndex = rest;
            }
        }


        return result;
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ utility - end                                                                                                ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve1D - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename O, typename T>
    auto separableConvolve1D(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& rawKernelExpression,
        const xvigra::KernelOptions& kernelOptions
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelContainerType = typename xt::xexpression<O>::derived_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();
        KernelContainerType rawKernel = rawKernelExpression.derived_cast();

        std::size_t numberOfDimensions = input.dimension();

        if (numberOfDimensions != 2) {
            throw std::invalid_argument("separableConvolve1D(): Need 2 dimensional (W x C) input!");
        }

        std::size_t locationOfChannel = 0;

        if (kernelOptions.channelPosition == xvigra::ChannelPosition::LAST) {
            locationOfChannel = numberOfDimensions - 1;
        } else if (kernelOptions.channelPosition == xvigra::ChannelPosition::FIRST) {
            locationOfChannel = 0;
        } else {
            throw std::invalid_argument("separableConvolve1D(): ChannelPosition for input can't be IMPLICIT.");
        }

        auto inputShape = input.shape();

        std::size_t kernelWidth = rawKernel.shape()[0];
        std::size_t channels = inputShape[locationOfChannel];

        xt::xtensor<KernelType, 3> kernel = xvigra::promoteKernelToFull1D(rawKernel, channels);

        for (std::size_t outIndex = 0; outIndex < channels; ++outIndex) {
            for (std::size_t inIndex = 0; inIndex < channels; ++inIndex) {
                for (std::size_t w = 0; w < kernelWidth; ++w) {
                    kernel(outIndex, inIndex, w) = outIndex == inIndex ? rawKernel(w) : static_cast<KernelType>(0);
                }
            }
        }
        
        return xt::xtensor<ResultType, 2>(xvigra::convolve1D(input, kernel, kernelOptions));
    }   

    template <typename T, typename O>
    auto separableConvolve1DImplicit(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& rawKernelExpression,
        const xvigra::KernelOptions& kernelOptions
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelContainerType = typename xt::xexpression<O>::derived_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();
        KernelContainerType rawKernel = rawKernelExpression.derived_cast();

        if (input.dimension() != 1) {
            throw std::invalid_argument("separableConvolve1DImplicit(): Need 1 dimensional (W) input!");
        }

        auto normalizedInput = xt::expand_dims(input, input.dimension());

        xvigra::KernelOptions copiedOptions(kernelOptions);
        copiedOptions.setChannelPosition(xvigra::ChannelPosition::LAST);

        auto result = separableConvolve1D(
            normalizedInput, 
            rawKernel, 
            copiedOptions
        );

        return xt::xtensor<ResultType, 1>(xt::squeeze(result));
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve1D - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve2D - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    template <typename ConstantType, typename KernelContainerType, std::size_t N, bool isBegin>
    void calculateNewConstantValue(
        xvigra::KernelOptions& toModify,
        std::size_t currentIndex,
        const std::array<KernelContainerType, N>& rawKernels
    ) {
        ConstantType initialValue;
        if constexpr (isBegin) {
            initialValue = toModify.borderTreatmentBegin.getValue<ConstantType>();
        } else {
            initialValue = toModify.borderTreatmentEnd.getValue<ConstantType>();
        }
        if (initialValue == static_cast<ConstantType>(0)) {
            return;
        }
        ConstantType result{initialValue};

        for (std::size_t i = 0; i < currentIndex; ++i) {
            result = xt::eval(xt::sum(rawKernels[i] * result))[0];
        }

         if constexpr (isBegin) {
            toModify.borderTreatmentBegin = xvigra::BorderTreatment::constant<ConstantType>(result);
        } else {
            toModify.borderTreatmentEnd = xvigra::BorderTreatment::constant<ConstantType>(result);
        }
    }

    template <typename KernelContainerType, typename ConstantType, std::size_t N>
    void updateConstantValueIfNecessary(
        xvigra::KernelOptions& toModify,
        std::size_t currentIndex,
        const std::array<KernelContainerType, N>& rawKernels
    ) {
        auto treatmentBegin = toModify.borderTreatmentBegin;
        auto treatmentEnd = toModify.borderTreatmentEnd;

        if (treatmentBegin.getType() == xvigra::BorderTreatmentType::CONSTANT) {
            calculateNewConstantValue<ConstantType, KernelContainerType, N, true>(
                toModify, 
                currentIndex, 
                rawKernels
            );
        } 

        if (treatmentEnd.getType() == xvigra::BorderTreatmentType::CONSTANT) {
            calculateNewConstantValue<ConstantType, KernelContainerType, N, false>(
                toModify, 
                currentIndex, 
                rawKernels
            );
        }
    }


    template <typename T, typename KernelContainerType>
    auto separableConvolve2D(
        const xt::xexpression<T>& inputExpression,
        const std::array<KernelContainerType, 2>& rawKernelExpressions,
        const std::array<xvigra::KernelOptions, 2>& kernelOptions
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();

        std::size_t numberOfDimensions = input.dimension();

        if (numberOfDimensions != 3) {
            throw std::invalid_argument("separableConvolve2D(): Need 3 dimensional (H x W x C or C x H x W) input!");
        }

        std::size_t locationOfChannel = 0;

        if (kernelOptions[0].channelPosition != kernelOptions[1].channelPosition) {
            throw std::invalid_argument("separableConvolve2D(): ChannelPosition in options for y and x direction are different!");
        }

        xvigra::ChannelPosition channelPosition = kernelOptions[0].channelPosition;

        std::size_t startAxis = 0;
        std::size_t endAxis = numberOfDimensions - 1;

        if (channelPosition == xvigra::ChannelPosition::LAST) {
            locationOfChannel = numberOfDimensions - 1;
            startAxis = 0;
            endAxis = numberOfDimensions - 1;
        } else if (channelPosition == xvigra::ChannelPosition::FIRST) {
            locationOfChannel = 0;
            startAxis = 1;
            endAxis = numberOfDimensions;
        } else {
            throw std::invalid_argument("separableConvolve2D(): ChannelPosition for input can't be IMPLICIT.");
        }
        
        std::size_t channels = input.shape()[locationOfChannel];
     
        xt::xtensor<ResultType, 3> result = xt::cast<ResultType>(input);
        std::array<std::size_t, 3> resultShape = result.shape();

        for (std::size_t currentAxis = startAxis; currentAxis < endAxis; ++currentAxis) {
            std::size_t index = currentAxis - startAxis;
            xvigra::KernelOptions options(kernelOptions[index]);
            updateConstantValueIfNecessary<KernelContainerType, ResultType, 2>(
                options, 
                currentAxis - startAxis, 
                rawKernelExpressions
            );

            KernelContainerType rawKernel = rawKernelExpressions[index];
            std::size_t kernelSize = rawKernel.shape()[0];
            xt::xtensor<KernelType, 3> kernel = xvigra::promoteKernelToFull1D(rawKernel, channels);

            int size = xvigra::calculateOutputSize(resultShape[currentAxis], kernelSize, options);
            resultShape[currentAxis] = static_cast<std::size_t>(size);
            xt::xtensor<ResultType, 3> tmp(resultShape);
            std::size_t maxIndex = xvigra::calculateMaxIndex<3>(
                resultShape, 
                currentAxis, 
                startAxis, 
                endAxis
            );

            for (std::size_t compoundIndex = 0; compoundIndex < maxIndex; ++compoundIndex) {
                xt::xstrided_slice_vector sliceVector = xvigra::decomposeIndex<3>(
                    compoundIndex, 
                    resultShape, 
                    currentAxis, 
                    startAxis,
                    endAxis
                );

                xt::xtensor<ResultType, 2> row(xt::strided_view(result, sliceVector));
                xt::xtensor<ResultType, 2> convolvedRow = xvigra::convolve1D(row, kernel, options);
                xt::strided_view(tmp, sliceVector) = convolvedRow;
            }
            
            result = tmp;
        }

        return result;
    }


    template <typename T, typename KernelContainerType>
    inline auto separableConvolve2D(
        const xt::xexpression<T>& inputExpression,
        const std::array<KernelContainerType, 2>& rawKernelExpressions,
        const xvigra::KernelOptions2D& options
    ) {
        return separableConvolve2D(
            inputExpression, 
            rawKernelExpressions, 
            std::array{options.optionsY, options.optionsX}
        );
    }


    template <typename T, typename O>
    inline auto separableConvolve2D(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& rawKernelExpression,
        const xvigra::KernelOptions2D& options
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelContainerType = typename xt::xexpression<O>::derived_type;
        using KernelType = typename KernelContainerType::value_type;

        xt::xarray<InputType> input = inputExpression.derived_cast();
        xt::xarray<KernelType> rawKernel = rawKernelExpression.derived_cast();

        return separableConvolve2D(
            input, 
            std::array{rawKernel, rawKernel}, 
            std::array{options.optionsY, options.optionsX}
        );
    }


    template <typename T, typename KernelContainerType>
    auto separableConvolve2DImplicit(
        const xt::xexpression<T>& inputExpression,
        const std::array<KernelContainerType, 2>& rawKernelExpressions,
        const std::array<xvigra::KernelOptions, 2>& kernelOptions
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();

        if (input.dimension() != 2) {
            throw std::invalid_argument("separableConvolve2DImplicit(): Need 2 dimensional (H x W) input!");
        }

        auto normalizedInput = xt::expand_dims(inputExpression.derived_cast(), input.dimension());

        std::array<xvigra::KernelOptions, 2> copiedOptions;
        for (std::size_t i = 0; i < copiedOptions.size(); ++i) {
            copiedOptions[i] = xvigra::KernelOptions(kernelOptions[i]);
            copiedOptions[i].setChannelPosition(xvigra::ChannelPosition::LAST);
        }

        auto result = separableConvolve2D(
            normalizedInput, 
            rawKernelExpressions, 
            copiedOptions
        );

        return xt::xtensor<ResultType, 2>(xt::squeeze(result));
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve2D - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolveND - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <std::size_t N, typename T, typename KernelContainerType>
    auto separableConvolveND(
        const xt::xexpression<T>& inputExpression,
        const std::array<KernelContainerType, N>& rawKernels,
        const std::array<xvigra::KernelOptions, N>& kernelOptions
    ) {
        for (std::size_t i = 0; i < N - 1; ++i) {
            if (kernelOptions[i].channelPosition != kernelOptions[i + 1].channelPosition) {
                throw std::invalid_argument("separableConvolveND(): Given options don't contain a consistent ChannelPosition!");
            }
        }

        xvigra::ChannelPosition channelPosition = kernelOptions[0].channelPosition;

        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();

        std::size_t numberOfDimensions = input.dimension();

        if (numberOfDimensions != N + 1) {
            throw std::invalid_argument("separableConvolveND(): Number of dimensions of input does not match the given non-channel dimension template parameter!");
        }

        std::size_t locationOfChannel = 0;
        std::size_t startAxis = 0;
        std::size_t endAxis = numberOfDimensions - 1;

        if (channelPosition == xvigra::ChannelPosition::LAST) {
            locationOfChannel = numberOfDimensions - 1;
            startAxis = 0;
            endAxis = numberOfDimensions - 1;
        } else if (channelPosition == xvigra::ChannelPosition::FIRST) {
            locationOfChannel = 0;
            startAxis = 1;
             endAxis = numberOfDimensions;
        } else {
            throw std::invalid_argument("separableConvolveND(): ChannelPosition for input can't be IMPLICIT.");
        }
        
        std::size_t channels = input.shape()[locationOfChannel];
     
        xt::xtensor<ResultType, N + 1> result(input);
        std::array<std::size_t, N + 1> resultShape = result.shape();

        
        for (std::size_t currentAxis = startAxis; currentAxis < endAxis; ++currentAxis) {
            std::size_t index = currentAxis - startAxis;
            xvigra::KernelOptions options(kernelOptions[index]);

            updateConstantValueIfNecessary<KernelContainerType, ResultType, N>(
                options, 
                currentAxis - startAxis, 
                rawKernels
            );

            xt::xtensor<KernelType, 1> rawKernel = rawKernels[index];
            std::size_t kernelSize = rawKernel.shape()[0];
            xt::xtensor<KernelType, 3> kernel = xvigra::promoteKernelToFull1D(rawKernel, channels);

            int size = xvigra::calculateOutputSize(resultShape[currentAxis], kernelSize, options);
            resultShape[currentAxis] = static_cast<std::size_t>(size);

            xt::xtensor<ResultType, N + 1> tmp(resultShape);
            std::size_t maxIndex = xvigra::calculateMaxIndex<N + 1>(resultShape, currentAxis, startAxis, endAxis);

            for (std::size_t compoundIndex = 0; compoundIndex < maxIndex; ++compoundIndex) {
                xt::xstrided_slice_vector sliceVector = xvigra::decomposeIndex<N + 1>(compoundIndex, resultShape, currentAxis, startAxis, endAxis);

                xt::xtensor<ResultType, 2> row(xt::strided_view(result, sliceVector));
                xt::xtensor<ResultType, 2> convolvedRow = xvigra::convolve1D(row, kernel, options);
                xt::strided_view(tmp, sliceVector) = convolvedRow;
            }

            result = tmp;
        }

        return result;
    }   

    template <std::size_t N, typename T, typename KernelContainerType>
    auto separableConvolveNDImplicit(
        const xt::xexpression<T>& inputExpression,
        const std::array<KernelContainerType, N>& rawKernels,
        const std::array<xvigra::KernelOptions, N>& kernelOptions
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();

        auto normalizedInput = xt::expand_dims(input, input.dimension());

        std::array<xvigra::KernelOptions, N> copiedOptions;
        for (std::size_t i = 0; i < N; ++i) {
            copiedOptions[i] = xvigra::KernelOptions(kernelOptions[i]);
            copiedOptions[i].setChannelPosition(xvigra::ChannelPosition::LAST);
        }

        auto result = separableConvolveND<N>(
            normalizedInput, 
            rawKernels, 
            copiedOptions
        );

        return xt::xtensor<ResultType, N>(xt::squeeze(result));
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolveND - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

     // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolveND - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <std::size_t N, typename T, typename KernelContainerType>
    auto separableConvolve(
        const xt::xexpression<T>& inputExpression,
        const std::array<KernelContainerType, N>& rawKernels,
        const std::array<xvigra::KernelOptions, N>& kernelOptions
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();

        if constexpr (N == 1) {
            return separableConvolve1D(input, rawKernels[0], kernelOptions[0]);
        } else if constexpr (N == 2) {
            return separableConvolve2D(input, rawKernels, kernelOptions);
        } else {
            return separableConvolveND(input, rawKernels, kernelOptions);
        }

    }   

    template <std::size_t N, typename T, typename KernelContainerType>
    auto separableConvolveImplicit(
        const xt::xexpression<T>& inputExpression,
        const std::array<KernelContainerType, N>& rawKernels,
        const std::array<xvigra::KernelOptions, N>& kernelOptions
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();

        auto normalizedInput = xt::expand_dims(input, input.dimension());

        std::array<xvigra::KernelOptions, N> copiedOptions;
        for (std::size_t i = 0; i < N; ++i) {
            copiedOptions[i] = xvigra::KernelOptions(kernelOptions[i]);
            copiedOptions[i].setChannelPosition(xvigra::ChannelPosition::LAST);
        }

        auto result = separableConvolve<N>(
            normalizedInput, 
            rawKernels, 
            copiedOptions
        );

        return xt::xtensor<ResultType, N>(xt::squeeze(result));
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolveND - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

}

#endif