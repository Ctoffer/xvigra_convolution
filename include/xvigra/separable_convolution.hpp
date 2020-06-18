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
#include "xvigra/io_util.hpp"

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

    template <typename InputType, typename KernelType, xvigra::ChannelPosition channelPosition>
    xt::xtensor<typename std::common_type_t<InputType, KernelType>, 2> separableConvolve1D(
        const xt::xtensor<InputType, 2>& input,
        const xt::xtensor<KernelType, 1>& rawKernel,
        const xvigra::KernelOptions& kernelOptions
    ) {
        using ResultType = typename std::common_type_t<InputType, KernelType>;
        auto inputShape = input.shape();
        std::size_t kernelWidth = rawKernel.shape()[0];
        xt::xtensor<KernelType, 3> kernel(xt::reshape_view(rawKernel, {1, 1, static_cast<int>(kernelWidth)}));
        xvigra::KernelOptions options(kernelOptions);
        options.channelPosition = xvigra::ChannelPosition::IMPLICIT;

        std::size_t numberOfDimensions = input.dimension();
        std::size_t locationOfChannel = 0;

        if constexpr (channelPosition == xvigra::ChannelPosition::LAST) {
            locationOfChannel = numberOfDimensions - 1;
        } else if constexpr (channelPosition == xvigra::ChannelPosition::FIRST) {
            locationOfChannel = 0;
        } else {
            throw std::invalid_argument("separableConvolve1D(): ChannelPosition for input can't be IMPLICIT.");
        }
        
        std::size_t channels = inputShape[locationOfChannel];

        std::array<std::size_t, 2> resultShape;
        if constexpr (channelPosition == xvigra::ChannelPosition::LAST) {
            int size = xvigra::calculateOutputSize(inputShape[0], kernelWidth, options);
            resultShape[0] = static_cast<std::size_t>(size);
            resultShape[1] = channels;
        } else {
            int size = xvigra::calculateOutputSize(inputShape[1], kernelWidth, options);
            resultShape[0] = channels;
            resultShape[1] = static_cast<std::size_t>(size);
        }
        xt::xtensor<ResultType, 2> result(resultShape);

        for (std::size_t channel = 0; channel < channels; ++channel) {
            xt::xstrided_slice_vector sliceVector(2, xt::all());
            if constexpr (channelPosition == xvigra::ChannelPosition::LAST) {
                sliceVector[1] = channel;
            } else {
                sliceVector[0] = channel;
            }

            xt::xtensor<InputType, 1> row(xt::strided_view(input, sliceVector));
            auto convolvedRow = xvigra::convolve1DImplicit(row, kernel, options);
            xt::strided_view(result, sliceVector) = convolvedRow;
        }

        return result;
    }   

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve1D - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve2D - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    template <typename ConstantType, typename KernelType, std::size_t N, bool isBegin>
    void calculateNewConstantValue(
        xvigra::KernelOptions& toModify,
        std::size_t currentIndex,
        const std::array<xt::xtensor<KernelType, 1>, N>& rawKernels
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

    template <typename KernelType, typename ConstantType, std::size_t N>
    void updateConstantValueIfNecessary(
        xvigra::KernelOptions& toModify,
        std::size_t currentIndex,
        const std::array<xt::xtensor<KernelType, 1>, N>& rawKernels
    ) {
        auto treatmentBegin = toModify.borderTreatmentBegin;
        auto treatmentEnd = toModify.borderTreatmentEnd;

        if (treatmentBegin.getType() == xvigra::BorderTreatmentType::CONSTANT) {
            calculateNewConstantValue<ConstantType, KernelType, N, true>(
                toModify, 
                currentIndex, 
                rawKernels
            );
        } 

        if (treatmentEnd.getType() == xvigra::BorderTreatmentType::CONSTANT) {
            calculateNewConstantValue<ConstantType, KernelType, N, false>(
                toModify, 
                currentIndex, 
                rawKernels
            );
        }
    }


    template <typename InputType, typename KernelType, xvigra::ChannelPosition channelPosition>
    xt::xtensor<typename std::common_type_t<InputType, KernelType>, 3> separableConvolve2D(
        const xt::xtensor<InputType, 3>& input,
        const std::array<xt::xtensor<KernelType, 1>, 2>& rawKernels,
        const std::array<xvigra::KernelOptions, 2>& kernelOptions
    ) {
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        std::size_t numberOfDimensions = input.shape().size();
        std::size_t locationOfChannel = 0;

        if constexpr (channelPosition == xvigra::ChannelPosition::LAST) {
            locationOfChannel = numberOfDimensions - 1;
        } else if constexpr (channelPosition == xvigra::ChannelPosition::FIRST) {
            locationOfChannel = 0;
        } else {
            throw std::invalid_argument("ChannelPosition for input can't be IMPLICIT.");
        }
        
        std::size_t channels = input.shape()[locationOfChannel];

        std::size_t startAxis = 0;
        std::size_t endAxis = numberOfDimensions - 1;

        if constexpr (channelPosition == xvigra::ChannelPosition::LAST) {
            startAxis = 0;
            endAxis = numberOfDimensions - 1;
        } else {
             startAxis = 1;
             endAxis = numberOfDimensions;
        }
     
        xt::xtensor<ResultType, 3> result = xt::cast<ResultType>(input);
        std::array<std::size_t, 3> resultShape = result.shape();

        for (std::size_t currentAxis = startAxis; currentAxis < endAxis; ++currentAxis) {
            std::size_t index = currentAxis - startAxis;
            xvigra::KernelOptions options(kernelOptions[index]);
            options.channelPosition = xvigra::ChannelPosition::IMPLICIT;
            updateConstantValueIfNecessary<KernelType, ResultType, 2>(
                options, 
                currentAxis - startAxis, 
                rawKernels
            );

            xt::xtensor<KernelType, 1> rawKernel = rawKernels[index];
            std::size_t kernelSize = rawKernel.shape()[0];
            std::vector<int> kernelShape{1, 1, static_cast<int>(kernelSize)};
            xt::xtensor<KernelType, 3> kernel(xt::reshape_view(rawKernel, kernelShape));

            int size = xvigra::calculateOutputSize(resultShape[currentAxis], kernelSize, options);
            resultShape[currentAxis] = static_cast<std::size_t>(size);

            xt::xtensor<ResultType, 3> tmp(resultShape);
            std::size_t maxIndex = xvigra::calculateMaxIndex<3>(
                resultShape, 
                currentAxis, 
                startAxis, 
                endAxis
            );

            for (std::size_t channel = 0; channel < channels; ++channel) {
                for (std::size_t compoundIndex = 0; compoundIndex < maxIndex; ++compoundIndex) {
                    xt::xstrided_slice_vector sliceVector = xvigra::decomposeIndex<3>(
                        compoundIndex, 
                        resultShape, 
                        currentAxis, 
                        startAxis,
                        endAxis
                    );
                    sliceVector[locationOfChannel] = channel;

                    xt::xtensor<ResultType, 1> row(xt::strided_view(result, sliceVector));
                    xt::xtensor<ResultType, 1> convolvedRow = xvigra::convolve1DImplicit(row, kernel, options);
                    xt::strided_view(tmp, sliceVector) = convolvedRow;
                }
            }

            result = tmp;
        }

        return result;
    }


    template <typename InputType, typename KernelType, xvigra::ChannelPosition channelPosition>
    xt::xtensor<typename std::common_type_t<InputType, KernelType>, 3> separableConvolve2D(
        const xt::xtensor<InputType, 3>& input,
        const xt::xtensor<KernelType, 1>& rawKernel,
        const xvigra::KernelOptions& options
    ) {
        return separableConvolve2D<InputType, KernelType, channelPosition>(
            input, 
            {rawKernel, rawKernel}, 
            {options, options}
        );
    }


    template <typename InputType, typename KernelType, xvigra::ChannelPosition channelPosition>
    xt::xtensor<typename std::common_type_t<InputType, KernelType>, 3> separableConvolve2D(
        const xt::xtensor<InputType, 3>& input,
        const xt::xtensor<KernelType, 1>& rawKernel,
        const xvigra::KernelOptions2D& options
    ) {
        return separableConvolve2D<InputType, KernelType, channelPosition>(
            input, 
            {rawKernel, rawKernel}, 
            {options.optionsY, options.optionsX}
        );
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve2D - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve - begin                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename InputType, typename KernelType, xvigra::ChannelPosition channelPosition, std::size_t N>
    xt::xtensor<typename std::common_type_t<InputType, KernelType>, N + 1> separableConvolve(
        const xt::xtensor<InputType, N + 1>& input,
        const std::array<xt::xtensor<KernelType, 1>, N>& rawKernels,
        const std::array<xvigra::KernelOptions, N>& kernelOptions
    ) {
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        std::size_t numberOfDimensions = input.shape().size();
        std::size_t locationOfChannel = 0;

        if constexpr (channelPosition == xvigra::ChannelPosition::LAST) {
            locationOfChannel = numberOfDimensions - 1;
        } else if constexpr (channelPosition == xvigra::ChannelPosition::FIRST) {
            locationOfChannel = 0;
        } else {
            throw std::invalid_argument("ChannelPosition for input can't be IMPLICIT.");
        }
        
        std::size_t channels = input.shape()[locationOfChannel];

        std::size_t startAxis = 0;
        std::size_t endAxis = numberOfDimensions - 1;

        if constexpr (channelPosition == xvigra::ChannelPosition::LAST) {
            startAxis = 0;
            endAxis = numberOfDimensions - 1;
        } else {
             startAxis = 1;
             endAxis = numberOfDimensions;
        }
     
        xt::xtensor<ResultType, N + 1> result(input);
        std::array<std::size_t, N + 1> resultShape = result.shape();

        for (std::size_t currentAxis = startAxis; currentAxis < endAxis; ++currentAxis) {
            std::size_t index = currentAxis - startAxis;
            xvigra::KernelOptions options(kernelOptions[index]);
            options.channelPosition = xvigra::ChannelPosition::IMPLICIT;
            updateConstantValueIfNecessary<KernelType, ResultType, N>(
                options, 
                currentAxis - startAxis, 
                rawKernels
            );

            xt::xtensor<KernelType, 1> rawKernel = rawKernels[index];
            std::size_t kernelSize = rawKernel.shape()[0];
            std::vector<int> kernelShape{1, 1, static_cast<int>(kernelSize)};
            xt::xtensor<KernelType, 3> kernel(xt::reshape_view(rawKernel, kernelShape));

            int size = xvigra::calculateOutputSize(resultShape[currentAxis], kernelSize, options);
            resultShape[currentAxis] = static_cast<std::size_t>(size);

            xt::xtensor<ResultType, N + 1> tmp(resultShape);
            std::size_t maxIndex = xvigra::calculateMaxIndex<N + 1>(resultShape, currentAxis, startAxis, endAxis);

            for (std::size_t channel = 0; channel < channels; ++channel) {
                for (std::size_t compoundIndex = 0; compoundIndex < maxIndex; ++compoundIndex) {
                    xt::xstrided_slice_vector sliceVector = xvigra::decomposeIndex<N + 1>(compoundIndex, resultShape, currentAxis, startAxis, endAxis);
                    sliceVector[locationOfChannel] = channel;

                    xt::xtensor<ResultType, 1> row(xt::strided_view(result, sliceVector));
                    xt::xtensor<ResultType, 1> convolvedRow = xvigra::convolve1DImplicit(row, kernel, options);
                    xt::strided_view(tmp, sliceVector) = convolvedRow;
                }
            }

            result = tmp;
        }

        return result;
    }   

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ separableConvolve - end                                                                                      ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

}

#endif