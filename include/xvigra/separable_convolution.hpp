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
    #define CONVOLVE_ALONG_AXIS(currentAxis) {                                                         \
            std::size_t index = (currentAxis) - startAxis;                                             \
            xvigra::KernelOptions options(kernelOptions[index]);                                       \
            updateConstantValueIfNecessary<KernelContainerType, ResultType, 2>(                        \
                options,                                                                               \
                currentAxis - startAxis,                                                               \
                rawKernelExpressions                                                                   \
            );                                                                                         \
                                                                                                       \
            KernelContainerType rawKernel = rawKernelExpressions[index];                               \
            std::size_t kernelSize = rawKernel.shape()[0];                                             \
            xt::xtensor<KernelType, 3> kernel = xvigra::promoteKernelToFull1D(rawKernel, channels);    \
                                                                                                       \
            int size = xvigra::calculateOutputSize(resultShape[(currentAxis)], kernelSize, options);   \
            resultShape[(currentAxis)] = static_cast<std::size_t>(size);                               \
            xt::xtensor<ResultType, 3> tmp(resultShape);                                               \
            std::size_t maxIndex = xvigra::calculateMaxIndex<3>(                                       \
                resultShape,                                                                           \
                (currentAxis),                                                                         \
                startAxis,                                                                             \
                endAxis                                                                                \
            );                                                                                         \
                                                                                                       \
            for (std::size_t compoundIndex = 0; compoundIndex < maxIndex; ++compoundIndex) {           \
                xt::xstrided_slice_vector sliceVector = xvigra::decomposeIndex<3>(                     \
                    compoundIndex,                                                                     \
                    resultShape,                                                                       \
                    (currentAxis),                                                                     \
                    startAxis,                                                                         \
                    endAxis                                                                            \
                );                                                                                     \
                                                                                                       \
                xt::xtensor<ResultType, 2> row(xt::strided_view(result, sliceVector));                 \
                xt::xtensor<ResultType, 2> convolvedRow = xvigra::convolve1D(row, kernel, options);    \
                xt::strided_view(tmp, sliceVector) = convolvedRow;                                     \
            }                                                                                          \
                                                                                                       \
            result = tmp;                                                                              \
                                                                                                       \
    }

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

    /*
     * <p>
     * Generates a slice vector containing xt::all() at position Dim and fills all other entries with the
     * corresponding decomposed parts of the given compound index.
     * </p>
     *
     * @tparam Dim number of dimensions in the shape.
     * @param compoundIndex number of dimensions in the shape.
     * @param shape the desired shape of the slice vector
     * @param currentAxis axis which should be ignored.
     * @param startAxis (inclusive) the first axis that should be used.
     * @param endAxis (exclusive) the last axis that should be used.
     * @return a slice vector containing the decomposed compound indices and xt::all at position Dim.
     */
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
            if (i != currentAxis) {
                auto maxIndex = xvigra::calculateMaxIndex<Dim>(shape, currentAxis, i + 1, endAxis);
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

    /*
     * <p>
     * Calculates the 1-dimensional separable convolution of the input with the given 1-dimensional kernel based on xvigra::convolve1D.
     * This function requires an input of shape  W x C or C x W and a kernel with at least 1 dimension or at maximum
     * a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::FIRST or ChannelPosition::LAST inputs; for ChannelPosition::IMPLICIT
     * use xvigra::separableConvolve1DImplicit.
     * </p>
     *
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpression xexpression containing the kernel data
     * @param kernelOptions object containing information about padding, stride, dilation, channel position and border
                            treatment
     * @return the result of the 1-dimensional convolution between the input and kernel as xt::xtensor
     * @throws std::invalid_argument if input does not match the required shape or if IMPLICIT channel position is
                                     requested.
     */
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
            throw std::invalid_argument("separableConvolve1D(): Need 2 dimensional (W x C or C x W) input!");
        }

        std::size_t locationOfChannel = 0;

        if (kernelOptions.channelPosition == xvigra::ChannelPosition::LAST) {
            locationOfChannel = numberOfDimensions - 1;
        } else if (kernelOptions.channelPosition == xvigra::ChannelPosition::FIRST) {
            locationOfChannel = 0;
        } else {
            throw std::invalid_argument("separableConvolve1D(): ChannelPosition for input can't be IMPLICIT.");
        }

        std::size_t channels = input.shape()[locationOfChannel];
        xt::xtensor<KernelType, 3> kernel = xvigra::promoteKernelToFull1D(rawKernel, channels);

        return xt::xtensor<ResultType, 2>(xvigra::convolve1D(input, kernel, kernelOptions));
    }

    /*
     * <p>
     * Calculates the 1-dimensional separable convolution of the input with the given 1-dimensional kernel based on xvigra::convolve1D.
     * This function requires an input of shape W and a kernel with at least 1 dimension or at maximum
     * a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::IMPLICIT inputs; for ChannelPosition::FIRST or ChannelPosition::LAST
     * use xvigra::separableConvolve1D.
     * </p>
     *
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpression xexpression containing the kernel data
     * @param kernelOptions object containing information about padding, stride, dilation, channel position and border
                            treatment
     * @return the result of the 1-dimensional convolution between the input and kernel as xt::xtensor
     * @throws std::invalid_argument if input does not match the required shape
     */
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

    /*
     * <p>
     * Calculates the 2-dimensional separable convolution of the input with the given 1-dimensional kernels based on
     * xvigra::convolve1D.
     * This function requires an input of shape H x W x C or C x H x W and two kernels with at least 1 dimension or at maximum
     * a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::FIRST or ChannelPosition::LAST inputs; for ChannelPosition::IMPLICIT
     * use xvigra::separableConvolve2DImplicit.
     * </p>
     *
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpressions array with two 1-dimensional kernels
     * @param kernelOptions array of options for each dimension containing independent information about padding, stride,
                            dilation and border treatment
     * @return the result of the 2-dimensional convolution between the input and 1-dimensional kernels as xt::xtensor
     * @throws std::invalid_argument if input does not match the required shape or if IMPLICIT channel position is
                                     requested.
     */
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

        if (input.dimension() != 3) {
            throw std::invalid_argument("separableConvolve2D(): Need 3 dimensional (H x W x C or C x H x W) input!");
        }

        if (kernelOptions[0].channelPosition != kernelOptions[1].channelPosition) {
            throw std::invalid_argument("separableConvolve2D(): ChannelPosition in options for y and x direction are different!");
        }

        xvigra::ChannelPosition channelPosition = kernelOptions[0].channelPosition;

        std::size_t locationOfChannel = 2;
        std::size_t startAxis = 0;
        std::size_t endAxis = 2;

        if (channelPosition == xvigra::ChannelPosition::FIRST) {
            locationOfChannel = 0;
            startAxis = 1;
            endAxis = 3;
        } else if (channelPosition != xvigra::ChannelPosition::LAST) {
            throw std::invalid_argument("separableConvolve2D(): ChannelPosition for input can't be IMPLICIT.");
        }

        std::size_t channels = input.shape()[locationOfChannel];

        xt::xtensor<ResultType, 3> result = xt::cast<ResultType>(input);
        std::array<std::size_t, 3> resultShape = result.shape();

        CONVOLVE_ALONG_AXIS(startAxis)
        CONVOLVE_ALONG_AXIS(endAxis - 1)

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

    /*
     * <p>
     * Calculates the 2-dimensional separable convolution of the input with the given 1-dimensional kernels based on xvigra::convolve1D.
     * This function requires an input of shape W and two kernels with at least 2 dimension or at maximum
     * a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::IMPLICIT inputs; for ChannelPosition::FIRST or ChannelPosition::LAST
     * use xvigra::separableConvolve2D.
     * </p>
     *
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpressions array with two 1-dimensional kernels
     * @param kernelOptions array of options for each dimension containing independent information about padding, stride,
                            dilation and border treatment
     * @return the result of the 2-dimensional convolution between the input and 1-dimensional kernels as xt::xtensor
     * @throws std::invalid_argument if input does not match the required shape
     */
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

     /*
     * <p>
     * Calculates the N-dimensional separable convolution of the input with the given 1-dimensional kernels based on
     * xvigra::convolve1D.
     * This function requires an input of shape D_N x ... x D_1 x C or C x D_N x ... x D_1 and N kernels with at least 1
     * dimension or at maximum a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::FIRST or ChannelPosition::LAST inputs; for ChannelPosition::IMPLICIT
     * use xvigra::separableConvolveNDImplicit.
     * </p>
     *
     * @tparam N number non-channel dimensions in the input
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpressions array with N 1-dimensional kernels
     * @param kernelOptions array of options for each dimension containing independent information about padding, stride,
                            dilation and border treatment
     * @return the result of the N-dimensional convolution between the input and 1-dimensional kernels as xt::xtensor
     * @throws std::invalid_argument if input does not match the required shape or if IMPLICIT channel position is
                                     requested.
     */
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

    /*
     * <p>
     * Calculates the N-dimensional separable convolution of the input with the given 1-dimensional kernels based on xvigra::convolve1D.
     * This function requires an input of shape D_N x ... x D_1 x C or C x D_N x ... x D_1 and N kernels with at least
     * 2 dimension or at maximum a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::IMPLICIT inputs; for ChannelPosition::FIRST or ChannelPosition::LAST
     * use xvigra::separableConvolveND.
     * </p>
     *
     * @tparam N number non-channel dimensions in the input
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpressions array with N 1-dimensional kernels
     * @param kernelOptions array of options for each dimension containing independent information about padding, stride,
                            dilation and border treatment
     * @return the result of the N-dimensional convolution between the input and 1-dimensional kernels as xt::xtensor
     * @throws std::invalid_argument if input does not match the required shape
     */
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
    // ║ separableConvolve - begin                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    /*
     * <p>
     * Calculates the N-dimensional separable convolution of the input with the given 1-dimensional kernels based on
     * xvigra::separableConvolve1D, xvigra::separableConvolve2D and xvigra::separableConvolveND.
     * This function requires an input of shape D_N x ... x D_1 x C or C x D_N x ... x D_1 and N kernels with at least 1
     * dimension or at maximum a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::FIRST or ChannelPosition::LAST inputs; for ChannelPosition::IMPLICIT
     * use xvigra::separableConvolveImplicit.
     * </p>
     *
     * @tparam N number non-channel dimensions in the input
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpressions array with N 1-dimensional kernels
     * @param kernelOptions array of options for each dimension containing independent information about padding, stride,
                            dilation and border treatment
     * @return the result of the N-dimensional convolution between the input and 1-dimensional kernels as xt::xtensor
     * @throws std::invalid_argument if input does not match the required shape or if IMPLICIT channel position is
                                     requested.
     */
    template <std::size_t N, typename T, typename KernelContainerType>
    auto separableConvolve(
        const xt::xexpression<T>& inputExpression,
        const std::array<KernelContainerType, N>& rawKernels,
        const std::array<xvigra::KernelOptions, N>& kernelOptions
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;

        InputContainerType input = inputExpression.derived_cast();

        if constexpr (N == 1) {
            return separableConvolve1D(input, std::get<0>(rawKernels), std::get<0>(kernelOptions));
        } else if constexpr (N == 2) {
            return separableConvolve2D(input, rawKernels, kernelOptions);
        } else {
            return separableConvolveND(input, rawKernels, kernelOptions);
        }

    }

    /*
     * <p>
     * Calculates the N-dimensional separable convolution of the input with the given 1-dimensional kernels based on
     * xvigra::separableConvolve1D, xvigra::separableConvolve2D and xvigra::separableConvolveND.
     * This function requires an input of shape D_N x ... x D_1 x C or C x D_N x ... x D_1 and N kernels with at least
     * 2 dimension or at maximum a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::IMPLICIT inputs; for ChannelPosition::FIRST or ChannelPosition::LAST
     * use xvigra::separableConvolve.
     * </p>
     *
     * @tparam N number non-channel dimensions in the input
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpressions array with N 1-dimensional kernels
     * @param kernelOptions array of options for each dimension containing independent information about padding, stride,
                            dilation and border treatment
     * @return the result of the N-dimensional convolution between the input and 1-dimensional kernels as xt::xtensor
     * @throws std::invalid_argument if input does not match the required shape
     */
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
    // ║ separableConvolve - end                                                                                      ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

}

#endif
