#ifndef XVIGRA_EXPLICIT_CONVOLUTION_HPP
#define XVIGRA_EXPLICIT_CONVOLUTION_HPP

#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xbuilder.hpp"
#include "xtensor/xtensor.hpp"

#include "xtensor-blas/xlinalg.hpp"

#include "xvigra/convolution_util.hpp"
#include "xvigra/iter_util.hpp"
#include "xvigra/kernel_util.hpp"

namespace xvigra {
    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ type definitions - begin                                                                                         ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T>
    using Tensor1D = typename xt::xtensor<T, 1>;
    template <typename T>
    using Tensor2D = typename xt::xtensor<T, 2>;
    template <typename T>
    using Tensor3D = typename xt::xtensor<T, 3>;
    template <typename T>
    using Tensor5D = typename xt::xtensor<T, 5>;

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ type definitions - end                                                                                           ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ utility - begin                                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    #define XVIGRA_SET_BEGIN_BORDER_VALUE_CHANNEL_FIRST {                                   \
        InputType value;                                                                    \
        auto treatment = options.borderTreatmentBegin;                                      \
                                                                                            \
        switch(treatment.getType()) {                                                       \
            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {                         \
                value = input(inputChannel, std::abs(inputX));                              \
                break;                                                                      \
            }                                                                               \
            case xvigra::BorderTreatmentType::AVOID: {                                      \
                throw std::domain_error(                                                    \
                    "convolve1D(): Border treatment AVOID should not be used here!"         \
                );                                                                          \
            }                                                                               \
            case xvigra::BorderTreatmentType::CONSTANT: {                                   \
                value = treatment.getValue<InputType>();                                    \
                break;                                                                      \
            }                                                                               \
            case xvigra::BorderTreatmentType::REPEAT: {                                     \
                value = input(inputChannel, 0);                                             \
                break;                                                                      \
            }                                                                               \
            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {                          \
                value = input(inputChannel, std::abs(inputX + 1));                          \
                break;                                                                      \
            }                                                                               \
            case xvigra::BorderTreatmentType::WRAP: {                                       \
                value = input(inputChannel, inputX + inputWidth);                           \
                break;                                                                      \
            }                                                                               \
            default: {                                                                      \
                throw std::domain_error("convolve1D(): Unknown begin border treatment!");   \
            }                                                                               \
        }                                                                                   \
                                                                                            \
        patch(inputChannel, patchKernelX, outIndex) = static_cast<ResultType>(value);       \
    }

    #define XVIGRA_SET_END_BORDER_VALUE_CHANNEL_FIRST {                                   \
        InputType value;                                                                  \
        auto treatment = options.borderTreatmentEnd;                                      \
                                                                                          \
        switch(treatment.getType()) {                                                     \
            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {                       \
                value = input(inputChannel, 2 * inputWidth - inputX - 2);                 \
                break;                                                                    \
            }                                                                             \
            case xvigra::BorderTreatmentType::AVOID: {                                    \
                throw std::domain_error(                                                  \
                    "convolve1D(): Border treatment AVOID should not be used here!"       \
                );                                                                        \
            }                                                                             \
            case xvigra::BorderTreatmentType::CONSTANT: {                                 \
                value = treatment.getValue<InputType>();                                  \
                break;                                                                    \
            }                                                                             \
            case xvigra::BorderTreatmentType::REPEAT: {                                   \
                value = input(inputChannel, inputWidth - 1);                              \
                break;                                                                    \
            }                                                                             \
            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {                        \
                value = input(inputChannel, 2 * inputWidth - inputX - 1);                 \
                break;                                                                    \
            }                                                                             \
            case xvigra::BorderTreatmentType::WRAP: {                                     \
                value = input(inputChannel, inputX - inputWidth);                         \
                break;                                                                    \
            }                                                                             \
            default: {                                                                    \
                throw std::domain_error("convolve1D(): Unknown end border treatment!");   \
            }                                                                             \
        }                                                                                 \
                                                                                          \
        patch(inputChannel, patchKernelX, outIndex) = static_cast<ResultType>(value);     \
    }

    #define XVIGRA_SET_BEGIN_BORDER_VALUE_CHANNEL_LAST {                                    \
        InputType value;                                                                    \
        auto treatment = options.borderTreatmentBegin;                                      \
                                                                                            \
        switch(treatment.getType()) {                                                       \
            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {                         \
                value = input(std::abs(inputX + 1), inputChannel);                          \
                break;                                                                      \
            }                                                                               \
            case xvigra::BorderTreatmentType::AVOID: {                                      \
                throw std::domain_error(                                                    \
                    "convolve1D(): Border treatment AVOID should not be used here!"         \
                );                                                                          \
            }                                                                               \
            case xvigra::BorderTreatmentType::CONSTANT: {                                   \
                value = treatment.getValue<InputType>();                                    \
                break;                                                                      \
            }                                                                               \
            case xvigra::BorderTreatmentType::REPEAT: {                                     \
                value = input(0, inputChannel);                                             \
                break;                                                                      \
            }                                                                               \
            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {                          \
                value = input(std::abs(inputX), inputChannel);                              \
                break;                                                                      \
            }                                                                               \
            case xvigra::BorderTreatmentType::WRAP: {                                       \
                value = input(inputX + inputWidth, inputChannel);                           \
                break;                                                                      \
            }                                                                               \
            default: {                                                                      \
                throw std::domain_error("convolve1D(): Unknown begin border treatment!");   \
            }                                                                               \
        }                                                                                   \
                                                                                            \
        patch(outIndex, inputChannel, patchKernelX) = static_cast<ResultType>(value);       \
    }

    #define XVIGRA_SET_END_BORDER_VALUE_CHANNEL_LAST {                                    \
        InputType value;                                                                  \
        auto treatment = options.borderTreatmentEnd;                                      \
                                                                                          \
        switch(treatment.getType()) {                                                     \
            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {                       \
                value = input(2 * inputWidth - inputX - 2, inputChannel);                 \
                break;                                                                    \
            }                                                                             \
            case xvigra::BorderTreatmentType::AVOID: {                                    \
                throw std::domain_error(                                                  \
                    "convolve1D(): Border treatment AVOID should not be used here!"       \
                );                                                                        \
            }                                                                             \
            case xvigra::BorderTreatmentType::CONSTANT: {                                 \
                value = treatment.getValue<InputType>();                                  \
                break;                                                                    \
            }                                                                             \
            case xvigra::BorderTreatmentType::REPEAT: {                                   \
                value = input(inputWidth - 1, inputChannel);                              \
                break;                                                                    \
            }                                                                             \
            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {                        \
                value = input(2 * inputWidth - inputX - 1, inputChannel);                 \
                break;                                                                    \
            }                                                                             \
            case xvigra::BorderTreatmentType::WRAP: {                                     \
                value = input(inputX - inputWidth, inputChannel);                         \
                break;                                                                    \
            }                                                                             \
            default: {                                                                    \
                throw std::domain_error("convolve1D(): Unknown end border treatment!");   \
            }                                                                             \
        }                                                                                 \
                                                                                          \
        patch(outIndex, inputChannel, patchKernelX) = static_cast<ResultType>(value);     \
    }

    #define XVIGRA_GET_BEGIN_BORDER_INDEX(value, treatment, index, size) {                      \
        switch((treatment).getType()) {                                                         \
            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {                             \
                (value) = std::abs((index));                                                    \
                break;                                                                          \
            }                                                                                   \
            case xvigra::BorderTreatmentType::AVOID: {                                          \
                throw std::domain_error(                                                        \
                    "getBorderIndex(): Border treatment AVOID should not be used here!"         \
                );                                                                              \
            }                                                                                   \
            case xvigra::BorderTreatmentType::CONSTANT: {                                       \
                (value) = -1;                                                                   \
                break;                                                                          \
            }                                                                                   \
            case xvigra::BorderTreatmentType::REPEAT: {                                         \
                (value) = 0;                                                                    \
                break;                                                                          \
            }                                                                                   \
            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {                              \
                (value) = std::abs((index) + 1);                                                \
                break;                                                                          \
            }                                                                                   \
            case xvigra::BorderTreatmentType::WRAP: {                                           \
                (value) = (index) + (size);                                                     \
                break;                                                                          \
            }                                                                                   \
            default: {                                                                          \
                throw std::domain_error("getBorderIndex(): Unknown begin border treatment!");   \
            }                                                                                   \
        }                                                                                       \
    }

    #define XVIGRA_GET_END_BORDER_INDEX(value, treatment, index, size) {                      \
        switch((treatment).getType()) {                                                       \
            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {                           \
                (value) = 2 * (size) - (index) - 2;                                           \
                break;                                                                        \
            }                                                                                 \
            case xvigra::BorderTreatmentType::AVOID: {                                        \
                throw std::domain_error(                                                      \
                    "getBorderIndex(): Border treatment AVOID should not be used here!"       \
                );                                                                            \
            }                                                                                 \
            case xvigra::BorderTreatmentType::CONSTANT: {                                     \
                (value) = -1;                                                                 \
                break;                                                                        \
            }                                                                                 \
            case xvigra::BorderTreatmentType::REPEAT: {                                       \
                (value) = (size) - 1;                                                         \
                break;                                                                        \
            }                                                                                 \
            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {                            \
                (value) = 2 * (size) - (index) - 1;                                           \
                break;                                                                        \
            }                                                                                 \
            case xvigra::BorderTreatmentType::WRAP: {                                         \
                (value) = (index) - (size);                                                   \
                break;                                                                        \
            }                                                                                 \
            default: {                                                                        \
                throw std::domain_error("getBorderIndex(): Unknown end border treatment!");   \
            }                                                                                 \
        }                                                                                     \
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ utility - end                                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ convolve1D - begin                                                                                               ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

     /*
     * <p>
     * Calculates the explicit 1-dimensional convolution of the input with the given 1-dimensional kernel based on the
     * GEMM-based algorithm of Chellapilla K., Puri S. and Simard P. .
     * This function requires an input of shape  W x C or C x W and a kernel with at least 1 dimension or at maximum
     * a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::FIRST or ChannelPosition::LAST inputs; for ChannelPosition::IMPLICIT
     * use xvigra::convolve1DImplicit.
     * </p>
     *
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpression xexpression containing the kernel data
     * @param options object containing information about padding, stride, dilation, channel position and border
                      treatment
     * @return the result of the 1-dimensional convolution between the input and kernel as xt::xtensor
     * @throws std::invalid_argument * if input does not match the required shape
                                     * if IMPLICIT channel position is requested.
                                     * if the input channels in the input and kernel do not align
                                     * if the padded input is smaller than the dilated kernel
     */
    template <typename T, typename O>
    auto convolve1D(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions& options
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelContainerType = typename xt::xexpression<O>::derived_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();

        if (options.channelPosition == xvigra::ChannelPosition::IMPLICIT) {
            throw std::invalid_argument(
                "convolve1D(): Implicit channel option is not supported for explicit channels in input!"
            );
        }

        if (input.dimension() != 2) {
            throw std::invalid_argument("convolve1D(): Need 2 dimensional (W x C or C x W) input!");
        }

        // Input Parameters
        int inputChannels;
        int inputWidth;
        if (options.channelPosition == xvigra::ChannelPosition::LAST) {
            inputWidth = static_cast<int>(input.shape()[0]);
            inputChannels = static_cast<int>(input.shape()[1]);
        } else {
            inputChannels = static_cast<int>(input.shape()[0]);
            inputWidth = static_cast<int>(input.shape()[1]);
        }

        // Kernel
        xt::xtensor<KernelType, 3> kernel = xvigra::promoteKernelToFull1D(kernelExpression.derived_cast(), inputChannels);

        // Filter Specifications
        int kernelSize = kernel.shape()[2];
        int radius = kernelSize / 2;
        int kernelMinimum;
        int kernelMaximum;

        if(kernelSize % 2 == 0) {
            kernelMinimum = 0;
            kernelMaximum = kernelSize;
        } else {
            kernelMinimum = -radius;
            kernelMaximum = radius + 1;
        }

        // checks
        if(inputChannels != static_cast<int>(kernel.shape()[1])) {
            throw std::invalid_argument("convolve1D(): Input channels of input and kernel do not align!");
        }

        if(inputWidth + options.paddingTotal() < (kernelSize - 1) * options.dilation + 1) {
            throw std::invalid_argument("convolve1D(): Kernel width is greater than padded input width!");
        }

        // input output meta data
        int inputWidthMinimum;
        int inputWidthMaximum;

        if(kernelSize % 2 == 0) {
            inputWidthMinimum = -options.paddingBegin();
            inputWidthMaximum = inputWidth
                                + options.paddingEnd()
                                - options.dilation * (kernelSize - 1);
        } else {
            inputWidthMinimum = -options.paddingBegin()
                                + options.dilation * std::abs(kernelMinimum);
            inputWidthMaximum = inputWidth
                                + options.paddingEnd()
                                - options.dilation * (radius);
        }

        int outputWidth = xvigra::calculateOutputSize(inputWidth, kernelSize, options);
        std::vector<int> inputWidthIndices = xvigra::range(inputWidthMinimum, inputWidthMaximum, options.stride);

        // calculate result
        Tensor2D<ResultType> result;

        if (options.channelPosition == xvigra::ChannelPosition::FIRST) {
            Tensor3D<ResultType> patch = xt::zeros<ResultType>({
                inputChannels,
                kernelSize,
                outputWidth
            });
            int outputChannels = kernel.shape()[0];

            for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                for (auto kernelX = kernelMinimum; kernelX < kernelMaximum; ++kernelX) {
                    auto kernelOffsetX = options.dilation * kernelX;
                    auto patchKernelX = kernelX + std::abs(kernelMinimum);

                    for (std::size_t outIndex = 0; outIndex < inputWidthIndices.size(); ++outIndex) {
                        auto inputX = inputWidthIndices.at(outIndex) + kernelOffsetX;

                        if(0 <= inputX && inputX < inputWidth) {
                            patch(inputChannel, patchKernelX, outIndex) = static_cast<ResultType>(input(inputChannel, inputX));
                        } else if (inputX < 0) {
                            XVIGRA_SET_BEGIN_BORDER_VALUE_CHANNEL_FIRST
                        } else if(inputWidth <= inputX) {
                            XVIGRA_SET_END_BORDER_VALUE_CHANNEL_FIRST
                        }
                    }
                }
            }

            auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * kernelSize});
            auto reshapedPatch = xt::reshape_view(patch, {inputChannels * kernelSize, outputWidth});
            result = xt::linalg::dot(reshapedKernel, reshapedPatch);
        } else {
            Tensor3D<ResultType> patch = xt::zeros<InputType>({
                outputWidth,
                inputChannels,
                kernelSize
            });
            int outputChannels = kernel.shape()[0];

            for (std::size_t outIndex = 0; outIndex < inputWidthIndices.size(); ++outIndex) {
                for (auto kernelX = kernelMinimum; kernelX < kernelMaximum; ++kernelX) {
                    auto kernelOffsetX = options.dilation * kernelX;
                    auto patchKernelX = kernelX + std::abs(kernelMinimum);
                    auto inputX = inputWidthIndices.at(outIndex) + kernelOffsetX;

                    if(0 <= inputX && inputX < inputWidth) {
                        for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                            patch(outIndex, inputChannel, patchKernelX) = static_cast<ResultType>(input(inputX, inputChannel));
                        }
                    } else if (inputX < 0) {
                        for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                            XVIGRA_SET_BEGIN_BORDER_VALUE_CHANNEL_LAST
                        }
                    } else if(inputWidth <= inputX) {
                        for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                            XVIGRA_SET_END_BORDER_VALUE_CHANNEL_LAST
                        }
                    }
                }
            }

            auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * kernelSize});
            auto reshapedPatch = xt::reshape_view(patch, {outputWidth, kernelSize * inputChannels});
            result = xt::linalg::dot(reshapedPatch, xt::transpose(reshapedKernel));
        }

        return result;
    }


    /*
     * <p>
     * Calculates the explicit 1-dimensional convolution of the input with the given 1-dimensional kernel based on the
     * GEMM-based algorithm of Chellapilla K., Puri S. and Simard P. .
     * This function requires an input of shape W and a kernel with at least 1 dimension or at maximum
     * a full filter of 3 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::IMPLICIT inputs; for ChannelPosition::FIRST or ChannelPosition::LAST
     * use xvigra::convolve1D.
     * </p>
     *
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpression xexpression containing the kernel data
     * @param kernelOptions object containing information about padding, stride, dilation, channel position and border
                            treatment
     * @return the result of the 1-dimensional convolution between the input and kernel as xt::xtensor
     * @throws std::invalid_argument * if input does not match the required shape
                                     * if the given channel position is not ChannelPosition::IMPLICIT
     */
    template <typename T, typename O>
    auto convolve1DImplicit(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions& options
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelContainerType = typename xt::xexpression<O>::derived_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();
        KernelContainerType kernel = kernelExpression.derived_cast();

        if (options.channelPosition != xvigra::ChannelPosition::IMPLICIT) {
            throw std::domain_error("convolve1DImplicit(): Expected implicit channels in options!");
        }

        if (input.dimension() != 1) {
            throw std::invalid_argument("convolve1DImplicit(): Need 1 dimensional (W) input!");
        }

        xvigra::KernelOptions tempOptions(options);
        tempOptions.channelPosition = xvigra::ChannelPosition::LAST;
        auto normalizedInput = xt::expand_dims(input, input.dimension());
        auto result = convolve1D(normalizedInput, kernel, tempOptions);

        return Tensor1D<ResultType>(xt::squeeze(result));
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ convolve1D - end                                                                                                 ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ convolve2D - begin                                                                                               ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

     /*
     * <p>
     * Calculates the explicit 2-dimensional convolution of the input with the given 2-dimensional kernel based on the
     * GEMM-based algorithm of Chellapilla K., Puri S. and Simard P. .
     * This function requires an input of shape H x W x C or C x H x W and a kernel with at least 2 dimension or at maximum
     * a full filter of 4 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull2D.
     * This function can only process ChannelPosition::FIRST or ChannelPosition::LAST inputs; for ChannelPosition::IMPLICIT
     * use xvigra::convolve2DImplicit.
     * </p>
     *
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpression xexpression containing the kernel data
     * @param options object containing information about padding, stride, dilation, channel position and border
                      treatment
     * @return the result of the 2-dimensional convolution between the input and kernel as xt::xtensor
     * @throws std::invalid_argument * if input does not match the required shape
                                     * if IMPLICIT channel position is requested.
                                     * if the input channels in the input and kernel do not align
                                     * if the padded input is smaller than the dilated kernel
     */
    template <typename T, typename O>
    auto convolve2D(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions& optionsY,
        const xvigra::KernelOptions& optionsX
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelContainerType = typename xt::xexpression<O>::derived_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();

        if (optionsY.channelPosition != optionsX.channelPosition) {
            throw std::invalid_argument(
                "convolve2D(): Channel can't be on different positions for optionsY and optionsX!"
            );
        }

        if (optionsY.channelPosition == xvigra::ChannelPosition::IMPLICIT) {
            throw std::invalid_argument(
                "convolve2D(): Implicit channel option is not supported for explicit channels in input!"
            );
        }

        if (input.dimension() != 3) {
            throw std::invalid_argument("convolve2D(): Need 3 dimensional (H x W x C or C x H x W) input!");
        }

        int inputChannels;
        int inputHeight;
        int inputWidth;

        if (optionsY.channelPosition == xvigra::ChannelPosition::FIRST) {
            inputChannels = input.shape()[0];
            inputHeight = input.shape()[1];
            inputWidth = input.shape()[2];
        } else {
            inputHeight = input.shape()[0];
            inputWidth = input.shape()[1];
            inputChannels = input.shape()[2];
        }

        // Kernel
        xt::xtensor<KernelType, 4> kernel = xvigra::promoteKernelToFull2D(kernelExpression.derived_cast(), inputChannels);

        int outputChannels = kernel.shape()[0];
        int kernelHeight = kernel.shape()[2];
        int kernelWidth = kernel.shape()[3];

        if(inputChannels != static_cast<int>(kernel.shape()[1])) {// dimension mismatch
            throw std::invalid_argument("convolve2D(): Input channels of input and kernel do not align!");
        }

        // size mismatch
        if (inputHeight + optionsY.paddingTotal() < (kernelHeight - 1) * optionsY.dilation + 1) {
            throw std::invalid_argument("convolve2D(): Kernel height is greater than padded input height!");
        }

        if (inputWidth + optionsX.paddingTotal() < (kernelWidth - 1) * optionsX.dilation + 1) {
            throw std::invalid_argument("convolve2D(): Kernel width is greater than padded input width!");
        }

        int kernelHeightRadius = kernelHeight / 2;
        int kernelHeightMinimum = kernelHeight % 2 == 0 ? 0 : -kernelHeightRadius;
        int kernelHeightMaximum = kernelHeight % 2 == 0 ? kernelHeight : kernelHeightRadius + 1;

        int kernelWidthRadius = kernelWidth / 2;
        int kernelWidthMinimum = kernelWidth % 2 == 0 ? 0 : -kernelWidthRadius;
        int kernelWidthMaximum = kernelWidth % 2 == 0 ? kernelWidth : kernelWidthRadius + 1;

        int outputHeight = xvigra::calculateOutputSize(inputHeight, kernelHeight, optionsY);
        int outputWidth = xvigra::calculateOutputSize(inputWidth, kernelWidth, optionsX);

        int heightMinimum = -optionsY.paddingBegin() + optionsY.dilation * std::abs(kernelHeight % 2 == 0 ? 0 : kernelHeightMinimum);
        int heightMaximum = inputHeight + optionsY.paddingEnd() - optionsY.dilation * (kernelHeight % 2 == 0 ? kernelHeight - 1 : kernelHeightRadius);

        int widthMinimum = -optionsX.paddingBegin() + optionsX.dilation * std::abs(kernelWidth % 2 == 0 ? 0 : kernelWidthMinimum);
        int widthMaximum = inputWidth + optionsX.paddingEnd() - optionsX.dilation * (kernelWidth % 2 == 0 ? kernelWidth - 1 : kernelWidthRadius);

        auto inputHeightIndices = xvigra::range(heightMinimum, heightMaximum, optionsY.stride);
        auto inputWidthIndices = xvigra::range(widthMinimum, widthMaximum, optionsX.stride);

        xt::xtensor<ResultType, 3> result;
        if (optionsY.channelPosition == xvigra::ChannelPosition::FIRST) {
           Tensor5D<ResultType> patch = xt::zeros<ResultType>({inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth});

            for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                for (auto kernelY = kernelHeightMinimum; kernelY < kernelHeightMaximum; ++kernelY) {
                    auto outKernelY = kernelY + std::abs(kernelHeightMinimum);
                    auto inputOffsetY = kernelY * optionsY.dilation;

                    for (auto kernelX = kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
                        auto outKernelX = kernelX + std::abs(kernelWidthMinimum);
                        auto inputOffsetX = kernelX * optionsX.dilation;

                        for (auto [outIndexY, inputIndexY] : xvigra::enumerate(inputHeightIndices)) {
                            auto inputY = inputIndexY + inputOffsetY;

                            xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                            int indexY = inputY;

                            if (indexY < 0) {
                                treatmentY = optionsY.borderTreatmentBegin;
                                XVIGRA_GET_BEGIN_BORDER_INDEX(indexY, treatmentY, indexY, inputHeight)
                            } else if(inputHeight <= indexY) {
                                treatmentY = optionsY.borderTreatmentEnd;
                                XVIGRA_GET_END_BORDER_INDEX(indexY, treatmentY, indexY, inputHeight)
                            }

                            for (auto [outIndexX, inputIndexX] : xvigra::enumerate(inputWidthIndices)) {
                                auto inputX = inputIndexX + inputOffsetX;

                                if (indexY == -1) {
                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(treatmentY.getValue<InputType>());
                                } else {
                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (indexX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        XVIGRA_GET_BEGIN_BORDER_INDEX(indexX, treatmentX, indexX, inputWidth)
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        XVIGRA_GET_END_BORDER_INDEX(indexX, treatmentX, indexX, inputWidth)
                                    }

                                    if (indexX == -1) {
                                        patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(treatmentX.getValue<InputType>());
                                    } else {
                                        patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = input(inputChannel, indexY, indexX);
                                    }

                                }
                            }
                        }
                    }
                }
            }

            auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * kernelHeight * kernelWidth});
            auto reshapedPatch = xt::reshape_view(patch, {inputChannels * kernelHeight * kernelWidth, outputHeight * outputWidth});
            result = xt::reshape_view(xt::linalg::dot(reshapedKernel, reshapedPatch), {outputChannels, outputHeight, outputWidth});

        } else {
            Tensor5D<ResultType> patch= xt::zeros<ResultType>({outputHeight, outputWidth, inputChannels, kernelHeight, kernelWidth});

            for (auto [outIndexY, inputIndexY] : xvigra::enumerate(inputHeightIndices)) {
                for (auto [outIndexX, inputIndexX] : xvigra::enumerate(inputWidthIndices)) {
                    for (auto kernelY = kernelHeightMinimum; kernelY < kernelHeightMaximum; ++kernelY) {
                        auto inputY = inputIndexY + kernelY * optionsY.dilation;
                        auto outKernelY = kernelY + std::abs(kernelHeightMinimum);

                        xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                        int indexY = inputY;

                        if (indexY < 0) {
                            treatmentY = optionsY.borderTreatmentBegin;
                            XVIGRA_GET_BEGIN_BORDER_INDEX(indexY, treatmentY, indexY, inputHeight)
                        } else if(inputHeight <= indexY) {
                            treatmentY = optionsY.borderTreatmentEnd;
                            XVIGRA_GET_END_BORDER_INDEX(indexY, treatmentY, indexY, inputHeight)
                        }

                        for (auto kernelX = kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
                            auto inputX = inputIndexX + kernelX * optionsX.dilation;
                            auto outKernelX = kernelX + std::abs(kernelWidthMinimum);

                            if (indexY == -1) {
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(treatmentY.getValue<InputType>());
                                }
                            } else {
                                xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                int indexX = inputX;

                                if (indexX < 0) {
                                    treatmentX = optionsX.borderTreatmentBegin;
                                    XVIGRA_GET_BEGIN_BORDER_INDEX(indexX, treatmentX, indexX, inputWidth)
                                } else if(inputWidth <= indexX) {
                                    treatmentX = optionsX.borderTreatmentEnd;
                                    XVIGRA_GET_END_BORDER_INDEX(indexX, treatmentX, indexX, inputWidth)
                                }

                                if (indexX == -1) {
                                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                        patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(treatmentX.getValue<InputType>());
                                    }
                                } else {
                                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                        patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(input(indexY, indexX, inputChannel));
                                    }
                                }
                            }
                        }
                    }
                }
            }

            auto reshapedKernel = xt::transpose(xt::reshape_view(kernel, {outputChannels, inputChannels * kernelHeight * kernelWidth}));
            auto reshapedPatch = xt::reshape_view(patch, {outputHeight*outputWidth, inputChannels*kernelHeight*kernelWidth});
            result = xt::reshape_view(xt::linalg::dot(reshapedPatch, reshapedKernel), {outputHeight, outputWidth, outputChannels});
        }

        return result;
    }


    template <typename T, typename O>
    inline auto convolve2D(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions2D& options2D
    ) {
        return convolve2D(
            inputExpression.derived_cast(),
            kernelExpression.derived_cast(),
            options2D.optionsY,
            options2D.optionsX
        );
    }

    /*
     * <p>
     * Calculates the explicit 2-dimensional convolution of the input with the given 2-dimensional kernel based on the
     * GEMM-based algorithm of Chellapilla K., Puri S. and Simard P. .
     * This function requires an input of shape H x W x C or C x H x W and a kernel with at least 2 dimension or at maximum
     * a full filter of 4 dimensions.
     * Missing kernel dimensions are inserted by xvigra::promoteKernelToFull1D.
     * This function can only process ChannelPosition::IMPLICIT inputs; for ChannelPosition::FIRST or ChannelPosition::LAST
     * use xvigra::convolve2D.
     * </p>
     *
     * @tparam O derived type of the input xexpression
     * @tparam T derived type of the kernel xexpression
     * @param inputExpression xexpression containing the input data
     * @param rawKernelExpression xexpression containing the kernel data
     * @param kernelOptions object containing information about padding, stride, dilation, channel position and border
                            treatment
     * @return the result of the 2-dimensional convolution between the input and kernel as xt::xtensor
     * @throws std::invalid_argument * if input does not match the required shape
                                     * if the given channel position is not ChannelPosition::IMPLICIT
     */
    template <typename T, typename O>
    auto convolve2DImplicit(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions2D& options2D
    ) {
        using InputContainerType = typename xt::xexpression<T>::derived_type;
        using InputType = typename InputContainerType::value_type;
        using KernelContainerType = typename xt::xexpression<O>::derived_type;
        using KernelType = typename KernelContainerType::value_type;
        using ResultType = typename std::common_type_t<InputType, KernelType>;

        InputContainerType input = inputExpression.derived_cast();
        KernelContainerType kernel = kernelExpression.derived_cast();

        if (options2D.optionsY.channelPosition != xvigra::ChannelPosition::IMPLICIT) {
            throw std::domain_error("convolve2DImplicit(): Expected implicit channels in options!");
        }

        if (input.dimension() != 2) {
            throw std::invalid_argument("convolve2DImplicit(): Need 2 dimensional (H x W) input!");
        }

        xvigra::KernelOptions2D tempOptions(options2D);
        tempOptions.setChannelPosition(xvigra::ChannelPosition::LAST);
        auto normalizedInput = xt::expand_dims(input, input.dimension());
        auto result = convolve2D(normalizedInput, kernel, tempOptions);

        return Tensor2D<ResultType>(xt::squeeze(result));
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ convolve2D - end                                                                                                 ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
} // xvigra

#endif // XVIGRA_EXPLICIT_CONVOLUTION_HPP
