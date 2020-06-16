#ifndef XVIGRA_LEGACY_EXPLICIT_CONVOLUTION_HPP
#define XVIGRA_LEGACY_EXPLICIT_CONVOLUTION_HPP

#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xtensor.hpp"

#include "xtensor-blas/xlinalg.hpp"

#include "xvigra/convolution_util.hpp"

namespace xvigra_legacy {

template <typename T>
using Tensor1D = typename xt::xtensor<T, 1>;
template <typename T>
using Tensor2D = typename xt::xtensor<T, 2>;
template <typename T>
using Tensor3D = typename xt::xtensor<T, 3>;
template <typename T>
using Tensor4D = typename xt::xtensor<T, 4>;
template <typename T>
using Tensor5D = typename xt::xtensor<T, 5>;


template <typename InputType, typename KernelType>
Tensor2D<typename std::common_type_t<InputType, KernelType>> convolve1D_v1(
    const Tensor2D<InputType>& input,
    const Tensor3D<KernelType>& kernel,
    const xvigra::KernelOptions& options
) {
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    if (options.channelPosition == xvigra::ChannelPosition::IMPLICIT) {
        throw std::invalid_argument(
            "convolve1D(): Implicit channel option is not supported for explicit channels in input!"
        );
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

    // Filter Specifications
    int size = kernel.shape()[2];
    int radius = size / 2;
    int kernelMinimum;
    int kernelMaximum;

    if(size % 2 == 0) {
        kernelMinimum = 0;
        kernelMaximum = size;
    } else {
        kernelMinimum = -radius;
        kernelMaximum = radius + 1;
    }

    // checks
    if(inputChannels != static_cast<int>(kernel.shape()[1])) {
        throw std::invalid_argument("convolve1D(): Input channels of input and kernel do not align!");
    }
    
    if(inputWidth + options.paddingTotal() < (size - 1) * options.dilation + 1) {
        throw std::invalid_argument("convolve1D(): Kernel width is greater than padded input width!");
    }

    // input output meta data
    int inputWidthMinimum;
    int inputWidthMaximum;

    // TODO avoid 
    if(size % 2 == 0) {
        inputWidthMinimum = -options.paddingBegin();
        inputWidthMaximum = inputWidth 
                            + options.paddingEnd()
                            - options.dilation * (size - 1);
    } else {
        inputWidthMinimum = -options.paddingBegin()
                            + options.dilation * std::abs(kernelMinimum);
        inputWidthMaximum = inputWidth 
                            + options.paddingEnd()
                            - options.dilation * (radius);
    }

    int outputWidth = xvigra::calculateOutputSize(inputWidth, size, options);
    std::vector<int> inputWidthIndices = xvigra::range(inputWidthMinimum, inputWidthMaximum, options.stride);

    // calculate result
    Tensor2D<ResultType> result;

    if (options.channelPosition == xvigra::ChannelPosition::FIRST) {
        Tensor3D<InputType> patch = xt::zeros<InputType>({
            inputChannels, 
            size, 
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
                        patch(inputChannel, patchKernelX, outIndex) = input(inputChannel, inputX);  
                    } else if (inputX < 0) {
                        InputType value;
                        auto treatment = options.borderTreatmentBegin;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(inputChannel, std::abs(inputX + 1)); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputChannel, 0); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(inputChannel, std::abs(inputX)); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputChannel, inputX + inputWidth); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown begin border treatment!");
                            }
                        }

                        patch(inputChannel, patchKernelX, outIndex) = value;
                    } else if(inputWidth <= inputX) {
                        InputType value;
                        auto treatment = options.borderTreatmentEnd;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(inputChannel, 2 * inputWidth - inputX - 2); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputChannel, inputX - 1); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(inputChannel, 2 * inputWidth - inputX - 1); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputChannel, inputX - inputWidth); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown end border treatment!");
                            }
                        }

                        patch(inputChannel, patchKernelX, outIndex) = value;
                    }
                }
            }
        }
        
        auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * size});
        auto reshapedPatch = xt::reshape_view(patch, {inputChannels * size, outputWidth});
        result = xt::linalg::dot(reshapedKernel, reshapedPatch);
    } else {
        Tensor3D<InputType> patch = xt::zeros<InputType>({
            outputWidth,
            inputChannels, 
            size
        });
        int outputChannels = kernel.shape()[0];

        for (std::size_t outIndex = 0; outIndex < inputWidthIndices.size(); ++outIndex) {
            for (auto kernelX = kernelMinimum; kernelX < kernelMaximum; ++kernelX) {
                auto kernelOffsetX = options.dilation * kernelX;
                auto patchKernelX = kernelX + std::abs(kernelMinimum);
                auto inputX = inputWidthIndices.at(outIndex) + kernelOffsetX;

                if(0 <= inputX && inputX < inputWidth) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        patch(outIndex, inputChannel, patchKernelX) = input(inputX, inputChannel); 
                    }
                } else if (inputX < 0) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) { 
                        auto treatment = options.borderTreatmentBegin;
                        InputType value;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(std::abs(inputX + 1), inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(0, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(std::abs(inputX), inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputX + inputWidth, inputChannel); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown begin border treatment!");
                            }
                        }

                        patch(outIndex, inputChannel, patchKernelX) = value;
                    }
                } else if(inputWidth <= inputX) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        InputType value;
                        auto treatment = options.borderTreatmentEnd;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(2 * inputWidth - inputX - 2, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputX - 1, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(2 * inputWidth - inputX - 1, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputX - inputWidth, inputChannel); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown end border treatment!");
                            }
                        }

                        patch(outIndex, inputChannel, patchKernelX) = value;
                    }
                }
            }
        }

        auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * size});
        auto reshapedPatch = xt::reshape_view(patch, {outputWidth, size * inputChannels});
        result = xt::linalg::dot(reshapedPatch, xt::transpose(reshapedKernel));
    }

    return result;
}


// ================================================================================================
// ================================================================================================
// ================================================================================================
// ================================================================================================

template <typename InputType, typename KernelType>
Tensor2D<typename std::common_type_t<InputType, KernelType>> convolve1D_v2(
    const Tensor2D<InputType>& input,
    const Tensor3D<KernelType>& kernel,
    const xvigra::KernelOptions& options
) {
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    xvigra::ChannelPosition channelPosition = options.channelPosition;
    int paddingBegin = options.paddingBegin();
    int paddingEnd = options.paddingEnd();
    int paddingTotal = options.paddingTotal();
    int dilation = options.dilation;
    int stride = options.stride;
    xvigra::BorderTreatment borderTreatmentBegin = options.borderTreatmentBegin;
    xvigra::BorderTreatment borderTreatmentEnd = options.borderTreatmentEnd;

    if (channelPosition == xvigra::ChannelPosition::IMPLICIT) {
        throw std::invalid_argument(
            "convolve1D(): Implicit channel option is not supported for explicit channels in input!"
        );
    }

    // Input Parameters
    int inputChannels;
    int inputWidth;
    if (channelPosition == xvigra::ChannelPosition::LAST) {
        inputWidth = static_cast<int>(input.shape()[0]);
        inputChannels = static_cast<int>(input.shape()[1]);
    } else {
        inputChannels = static_cast<int>(input.shape()[0]);
        inputWidth = static_cast<int>(input.shape()[1]);
    }

    // Filter Specifications
    int size = kernel.shape()[2];
    int radius = size / 2;
    int kernelMinimum;
    int kernelMaximum;

    if(size % 2 == 0) {
        kernelMinimum = 0;
        kernelMaximum = size;
    } else {
        kernelMinimum = -radius;
        kernelMaximum = radius + 1;
    }

    // checks
    if(inputChannels != static_cast<int>(kernel.shape()[1])) {
        throw std::invalid_argument("convolve1D(): Input channels of input and kernel do not align!");
    }
    
    if(inputWidth + paddingTotal < (size - 1) * dilation + 1) {
        throw std::invalid_argument("convolve1D(): Kernel width is greater than padded input width!");
    }

    // input output meta data
    int inputWidthMinimum;
    int inputWidthMaximum;

    // TODO avoid 
    if(size % 2 == 0) {
        inputWidthMinimum = -paddingBegin;
        inputWidthMaximum = inputWidth 
                            + paddingEnd
                            - dilation * (size - 1);
    } else {
        inputWidthMinimum = -paddingBegin
                            + dilation * std::abs(kernelMinimum);
        inputWidthMaximum = inputWidth 
                            + paddingEnd
                            - dilation * (radius);
    }

    int outputWidth = xvigra::calculateOutputSize(inputWidth, size, paddingTotal, stride, dilation);
    std::vector<int> inputWidthIndices = xvigra::range(inputWidthMinimum, inputWidthMaximum, stride);

    // calculate result
    Tensor2D<ResultType> result;

    if (channelPosition == xvigra::ChannelPosition::FIRST) {
        Tensor3D<InputType> patch = xt::zeros<InputType>({
            inputChannels, 
            size, 
            outputWidth
        });
        int outputChannels = kernel.shape()[0];
        
        for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
            for (auto kernelX = kernelMinimum; kernelX < kernelMaximum; ++kernelX) {
                auto kernelOffsetX = dilation * kernelX;
                auto patchKernelX = kernelX + std::abs(kernelMinimum);
                
                for (std::size_t outIndex = 0; outIndex < inputWidthIndices.size(); ++outIndex) {
                    auto inputX = inputWidthIndices.at(outIndex) + kernelOffsetX;
                    
                    if(0 <= inputX && inputX < inputWidth) {
                        patch(inputChannel, patchKernelX, outIndex) = input(inputChannel, inputX);  
                    } else if (inputX < 0) {
                        InputType value;
                        auto treatment = borderTreatmentBegin;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(inputChannel, std::abs(inputX + 1)); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputChannel, 0); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(inputChannel, std::abs(inputX)); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputChannel, inputX + inputWidth); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown begin border treatment!");
                            }
                        }

                        patch(inputChannel, patchKernelX, outIndex) = value;
                    } else if(inputWidth <= inputX) {
                        InputType value;
                        auto treatment = borderTreatmentEnd;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(inputChannel, 2 * inputWidth - inputX - 2); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputChannel, inputX - 1); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(inputChannel, 2 * inputWidth - inputX - 1); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputChannel, inputX - inputWidth); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown end border treatment!");
                            }
                        }

                        patch(inputChannel, patchKernelX, outIndex) = value;
                    }
                }
            }
        }
        
        auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * size});
        auto reshapedPatch = xt::reshape_view(patch, {inputChannels * size, outputWidth});
        result = xt::linalg::dot(reshapedKernel, reshapedPatch);
    } else {
        Tensor3D<InputType> patch = xt::zeros<InputType>({
            outputWidth,
            inputChannels, 
            size
        });
        int outputChannels = kernel.shape()[0];

        for (std::size_t outIndex = 0; outIndex < inputWidthIndices.size(); ++outIndex) {
            for (auto kernelX = kernelMinimum; kernelX < kernelMaximum; ++kernelX) {
                auto kernelOffsetX = dilation * kernelX;
                auto patchKernelX = kernelX + std::abs(kernelMinimum);
                auto inputX = inputWidthIndices.at(outIndex) + kernelOffsetX;

                if(0 <= inputX && inputX < inputWidth) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        patch(outIndex, inputChannel, patchKernelX) = input(inputX, inputChannel); 
                    }
                } else if (inputX < 0) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        InputType value;
                        auto treatment = borderTreatmentBegin;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(std::abs(inputX + 1), inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(0, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(std::abs(inputX), inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputX + inputWidth, inputChannel); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown begin border treatment!");
                            }
                        }

                        patch(outIndex, inputChannel, patchKernelX) = value;
                    }
                } else if(inputWidth <= inputX) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        InputType value;
                        auto treatment = borderTreatmentEnd;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(2 * inputWidth - inputX - 2, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputX - 1, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(2 * inputWidth - inputX - 1, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputX - inputWidth, inputChannel); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown end border treatment!");
                            }
                        }

                        patch(outIndex, inputChannel, patchKernelX) = value;
                    }
                }
            }
        }

        auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * size});
        auto reshapedPatch = xt::reshape_view(patch, {outputWidth, size * inputChannels});
        result = xt::linalg::dot(reshapedPatch, xt::transpose(reshapedKernel));
    }

    return result;
}

// ================================================================================================
// ================================================================================================
// ================================================================================================
// ================================================================================================

template <typename InputType, typename KernelType>
Tensor2D<typename std::common_type_t<InputType, KernelType>> convolve1D_v3(
    const Tensor2D<InputType>& input,
    const Tensor3D<KernelType>& kernel,

    const xvigra::ChannelPosition& channelPosition,
    int paddingBegin,
    int paddingEnd,
    int paddingTotal,
    int dilation,
    int stride,
    const xvigra::BorderTreatment& borderTreatmentBegin,
    const xvigra::BorderTreatment& borderTreatmentEnd
) {
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    if (channelPosition == xvigra::ChannelPosition::IMPLICIT) {
        throw std::invalid_argument(
            "convolve1D(): Implicit channel option is not supported for explicit channels in input!"
        );
    }

    // Input Parameters
    int inputChannels;
    int inputWidth;
    if (channelPosition == xvigra::ChannelPosition::LAST) {
        inputWidth = static_cast<int>(input.shape()[0]);
        inputChannels = static_cast<int>(input.shape()[1]);
    } else {
        inputChannels = static_cast<int>(input.shape()[0]);
        inputWidth = static_cast<int>(input.shape()[1]);
    }

    // Filter Specifications
    int size = kernel.shape()[2];
    int radius = size / 2;
    int kernelMinimum;
    int kernelMaximum;

    if(size % 2 == 0) {
        kernelMinimum = 0;
        kernelMaximum = size;
    } else {
        kernelMinimum = -radius;
        kernelMaximum = radius + 1;
    }

    // checks
    if(inputChannels != static_cast<int>(kernel.shape()[1])) {
        throw std::invalid_argument("convolve1D(): Input channels of input and kernel do not align!");
    }
    
    if(inputWidth + paddingTotal < (size - 1) * dilation + 1) {
        throw std::invalid_argument("convolve1D(): Kernel width is greater than padded input width!");
    }

    // input output meta data
    int inputWidthMinimum;
    int inputWidthMaximum;

    // TODO avoid 
    if(size % 2 == 0) {
        inputWidthMinimum = -paddingBegin;
        inputWidthMaximum = inputWidth 
                            + paddingEnd
                            - dilation * (size - 1);
    } else {
        inputWidthMinimum = -paddingBegin
                            + dilation * std::abs(kernelMinimum);
        inputWidthMaximum = inputWidth 
                            + paddingEnd
                            - dilation * (radius);
    }

    int outputWidth = xvigra::calculateOutputSize(inputWidth, size, paddingTotal, stride, dilation);
    std::vector<int> inputWidthIndices = xvigra::range(inputWidthMinimum, inputWidthMaximum, stride);

    // calculate result
    Tensor2D<ResultType> result;

    if (channelPosition == xvigra::ChannelPosition::FIRST) {
        Tensor3D<InputType> patch = xt::zeros<InputType>({
            inputChannels, 
            size, 
            outputWidth
        });
        int outputChannels = kernel.shape()[0];
        
        for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
            for (auto kernelX = kernelMinimum; kernelX < kernelMaximum; ++kernelX) {
                auto kernelOffsetX = dilation * kernelX;
                auto patchKernelX = kernelX + std::abs(kernelMinimum);
                
                for (std::size_t outIndex = 0; outIndex < inputWidthIndices.size(); ++outIndex) {
                    auto inputX = inputWidthIndices.at(outIndex) + kernelOffsetX;
                    
                    if(0 <= inputX && inputX < inputWidth) {
                        patch(inputChannel, patchKernelX, outIndex) = input(inputChannel, inputX);  
                    } else if (inputX < 0) {
                        InputType value;
                        auto treatment = borderTreatmentBegin;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(inputChannel, std::abs(inputX + 1)); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputChannel, 0); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(inputChannel, std::abs(inputX)); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputChannel, inputX + inputWidth); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown begin border treatment!");
                            }
                        }

                        patch(inputChannel, patchKernelX, outIndex) = value;
                    } else if(inputWidth <= inputX) {
                        InputType value;
                        auto treatment = borderTreatmentEnd;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(inputChannel, 2 * inputWidth - inputX - 2); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputChannel, inputX - 1); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(inputChannel, 2 * inputWidth - inputX - 1); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputChannel, inputX - inputWidth); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown end border treatment!");
                            }
                        }

                        patch(inputChannel, patchKernelX, outIndex) = value;
                    }
                }
            }
        }
        
        auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * size});
        auto reshapedPatch = xt::reshape_view(patch, {inputChannels * size, outputWidth});
        result = xt::linalg::dot(reshapedKernel, reshapedPatch);
    } else {
        Tensor3D<InputType> patch = xt::zeros<InputType>({
            outputWidth,
            inputChannels, 
            size
        });
        int outputChannels = kernel.shape()[0];

        for (std::size_t outIndex = 0; outIndex < inputWidthIndices.size(); ++outIndex) {
            for (auto kernelX = kernelMinimum; kernelX < kernelMaximum; ++kernelX) {
                auto kernelOffsetX = dilation * kernelX;
                auto patchKernelX = kernelX + std::abs(kernelMinimum);
                auto inputX = inputWidthIndices.at(outIndex) + kernelOffsetX;

                if(0 <= inputX && inputX < inputWidth) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        patch(outIndex, inputChannel, patchKernelX) = input(inputX, inputChannel); 
                    }
                } else if (inputX < 0) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        InputType value;
                        auto treatment = borderTreatmentBegin;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(std::abs(inputX + 1), inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(0, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(std::abs(inputX), inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputX + inputWidth, inputChannel); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown begin border treatment!");
                            }
                        }

                        patch(outIndex, inputChannel, patchKernelX) = value;
                    }
                } else if(inputWidth <= inputX) {
                    for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        InputType value;
                        auto treatment = borderTreatmentEnd;

                        switch(treatment.getType()) {
                            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                                value = input(2 * inputWidth - inputX - 2, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::AVOID: {
                                throw std::domain_error(
                                    "convolve1D(): Border treatment AVOID should not be used here!"
                                );
                            }
                            case xvigra::BorderTreatmentType::CONSTANT: {
                                value = treatment.getValue<InputType>();
                                break;
                            }
                            case xvigra::BorderTreatmentType::REPEAT: {
                                value = input(inputX - 1, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                                value = input(2 * inputWidth - inputX - 1, inputChannel); 
                                break;
                            }
                            case xvigra::BorderTreatmentType::WRAP: {
                                value = input(inputX - inputWidth, inputChannel); 
                                break;
                            }
                            default: {
                                throw std::domain_error("convolve1D(): Unknown end border treatment!");
                            }
                        }

                        patch(outIndex, inputChannel, patchKernelX) = value;
                    }
                }
            }
        }

        auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * size});
        auto reshapedPatch = xt::reshape_view(patch, {outputWidth, size * inputChannels});
        result = xt::linalg::dot(reshapedPatch, xt::transpose(reshapedKernel));
    }

    return result;
}

// ================================================================================================
// ================================================================================================
// ================================================================================================
// ================================================================================================

template <bool isBegin>
int getBorderIndex(const xvigra::BorderTreatment& treatment, int index, int size) {
    if (0 <= index && index < size) {
        return index;
    }

    if constexpr (isBegin) {
        switch(treatment.getType()) {
            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                return std::abs(index); 
            }
            case xvigra::BorderTreatmentType::AVOID: {
                throw std::domain_error(
                    "getBorderIndex(): Border treatment AVOID should not be used here!"
                );
            }
            case xvigra::BorderTreatmentType::CONSTANT: {
                return -1;
            }
            case xvigra::BorderTreatmentType::REPEAT: {
                return 0; 
            }
            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                return std::abs(index + 1); 
            }
            case xvigra::BorderTreatmentType::WRAP: {
                return index + size; 
            }
            default: {
                throw std::domain_error("getBorderIndex(): Unknown begin border treatment!");
            }
        }
    } else {
        switch(treatment.getType()) {
            case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {
                return 2 * size - index - 2; 
            }
            case xvigra::BorderTreatmentType::AVOID: {
                throw std::domain_error(
                    "getBorderIndex(): Border treatment AVOID should not be used here!"
                );
            }
            case xvigra::BorderTreatmentType::CONSTANT: {
                return -1;
            }
            case xvigra::BorderTreatmentType::REPEAT: {
                return size - 1; 
            }
            case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {
                return 2 * size - index - 1; 
            }
            case xvigra::BorderTreatmentType::WRAP: {
                return index - size; 
            }
            default: {
                throw std::domain_error("getBorderIndex(): Unknown end border treatment!");
            }
        }
    }   
}


template <typename InputType, typename KernelType>
Tensor2D<typename std::common_type_t<InputType, KernelType>> convolve1D_v4(
    const Tensor2D<InputType>& input,
    const Tensor3D<KernelType>& kernel,
    const xvigra::KernelOptions& options
) {
    using ResultType = typename std::common_type_t<InputType, KernelType>;

    if (options.channelPosition == xvigra::ChannelPosition::IMPLICIT) {
        throw std::invalid_argument(
            "convolve1D(): Implicit channel option is not supported for explicit channels in input!"
        );
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
        Tensor3D<InputType> patch = xt::zeros<InputType>({
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
                        patch(inputChannel, patchKernelX, outIndex) = input(inputChannel, inputX);  
                    } else {
                        xvigra::BorderTreatment treatment = xvigra::BorderTreatment::avoid();
                        int index = -1;

                        if (inputX < 0) {
                            treatment = options.borderTreatmentBegin;
                            index = getBorderIndex<true>(treatment, inputX, inputWidth);
                        } else if(inputWidth <= inputX) {
                            treatment = options.borderTreatmentEnd;
                            index = getBorderIndex<false>(treatment, inputX, inputWidth);
                        }

                        InputType value;
                        if (index == -1) {
                            value = treatment.getValue<InputType>();
                        } else {
                            value = input(inputChannel, index);
                        }

                        patch(inputChannel, patchKernelX, outIndex) = value;
                    }
                }
            }
        }
        
        auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * kernelSize});
        auto reshapedPatch = xt::reshape_view(patch, {inputChannels * kernelSize, outputWidth});
        result = xt::linalg::dot(reshapedKernel, reshapedPatch);
    } else {
        Tensor3D<InputType> patch = xt::zeros<InputType>({
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
                        patch(outIndex, inputChannel, patchKernelX) = input(inputX, inputChannel); 
                    }
                } else {
                     for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        xvigra::BorderTreatment treatment = xvigra::BorderTreatment::avoid();
                        int index = -1;

                        if (inputX < 0) {
                            treatment = options.borderTreatmentBegin;
                            index = getBorderIndex<true>(treatment, inputX, inputWidth);
                        } else if(inputWidth <= inputX) {
                            treatment = options.borderTreatmentEnd;
                            index = getBorderIndex<false>(treatment, inputX, inputWidth);
                        }

                        InputType value;
                        if (index == -1) {
                            value = treatment.getValue<InputType>();
                        } else {
                            value = input(index, inputChannel);
                        }

                        patch(outIndex, inputChannel, patchKernelX) = value;
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

// ================================================================================================
// ================================================================================================
// ================================================================================================
// ================================================================================================


template <typename T, typename O>
auto convolve1D_v5(
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

    if (options.channelPosition == xvigra::ChannelPosition::IMPLICIT) {
        throw std::invalid_argument(
            "convolve1D(): Implicit channel option is not supported for explicit channels in input!"
        );
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
        Tensor3D<InputType> patch = xt::zeros<InputType>({
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
                        patch(inputChannel, patchKernelX, outIndex) = input(inputChannel, inputX);  
                    } else {
                        xvigra::BorderTreatment treatment = xvigra::BorderTreatment::avoid();
                        int index = -1;

                        if (inputX < 0) {
                            treatment = options.borderTreatmentBegin;
                            index = getBorderIndex<true>(treatment, inputX, inputWidth);
                        } else if(inputWidth <= inputX) {
                            treatment = options.borderTreatmentEnd;
                            index = getBorderIndex<false>(treatment, inputX, inputWidth);
                        }

                        InputType value;
                        if (index == -1) {
                            value = treatment.getValue<InputType>();
                        } else {
                            value = input(inputChannel, index);
                        }

                        patch(inputChannel, patchKernelX, outIndex) = value;
                    }
                }
            }
        }
        
        auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputChannels * kernelSize});
        auto reshapedPatch = xt::reshape_view(patch, {inputChannels * kernelSize, outputWidth});
        result = xt::linalg::dot(reshapedKernel, reshapedPatch);
    } else {
        Tensor3D<InputType> patch = xt::zeros<InputType>({
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
                        patch(outIndex, inputChannel, patchKernelX) = input(inputX, inputChannel); 
                    }
                } else {
                     for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                        xvigra::BorderTreatment treatment = xvigra::BorderTreatment::avoid();
                        int index = -1;

                        if (inputX < 0) {
                            treatment = options.borderTreatmentBegin;
                            index = getBorderIndex<true>(treatment, inputX, inputWidth);
                        } else if(inputWidth <= inputX) {
                            treatment = options.borderTreatmentEnd;
                            index = getBorderIndex<false>(treatment, inputX, inputWidth);
                        }

                        InputType value;
                        if (index == -1) {
                            value = treatment.getValue<InputType>();
                        } else {
                            value = input(index, inputChannel);
                        }

                        patch(outIndex, inputChannel, patchKernelX) = value;
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


// ================================================================================================
// ================================================================================================
// ================================================================================================
// ================================================================================================

} // xvigra_legacy

#endif // XVIGRA_LEGACY_EXPLICIT_CONVOLUTION_HPP