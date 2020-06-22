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

template <bool isBegin>
int getBorderIndex(const BorderTreatment& treatment, int index, int size) {
    if (0 <= index && index < size) {
        return index;
    }

    if constexpr (isBegin) {
        switch(treatment.getType()) {
            case BorderTreatmentType::ASYMMETRIC_REFLECT: {
                return std::abs(index); 
            }
            case BorderTreatmentType::AVOID: {
                throw std::domain_error(
                    "getBorderIndex(): Border treatment AVOID should not be used here!"
                );
            }
            case BorderTreatmentType::CONSTANT: {
                return -1;
            }
            case BorderTreatmentType::REPEAT: {
                return 0; 
            }
            case BorderTreatmentType::SYMMETRIC_REFLECT: {
                return std::abs(index + 1); 
            }
            case BorderTreatmentType::WRAP: {
                return index + size; 
            }
            default: {
                throw std::domain_error("getBorderIndex(): Unknown begin border treatment!");
            }
        }
    } else {
        switch(treatment.getType()) {
            case BorderTreatmentType::ASYMMETRIC_REFLECT: {
                return 2 * size - index - 2; 
            }
            case BorderTreatmentType::AVOID: {
                throw std::domain_error(
                    "getBorderIndex(): Border treatment AVOID should not be used here!"
                );
            }
            case BorderTreatmentType::CONSTANT: {
                return -1;
            }
            case BorderTreatmentType::REPEAT: {
                return size - 1; 
            }
            case BorderTreatmentType::SYMMETRIC_REFLECT: {
                return 2 * size - index - 1; 
            }
            case BorderTreatmentType::WRAP: {
                return index - size; 
            }
            default: {
                throw std::domain_error("getBorderIndex(): Unknown end border treatment!");
            }
        }
    }   
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ utility - end                                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ convolve1D - begin                                                                                               ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

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
    KernelContainerType kernel = kernelExpression.derived_cast();

    if (options.channelPosition == xvigra::ChannelPosition::IMPLICIT) {
        throw std::invalid_argument(
            "convolve1D(): Implicit channel option is not supported for explicit channels in input!"
        );
    }

    if (input.dimension() != 2) {
        throw std::invalid_argument("convolve1D(): Need 2 dimensional (W x C or C x W) input!");
    }

    if (kernel.dimension() != 3) {
        throw std::invalid_argument("convolve1D(): Need a full 3 dimensional (C_{out} x C_{in} x K_{W}) kernel to operate!");
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

                        patch(inputChannel, patchKernelX, outIndex) = static_cast<ResultType>(value);
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

                        patch(outIndex, inputChannel, patchKernelX) = static_cast<ResultType>(value);
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

    if (kernel.dimension() != 3) {
        throw std::invalid_argument("convolve1DImplicit(): Need a full 3 dimensional (C_{out} x C_{in} x K_{W}) kernel to operate!");
    }

    xvigra::KernelOptions tempOptions(options);
    tempOptions.channelPosition = xvigra::ChannelPosition::LAST;
    auto normalizedInput = xt::reshape_view(input, {static_cast<int>(input.shape()[0]), 1});
    auto result = convolve1D(normalizedInput, kernel, tempOptions);

    return Tensor1D<ResultType>(xt::reshape_view(result, {result.shape()[0]}));
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ convolve1D - end                                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ convolve2D - begin                                                                                               ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

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
    KernelContainerType kernel = kernelExpression.derived_cast();

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

    if (kernel.dimension() != 4) {
        throw std::invalid_argument("convolve2D(): Need a full 4 dimensional (C_{out} x C_{in} x K_{H} x K_{W}) kernel to operate!");
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
                            indexY = getBorderIndex<true>(treatmentY, indexY, inputHeight);
                        } else if(inputHeight <= indexY) {
                            treatmentY = optionsY.borderTreatmentEnd;
                            indexY = getBorderIndex<false>(treatmentY, indexY, inputHeight);
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
                                    indexX = getBorderIndex<true>(treatmentX, indexX, inputWidth);
                                } else if(inputWidth <= indexX) {
                                    treatmentX = optionsX.borderTreatmentEnd;
                                    indexX = getBorderIndex<false>(treatmentX, indexX, inputWidth);
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
                        indexY = getBorderIndex<true>(treatmentY, indexY, inputHeight);
                    } else if(inputHeight <= indexY) {
                        treatmentY = optionsY.borderTreatmentEnd;
                        indexY = getBorderIndex<false>(treatmentY, indexY, inputHeight);
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
                                indexX = getBorderIndex<true>(treatmentX, indexX, inputWidth);
                            } else if(inputWidth <= indexX) {
                                treatmentX = optionsX.borderTreatmentEnd;
                                indexX = getBorderIndex<false>(treatmentX, indexX, inputWidth);
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

    if (kernel.dimension() != 4) {
        throw std::invalid_argument("convolve2DImplicit(): Need a full 4 dimensional (C_{out} x C_{in} x K_{H} x K_{W}) kernel to operate!");
    }

    xvigra::KernelOptions2D tempOptions(options2D);
    tempOptions.setChannelPosition(xvigra::ChannelPosition::LAST);
    auto normalizedInput = xt::reshape_view(input, {
        static_cast<int>(input.shape()[0]), 
        static_cast<int>(input.shape()[1]),
        1
    });
    auto result = convolve2D(normalizedInput, kernel, tempOptions);

    return Tensor2D<ResultType>(xt::reshape_view(result, {
        result.shape()[0],
        result.shape()[1]
    }));
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ convolve2D - end                                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
} // xvigra

#endif // XVIGRA_EXPLICIT_CONVOLUTION_HPP