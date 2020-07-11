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
#include "xvigra/iter_util.hpp"

namespace xvigra_legacy {
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
using Tensor4D = typename xt::xtensor<T, 4>;
template <typename T>
using Tensor5D = typename xt::xtensor<T, 5>;

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ type definitions - end                                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ utility legacy - begin                                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

inline int calculateOutputSize(
    int inputSize, 
    int kernelSize, 
    int paddingTotal,
    int stride,
    int dilation
) {
    return static_cast<int>(std::floor((static_cast<double>(inputSize + paddingTotal - dilation * (kernelSize - 1) - 1) / stride) + 1));
}


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

#define XVIGRA_LEGACY_SET_BEGIN_BORDER_VALUE_CHANNEL_FIRST {                          \
    InputType value;                                                                  \
    auto treatment = options.borderTreatmentBegin;                                    \
                                                                                      \
    switch(treatment.getType()) {                                                     \
        case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {                       \
            value = input(inputChannel, std::abs(inputX));                            \
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
            value = input(inputChannel, 0);                                           \
            break;                                                                    \
        }                                                                             \
        case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {                        \
            value = input(inputChannel, std::abs(inputX + 1));                        \
            break;                                                                    \
        }                                                                             \
        case xvigra::BorderTreatmentType::WRAP: {                                     \
            value = input(inputChannel, inputX + inputWidth);                         \
            break;                                                                    \
        }                                                                             \
        default: {                                                                    \
            throw std::domain_error("convolve1D(): Unknown begin border treatment!"); \
        }                                                                             \
    }                                                                                 \
                                                                                      \
    patch(inputChannel, patchKernelX, outIndex) = value;                              \
}

#define XVIGRA_LEGACY_SET_END_BORDER_VALUE_CHANNEL_FIRST {\
    InputType value;\
    auto treatment = options.borderTreatmentEnd;\
    \
    switch(treatment.getType()) {\
        case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {\
            value = input(inputChannel, 2 * inputWidth - inputX - 2); \
            break;\
        }\
        case xvigra::BorderTreatmentType::AVOID: {\
            throw std::domain_error(\
                "convolve1D(): Border treatment AVOID should not be used here!"\
            );\
        }\
        case xvigra::BorderTreatmentType::CONSTANT: {\
            value = treatment.getValue<InputType>();\
            break;\
        }\
        case xvigra::BorderTreatmentType::REPEAT: {\
            value = input(inputChannel, inputWidth - 1);\
            break;\
        }\
        case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {\
            value = input(inputChannel, 2 * inputWidth - inputX - 1); \
            break;\
        }\
        case xvigra::BorderTreatmentType::WRAP: {\
            value = input(inputChannel, inputX - inputWidth); \
            break;\
        }\
        default: {\
            throw std::domain_error("convolve1D(): Unknown end border treatment!");\
        }\
    }\
    \
    patch(inputChannel, patchKernelX, outIndex) = value;\
}

#define XVIGRA_LEGACY_SET_BEGIN_BORDER_VALUE_CHANNEL_LAST {                                     \
    InputType value;                                                                  \
    auto treatment = options.borderTreatmentBegin;                                            \
                                                                                      \
    switch(treatment.getType()) {                                                     \
        case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {                       \
            value = input(std::abs(inputX), inputChannel);                        \
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
            value = input(0, inputChannel);                                           \
            break;                                                                    \
        }                                                                             \
        case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {                        \
            value = input(std::abs(inputX + 1), inputChannel);                            \
            break;                                                                    \
        }                                                                             \
        case xvigra::BorderTreatmentType::WRAP: {                                     \
            value = input(inputX + inputWidth, inputChannel);                         \
            break;                                                                    \
        }                                                                             \
        default: {                                                                    \
            throw std::domain_error("convolve1D(): Unknown begin border treatment!"); \
        }                                                                             \
    }                                                                                 \
                                                                                      \
    patch(outIndex, inputChannel, patchKernelX) = value;\
}

#define XVIGRA_LEGACY_SET_END_BORDER_VALUE_CHANNEL_LAST {\
    InputType value;\
    auto treatment = options.borderTreatmentEnd;\
    \
    switch(treatment.getType()) {\
        case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {\
            value = input(2 * inputWidth - inputX - 2, inputChannel); \
            break;\
        }\
        case xvigra::BorderTreatmentType::AVOID: {\
            throw std::domain_error(\
                "convolve1D(): Border treatment AVOID should not be used here!"\
            );\
        }\
        case xvigra::BorderTreatmentType::CONSTANT: {\
            value = treatment.getValue<InputType>();\
            break;\
        }\
        case xvigra::BorderTreatmentType::REPEAT: {\
            value = input(inputWidth - 1, inputChannel);\
            break;\
        }\
        case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {\
            value = input(2 * inputWidth - inputX - 1, inputChannel); \
            break;\
        }\
        case xvigra::BorderTreatmentType::WRAP: {\
            value = input(inputX - inputWidth, inputChannel); \
            break;\
        }\
        default: {\
            throw std::domain_error("convolve1D(): Unknown end border treatment!");\
        }\
    }\
    \
    patch(outIndex, inputChannel, patchKernelX) = value;\
}

#define XVIGRA_LEGACY_GET_BEGIN_BORDER_INDEX(value, treatment, index, size) {\
    switch((treatment).getType()) {\
        case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {\
            (value) = std::abs((index)); \
            break;\
        }\
        case xvigra::BorderTreatmentType::AVOID: {\
            throw std::domain_error(\
                "getBorderIndex(): Border treatment AVOID should not be used here!"\
            );\
        }\
        case xvigra::BorderTreatmentType::CONSTANT: {\
            (value) = -1;\
            break;\
        }\
        case xvigra::BorderTreatmentType::REPEAT: {\
            (value) = 0; \
            break;\
        }\
        case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {\
            (value) = std::abs((index) + 1); \
            break;\
        }\
        case xvigra::BorderTreatmentType::WRAP: {\
            (value) = (index) + (size); \
            break;\
        }\
        default: {\
            throw std::domain_error("getBorderIndex(): Unknown begin border treatment!");\
        }\
    }\
}

#define XVIGRA_LEGACY_GET_END_BORDER_INDEX(value, treatment, index, size) {\
    switch((treatment).getType()) {\
        case xvigra::BorderTreatmentType::ASYMMETRIC_REFLECT: {\
            (value) = 2 * (size) - (index) - 2; \
            break;\
        }\
        case xvigra::BorderTreatmentType::AVOID: {\
            throw std::domain_error(\
                "getBorderIndex(): Border treatment AVOID should not be used here!"\
            );\
        }\
        case xvigra::BorderTreatmentType::CONSTANT: {\
            (value) = -1;\
            break;\
        }\
        case xvigra::BorderTreatmentType::REPEAT: {\
            (value) = (size) - 1; \
            break;\
        }\
        case xvigra::BorderTreatmentType::SYMMETRIC_REFLECT: {\
            (value) = 2 * (size) - (index) - 1; \
            break;\
        }\
        case xvigra::BorderTreatmentType::WRAP: {\
            (value) = (index) - (size); \
            break;\
        }\
        default: {\
            throw std::domain_error("getBorderIndex(): Unknown end border treatment!");\
        }\
    }\
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ utility legacy - end                                                                                             ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ convolve1D legacy - begin   ´                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 1 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

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
                                    value = input(inputChannel, std::abs(inputX)); 
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
                                    value = input(inputChannel, std::abs(inputX + 1)); 
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
                                    value = input(inputChannel, inputWidth - 1); 
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
                                    value = input(std::abs(inputX), inputChannel); 
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
                                    value = input(std::abs(inputX + 1), inputChannel); 
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
                                    value = input(inputWidth - 1, inputChannel); 
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

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 1 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 2 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

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

        int outputWidth = xvigra_legacy::calculateOutputSize(inputWidth, size, paddingTotal, stride, dilation);
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
                                    value = input(inputChannel, std::abs(inputX)); 
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
                                    value = input(inputChannel, std::abs(inputX + 1)); 
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
                                    value = input(inputChannel, inputWidth - 1); 
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
                                    value = input(std::abs(inputX), inputChannel); 
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
                                    value = input(std::abs(inputX + 1), inputChannel); 
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
                                    value = input(inputWidth - 1, inputChannel); 
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

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 2 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 3 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

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

        int outputWidth = xvigra_legacy::calculateOutputSize(inputWidth, size, paddingTotal, stride, dilation);
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
                                    value = input(inputChannel, std::abs(inputX)); 
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
                                    value = input(inputChannel, std::abs(inputX + 1)); 
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
                                    value = input(inputChannel, inputWidth - 1); 
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
                                    value = input(std::abs(inputX), inputChannel); 
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
                                    value = input(std::abs(inputX + 1), inputChannel); 
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
                                    value = input(inputWidth - 1, inputChannel); 
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

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 3 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 4 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

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

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 4 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 5 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

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

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 5 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 6 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename InputType, typename KernelType>
    Tensor2D<typename std::common_type_t<InputType, KernelType>> convolve1D_v6(
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
                            XVIGRA_LEGACY_SET_BEGIN_BORDER_VALUE_CHANNEL_FIRST
                        } else if(inputWidth <= inputX) {
                            XVIGRA_LEGACY_SET_END_BORDER_VALUE_CHANNEL_FIRST
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
                            XVIGRA_LEGACY_SET_BEGIN_BORDER_VALUE_CHANNEL_LAST
                        }
                    } else if(inputWidth <= inputX) {
                        for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                            XVIGRA_LEGACY_SET_END_BORDER_VALUE_CHANNEL_LAST
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

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve1D version 6 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ convolve1D legacy - end     ´                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ convolve2D legacy - begin   ´                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 1 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T, typename O>
    auto convolve2D_v1(
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

        std::vector<int> inputHeightIndices = xvigra::range(heightMinimum, heightMaximum, optionsY.stride);
        std::vector<int> inputWidthIndices = xvigra::range(widthMinimum, widthMaximum, optionsX.stride);
        
        xt::xtensor<ResultType, 3> result;
        if (optionsY.channelPosition == xvigra::ChannelPosition::FIRST) {
           Tensor5D<ResultType> patch = xt::zeros<ResultType>({inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth});
            
            for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                for (auto kernelY = kernelHeightMinimum; kernelY < kernelHeightMaximum; ++kernelY) {
                    for (auto kernelX = kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
                        for (std::size_t outIndexY = 0; outIndexY < inputHeightIndices.size(); ++outIndexY) {
                            auto inputY = inputHeightIndices.at(outIndexY) + kernelY * optionsY.dilation;
                            auto outKernelY = kernelY + std::abs(kernelHeightMinimum);
                            
                            for (std::size_t outIndexX = 0; outIndexX < inputWidthIndices.size(); ++outIndexX) {
                                auto inputX = inputWidthIndices.at(outIndexX) + kernelX * optionsX.dilation;
                                auto outKernelX = kernelX + std::abs(kernelWidthMinimum);
                                
                                if((0 <= inputY && inputY < inputHeight) && (0 <= inputX && inputX < inputWidth)) {
                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(input(inputChannel, inputY, inputX));
                                } else {
                                    xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                                    int indexY = inputY;

                                    if (indexY < 0) {
                                        treatmentY = optionsY.borderTreatmentBegin;
                                        indexY = getBorderIndex<true>(treatmentY, indexY, inputHeight);
                                    } else if(inputHeight <= indexY) {
                                        treatmentY = optionsY.borderTreatmentEnd;
                                        indexY = getBorderIndex<false>(treatmentY, indexY, inputHeight);
                                    }

                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (indexX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        indexX = getBorderIndex<true>(treatmentX, indexX, inputWidth);
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        indexX = getBorderIndex<false>(treatmentX, indexX, inputWidth);
                                    }

                                    InputType value;
                                    if (indexY == -1 || indexX == -1) {
                                        if (indexY == -1 && indexX != -1) {
                                            value = treatmentY.getValue<InputType>();
                                        } else if (indexX == -1 && indexY != -1) {
                                            value = treatmentX.getValue<InputType>();
                                        } else {
                                            value = treatmentX.getValue<InputType>();
                                        }
                                    } else {
                                        value = input(inputChannel, indexY, indexX);
                                    }

                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(value);  
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
            
            for (std::size_t outIndexY = 0; outIndexY < inputHeightIndices.size(); ++outIndexY) {
                for (std::size_t outIndexX = 0; outIndexX < inputWidthIndices.size(); ++outIndexX) {
                    for (auto kernelY = kernelHeightMinimum; kernelY < kernelHeightMaximum; ++kernelY) {
                        auto inputY = inputHeightIndices.at(outIndexY) + kernelY * optionsY.dilation;
                        auto outKernelY = kernelY + std::abs(kernelHeightMinimum);
                        
                        for (auto kernelX = kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
                            auto inputX = inputWidthIndices.at(outIndexX) + kernelX * optionsX.dilation;
                            auto outKernelX = kernelX + std::abs(kernelWidthMinimum);
                            
                            if((0 <= inputY && inputY < inputHeight) && (0 <= inputX && inputX < inputWidth)){
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(input(inputY, inputX, inputChannel));
                                }
                            } else {
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                                    int indexY = inputY;

                                    if (indexY < 0) {
                                        treatmentY = optionsY.borderTreatmentBegin;
                                        indexY = getBorderIndex<true>(treatmentY, indexY, inputHeight);
                                    } else if(inputHeight <= indexY) {
                                        treatmentY = optionsY.borderTreatmentEnd;
                                        indexY = getBorderIndex<false>(treatmentY, indexY, inputHeight);
                                    }

                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (inputX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        indexX = getBorderIndex<true>(treatmentX, indexX, inputWidth);
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        indexX = getBorderIndex<false>(treatmentX, indexX, inputWidth);
                                    }

                                    InputType value;
                                    if (indexY == -1 || indexX == -1) {
                                        if (indexY == -1 && indexX != -1) {
                                            value = treatmentY.getValue<InputType>();
                                        } else if (indexX == -1 && indexY != -1) {
                                            value = treatmentX.getValue<InputType>();
                                        } else {
                                            value = treatmentX.getValue<InputType>();
                                        }
                                    } else {
                                        value = input(indexY, indexX, inputChannel);
                                    }

                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(value); 
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
    inline auto convolve2D_v1(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions2D& options2D
    ) {
        return convolve2D_v1(
            inputExpression.derived_cast(), 
            kernelExpression.derived_cast(), 
            options2D.optionsY, 
            options2D.optionsX
        );
    }

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 1 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 2 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T, typename O>
    auto convolve2D_v2(
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

        std::vector<int> inputHeightIndices = xvigra::range(heightMinimum, heightMaximum, optionsY.stride);
        std::vector<int> inputWidthIndices = xvigra::range(widthMinimum, widthMaximum, optionsX.stride);
        
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

                        for (std::size_t outIndexY = 0; outIndexY < inputHeightIndices.size(); ++outIndexY) {
                            auto inputY = inputHeightIndices.at(outIndexY) + inputOffsetY;
                            
                            for (std::size_t outIndexX = 0; outIndexX < inputWidthIndices.size(); ++outIndexX) {
                                auto inputX = inputWidthIndices.at(outIndexX) + inputOffsetX;
                                
                                if((0 <= inputY && inputY < inputHeight) && (0 <= inputX && inputX < inputWidth)) {
                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(input(inputChannel, inputY, inputX));
                                } else {
                                    xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                                    int indexY = inputY;

                                    if (indexY < 0) {
                                        treatmentY = optionsY.borderTreatmentBegin;
                                        indexY = getBorderIndex<true>(treatmentY, indexY, inputHeight);
                                    } else if(inputHeight <= indexY) {
                                        treatmentY = optionsY.borderTreatmentEnd;
                                        indexY = getBorderIndex<false>(treatmentY, indexY, inputHeight);
                                    }

                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (indexX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        indexX = getBorderIndex<true>(treatmentX, indexX, inputWidth);
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        indexX = getBorderIndex<false>(treatmentX, indexX, inputWidth);
                                    }

                                    InputType value;
                                    if (indexY == -1 || indexX == -1) {
                                        if (indexY == -1 && indexX != -1) {
                                            value = treatmentY.getValue<InputType>();
                                        } else if (indexX == -1 && indexY != -1) {
                                            value = treatmentX.getValue<InputType>();
                                        } else {
                                            value = treatmentX.getValue<InputType>();
                                        }
                                    } else {
                                        value = input(inputChannel, indexY, indexX);
                                    }

                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(value);  
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
            
            for (std::size_t outIndexY = 0; outIndexY < inputHeightIndices.size(); ++outIndexY) {
                auto inputIndexY = inputHeightIndices.at(outIndexY);

                for (std::size_t outIndexX = 0; outIndexX < inputWidthIndices.size(); ++outIndexX) {
                    auto inputIndexX = inputWidthIndices.at(outIndexX);

                    for (auto kernelY = kernelHeightMinimum; kernelY < kernelHeightMaximum; ++kernelY) {
                        auto inputY = inputIndexY + kernelY * optionsY.dilation;
                        auto outKernelY = kernelY + std::abs(kernelHeightMinimum);
                        
                        for (auto kernelX = kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
                            auto inputX = inputIndexX + kernelX * optionsX.dilation;
                            auto outKernelX = kernelX + std::abs(kernelWidthMinimum);
                            
                            if((0 <= inputY && inputY < inputHeight) && (0 <= inputX && inputX < inputWidth)){
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(input(inputY, inputX, inputChannel));
                                }
                            } else {
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                                    int indexY = inputY;

                                    if (indexY < 0) {
                                        treatmentY = optionsY.borderTreatmentBegin;
                                        indexY = getBorderIndex<true>(treatmentY, indexY, inputHeight);
                                    } else if(inputHeight <= indexY) {
                                        treatmentY = optionsY.borderTreatmentEnd;
                                        indexY = getBorderIndex<false>(treatmentY, indexY, inputHeight);
                                    }

                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (inputX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        indexX = getBorderIndex<true>(treatmentX, indexX, inputWidth);
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        indexX = getBorderIndex<false>(treatmentX, indexX, inputWidth);
                                    }

                                    InputType value;
                                    if (indexY == -1 || indexX == -1) {
                                        if (indexY == -1 && indexX != -1) {
                                            value = treatmentY.getValue<InputType>();
                                        } else if (indexX == -1 && indexY != -1) {
                                            value = treatmentX.getValue<InputType>();
                                        } else {
                                            value = treatmentX.getValue<InputType>();
                                        }
                                    } else {
                                        value = input(indexY, indexX, inputChannel);
                                    }

                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(value); 
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
    inline auto convolve2D_v2(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions2D& options2D
    ) {
        return convolve2D_v2(
            inputExpression.derived_cast(), 
            kernelExpression.derived_cast(), 
            options2D.optionsY, 
            options2D.optionsX
        );
    }

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 2 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 3 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T, typename O>
    auto convolve2D_v3(
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
                            
                            for (auto [outIndexX, inputIndexX] : xvigra::enumerate(inputWidthIndices)) {
                                auto inputX = inputIndexX + inputOffsetX;
                                
                                if((0 <= inputY && inputY < inputHeight) && (0 <= inputX && inputX < inputWidth)) {
                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(input(inputChannel, inputY, inputX));
                                } else {
                                    xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                                    int indexY = inputY;

                                    if (indexY < 0) {
                                        treatmentY = optionsY.borderTreatmentBegin;
                                        indexY = getBorderIndex<true>(treatmentY, indexY, inputHeight);
                                    } else if(inputHeight <= indexY) {
                                        treatmentY = optionsY.borderTreatmentEnd;
                                        indexY = getBorderIndex<false>(treatmentY, indexY, inputHeight);
                                    }

                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (indexX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        indexX = getBorderIndex<true>(treatmentX, indexX, inputWidth);
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        indexX = getBorderIndex<false>(treatmentX, indexX, inputWidth);
                                    }

                                    InputType value;
                                    if (indexY == -1 || indexX == -1) {
                                        if (indexY == -1 && indexX != -1) {
                                            value = treatmentY.getValue<InputType>();
                                        } else if (indexX == -1 && indexY != -1) {
                                            value = treatmentX.getValue<InputType>();
                                        } else {
                                            value = treatmentX.getValue<InputType>();
                                        }
                                    } else {
                                        value = input(inputChannel, indexY, indexX);
                                    }

                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(value);  
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
                        
                        for (auto kernelX = kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
                            auto inputX = inputIndexX + kernelX * optionsX.dilation;
                            auto outKernelX = kernelX + std::abs(kernelWidthMinimum);
                            
                            if((0 <= inputY && inputY < inputHeight) && (0 <= inputX && inputX < inputWidth)){
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(input(inputY, inputX, inputChannel));
                                }
                            } else {
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                                    int indexY = inputY;

                                    if (indexY < 0) {
                                        treatmentY = optionsY.borderTreatmentBegin;
                                        indexY = getBorderIndex<true>(treatmentY, indexY, inputHeight);
                                    } else if(inputHeight <= indexY) {
                                        treatmentY = optionsY.borderTreatmentEnd;
                                        indexY = getBorderIndex<false>(treatmentY, indexY, inputHeight);
                                    }

                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (inputX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        indexX = getBorderIndex<true>(treatmentX, indexX, inputWidth);
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        indexX = getBorderIndex<false>(treatmentX, indexX, inputWidth);
                                    }

                                    InputType value;
                                    if (indexY == -1 || indexX == -1) {
                                        if (indexY == -1 && indexX != -1) {
                                            value = treatmentY.getValue<InputType>();
                                        } else if (indexX == -1 && indexY != -1) {
                                            value = treatmentX.getValue<InputType>();
                                        } else {
                                            value = treatmentX.getValue<InputType>();
                                        }
                                    } else {
                                        value = input(indexY, indexX, inputChannel);
                                    }

                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(value); 
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
    inline auto convolve2D_v3(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions2D& options2D
    ) {
        return convolve2D_v3(
            inputExpression.derived_cast(), 
            kernelExpression.derived_cast(), 
            options2D.optionsY, 
            options2D.optionsX
        );
    }

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 3 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 4 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T, typename O>
    auto convolve2D_v4(
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
    inline auto convolve2D_v4(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions2D& options2D
    ) {
        return convolve2D_v4(
            inputExpression.derived_cast(), 
            kernelExpression.derived_cast(), 
            options2D.optionsY, 
            options2D.optionsX
        );
    }

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 4 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 5 - begin                                                                                  ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T, typename O>
    auto convolve2D_v5(
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
                            
                            for (auto [outIndexX, inputIndexX] : xvigra::enumerate(inputWidthIndices)) {
                                auto inputX = inputIndexX + inputOffsetX;
                                
                                if((0 <= inputY && inputY < inputHeight) && (0 <= inputX && inputX < inputWidth)) {
                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(input(inputChannel, inputY, inputX));
                                } else {
                                    xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                                    int indexY = inputY;

                                    if (indexY < 0) {
                                        treatmentY = optionsY.borderTreatmentBegin;
                                        XVIGRA_LEGACY_GET_BEGIN_BORDER_INDEX(indexY, treatmentY, indexY, inputHeight)
                                    } else if(inputHeight <= indexY) {
                                        treatmentY = optionsY.borderTreatmentEnd;
                                        XVIGRA_LEGACY_GET_END_BORDER_INDEX(indexY, treatmentY, indexY, inputHeight)
                                    }

                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (indexX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        XVIGRA_LEGACY_GET_BEGIN_BORDER_INDEX(indexX, treatmentX, indexX, inputWidth)
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        XVIGRA_LEGACY_GET_END_BORDER_INDEX(indexX, treatmentX, indexX, inputWidth)
                                    }

                                    InputType value;
                                    if (indexY == -1 || indexX == -1) {
                                        if (indexY == -1 && indexX != -1) {
                                            value = treatmentY.getValue<InputType>();
                                        } else if (indexX == -1 && indexY != -1) {
                                            value = treatmentX.getValue<InputType>();
                                        } else {
                                            value = treatmentX.getValue<InputType>();
                                        }
                                    } else {
                                        value = input(inputChannel, indexY, indexX);
                                    }

                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = static_cast<ResultType>(value);  
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
                        
                        for (auto kernelX = kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
                            auto inputX = inputIndexX + kernelX * optionsX.dilation;
                            auto outKernelX = kernelX + std::abs(kernelWidthMinimum);
                            
                            if((0 <= inputY && inputY < inputHeight) && (0 <= inputX && inputX < inputWidth)){
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(input(inputY, inputX, inputChannel));
                                }
                            } else {
                                for (auto inputChannel = 0; inputChannel < inputChannels; ++inputChannel) {
                                    xvigra::BorderTreatment treatmentY = xvigra::BorderTreatment::avoid();
                                    int indexY = inputY;

                                    if (indexY < 0) {
                                        treatmentY = optionsY.borderTreatmentBegin;
                                        XVIGRA_LEGACY_GET_BEGIN_BORDER_INDEX(indexY, treatmentY, indexY, inputHeight)
                                    } else if(inputHeight <= indexY) {
                                        treatmentY = optionsY.borderTreatmentEnd;
                                        XVIGRA_LEGACY_GET_END_BORDER_INDEX(indexY, treatmentY, indexY, inputHeight)
                                    }

                                    xvigra::BorderTreatment treatmentX = xvigra::BorderTreatment::avoid();
                                    int indexX = inputX;

                                    if (inputX < 0) {
                                        treatmentX = optionsX.borderTreatmentBegin;
                                        XVIGRA_LEGACY_GET_BEGIN_BORDER_INDEX(indexX, treatmentX, indexX, inputWidth)
                                    } else if(inputWidth <= indexX) {
                                        treatmentX = optionsX.borderTreatmentEnd;
                                        XVIGRA_LEGACY_GET_END_BORDER_INDEX(indexX, treatmentX, indexX, inputWidth)
                                    }

                                    InputType value;
                                    if (indexY == -1 || indexX == -1) {
                                        if (indexY == -1 && indexX != -1) {
                                            value = treatmentY.getValue<InputType>();
                                        } else if (indexX == -1 && indexY != -1) {
                                            value = treatmentX.getValue<InputType>();
                                        } else {
                                            value = treatmentX.getValue<InputType>();
                                        }
                                    } else {
                                        value = input(indexY, indexX, inputChannel);
                                    }

                                    patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = static_cast<ResultType>(value); 
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
    inline auto convolve2D_v5(
        const xt::xexpression<T>& inputExpression,
        const xt::xexpression<O>& kernelExpression,
        const xvigra::KernelOptions2D& options2D
    ) {
        return convolve2D_v5(
            inputExpression.derived_cast(), 
            kernelExpression.derived_cast(), 
            options2D.optionsY, 
            options2D.optionsX
        );
    }

//  ╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
//  ║ convolve2D version 5 - end                                                                                    ║
//  ╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ convolve2D legacy - end     ´                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

} // xvigra_legacy

#endif // XVIGRA_LEGACY_EXPLICIT_CONVOLUTION_HPP