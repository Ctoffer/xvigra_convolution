#ifndef XVIGRA_CONVOLUTION_UTIL_HPP
#define XVIGRA_CONVOLUTION_UTIL_HPP

#include <tuple>
#include <vector>

#include "xtensor/xexpression.hpp"

// --------------------------------------------------------------------------------------------------------------------
// 1D Utility
// --------------------------------------------------------------------------------------------------------------------


struct Conv1DParameters {
    int padding;
    int dilation;
    int stride;
};

inline int calculateOutputSize(int inputSize, 
                               int kernelSize, 
	                           int padding, 
	                           int dilation, 
	                           int stride) {
	return static_cast<int>(std::floor((static_cast<double>(inputSize + 2 * padding - dilation * (kernelSize - 1) - 1) / stride) + 1));
}

inline int calculateOutputSize(int inputSize, int kernelSize, const Conv1DParameters& parameters) {
    return calculateOutputSize(inputSize, kernelSize, parameters.padding, parameters.dilation, parameters.stride);
}


template <typename T=int>
struct InputDimensions1D {
    T inputChannels;
    T inputWidth;

    template <typename E>
    InputDimensions1D(const E& xContainer, bool channelIsInnerDimension=true) {
        if (channelIsInnerDimension) {
            inputWidth = static_cast<T>(xContainer.shape()[0]);
            inputChannels = static_cast<T>(xContainer.shape()[1]);
        } else {
            inputChannels = static_cast<T>(xContainer.shape()[0]);
            inputWidth = static_cast<T>(xContainer.shape()[1]);
        }
    }
};

template <typename T=int>
struct FilterSpecs1D {
    T size;
    T radius;
    T minimum;
    T maximum;

    FilterSpecs1D(const T size) {
        this->size = size;
        radius = size / 2;
        if(size % 2 == 0) {
            minimum = 0;
            maximum = size;
        } else {
            minimum = -radius;
            maximum = radius + 1;
        }
    }
};

template <typename T=int>
std::vector<T> range(T start, T stop, T step) {
    std::vector<int> result;
    for(T i=start; i<stop; i+=step) {
        result.push_back(i);
    }
    return result;
}

template <typename T=int>
struct InputOutputBridge1D {
    T inputWidthMinimum;
    T inputWidthMaximum;
    T outputWidth;
    std::vector<T> inputWidthIndices;

    InputOutputBridge1D(const InputDimensions1D<T>& inputDimensions, 
                        const FilterSpecs1D<T>& filterSpecs,
                        const Conv1DParameters& parameters) {
        if(filterSpecs.size % 2 == 0) {
            inputWidthMinimum = -parameters.padding;
            inputWidthMaximum = inputDimensions.inputWidth 
                                + parameters.padding 
                                - parameters.dilation * (filterSpecs.size - 1);
        } else {
            inputWidthMinimum = -parameters.padding
                                + parameters.dilation * std::abs(filterSpecs.minimum);
            inputWidthMaximum = inputDimensions.inputWidth 
                                + parameters.padding
                                - parameters.dilation * (filterSpecs.radius);
        }

        outputWidth = calculateOutputSize(inputDimensions.inputWidth, filterSpecs.size, parameters);
        inputWidthIndices = range(inputWidthMinimum, inputWidthMaximum, parameters.stride);
    }
};


// --------------------------------------------------------------------------------------------------------------------
// 2D Utility
// --------------------------------------------------------------------------------------------------------------------

inline std::tuple<int, int> calculateOutputSize(const std::tuple<int, int>& inputSize, 
                                                const std::tuple<int, int>& kernelSize, 
                                                const std::tuple<int, int>& padding,
                                                const std::tuple<int, int>& dilation,
                                                const std::tuple<int, int>& stride) {
    auto height = calculateOutputSize(
        std::get<0>(inputSize), 
        std::get<0>(kernelSize), 
        std::get<0>(padding),
        std::get<0>(dilation), 
        std::get<0>(stride)
    );
    auto width = calculateOutputSize(
        std::get<1>(inputSize), 
        std::get<1>(kernelSize), 
        std::get<1>(padding),
        std::get<1>(dilation), 
        std::get<1>(stride)
    );
    return std::make_tuple(height, width);
}


#endif
