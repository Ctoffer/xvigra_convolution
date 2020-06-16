#ifndef XVIGRA_CONV1D_HPP
#define XVIGRA_CONV1D_HPP

#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "xtensor/xtensor.hpp"
#include "xtensor-blas/xlinalg.hpp"

#include "convolution-util.hpp"

class Conv1D {
private:
	int paddingX;
	int dilationX;
	int strideX;
	
public:
	Conv1D(int paddingX, int dilationX, int strideX):
		paddingX(paddingX), dilationX(dilationX), strideX(strideX)
	{}
	
	template <typename ElementType>
	xt::xtensor<ElementType, 2> operator()(const xt::xtensor<ElementType, 2>& input, 
	                                       const xt::xtensor<ElementType, 3>& kernel,
										   const bool channelFirst=true) {
		InputDimensions1D<int> inputDimensions(input, !channelFirst);
		FilterSpecs1D<int> filterSpecs(static_cast<int>(kernel.shape()[2]));

		if(inputDimensions.inputChannels != static_cast<int>(kernel.shape()[1])) {// dimension mismatch
			throw std::invalid_argument("Input channels of input and kernel do not align!");
		}
		
		if(inputDimensions.inputWidth + 2 * paddingX < (filterSpecs.size - 1) * dilationX + 1) {// size mismatch
			throw std::invalid_argument("Kernel width is greater than padded input width!");
		}

		InputOutputBridge1D<int> inputOutputBridge(inputDimensions, filterSpecs, {paddingX, dilationX, strideX});
	
		xt::xtensor<ElementType, 2> result;
		
		if(channelFirst) {
			result = convolutionChannelOuterDimension(inputDimensions, filterSpecs, inputOutputBridge, input, kernel);
		} 
		else {
			result = convolutionChannelInnerDimension(inputDimensions, filterSpecs, inputOutputBridge, input, kernel);
		}
	
		return result;
	}

	template <typename ElementType>
	xt::xtensor<ElementType, 2> convolutionChannelOuterDimension(const InputDimensions1D<int>& inputDimensions, 
																 const FilterSpecs1D<int>& filterSpecs,
																 const InputOutputBridge1D <int>& inputOutputBridge,
																 const xt::xtensor<ElementType, 2>& input, 
	                                       						 const xt::xtensor<ElementType, 3>& kernel
																 ) {
		xt::xtensor<ElementType, 3> patch = xt::zeros<ElementType>({
			inputDimensions.inputChannels, 
			filterSpecs.size, 
			inputOutputBridge.outputWidth
		});
		int outputChannels = kernel.shape()[0];
		
		for(auto inputChannel=0; inputChannel < inputDimensions.inputChannels; ++inputChannel) {
			for(auto kernelX = filterSpecs.minimum; kernelX < filterSpecs.maximum; ++kernelX) {
				auto kernelOffsetX = dilationX*kernelX;
				auto patchKernelX = kernelX + std::abs(filterSpecs.minimum);
				
				for(std::size_t outIndex = 0; outIndex < inputOutputBridge.inputWidthIndices.size(); ++outIndex) {
					auto inputX = inputOutputBridge.inputWidthIndices.at(outIndex) + kernelOffsetX;
					
					if(0 <= inputX && inputX < inputDimensions.inputWidth) {
						patch(inputChannel, patchKernelX, outIndex) = input(inputChannel, inputX);	
					} 
				}
			}
		}
		
		auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputDimensions.inputChannels * filterSpecs.size});
		auto reshapedPatch = xt::reshape_view(patch, {inputDimensions.inputChannels*filterSpecs.size, inputOutputBridge.outputWidth});
		return xt::linalg::dot(reshapedKernel, reshapedPatch);
	}

	template <typename ElementType>
	xt::xtensor<ElementType, 2> convolutionChannelInnerDimension(const InputDimensions1D<int>& inputDimensions, 
																 const FilterSpecs1D<int>& filterSpecs,
																 const InputOutputBridge1D <int>& inputOutputBridge,
																 const xt::xtensor<ElementType, 2>& input, 
	                                       						 const xt::xtensor<ElementType, 3>& kernel
																 ) {
		xt::xtensor<ElementType, 3> patch = xt::zeros<ElementType>({
			inputOutputBridge.outputWidth, 
			inputDimensions.inputChannels, 
			filterSpecs.size
		});
		int outputChannels = kernel.shape()[0];
		
		for(std::size_t outIndex = 0; outIndex < inputOutputBridge.inputWidthIndices.size(); ++outIndex) {
			for(auto kernelX = filterSpecs.minimum; kernelX < filterSpecs.maximum; ++kernelX) {
				auto inputX = inputOutputBridge.inputWidthIndices.at(outIndex) + dilationX*kernelX;
				auto patchKernelX = kernelX + std::abs(filterSpecs.minimum);
					
				if(0 <= inputX && inputX < inputDimensions.inputWidth) {
					for(auto inputChannel=0; inputChannel < inputDimensions.inputChannels; ++inputChannel) {
						patch(outIndex, inputChannel, patchKernelX) = input(inputX, inputChannel);
					} 
				}
			}
		}
		
		auto reshapedKernel = xt::reshape_view(kernel, {outputChannels, inputDimensions.inputChannels * filterSpecs.size});
		auto reshapedPatch = xt::reshape_view(patch, {inputOutputBridge.outputWidth, filterSpecs.size*inputDimensions.inputChannels});
		return xt::linalg::dot(reshapedPatch, xt::transpose(reshapedKernel));
	}
};


#endif