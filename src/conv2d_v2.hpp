#ifndef XVIGRA_CONV2D_HPP
#define XVIGRA_CONV2D_HPP

#include <tuple>
#include <type_traits>
#include <vector>

#include "xtensor/xtensor.hpp"
#include "xtensor-blas/xlinalg.hpp"

#include "convolution-util.hpp"



class Conv2D {
private:
	std::tuple<int, int> padding;
	std::tuple<int, int> dilation;
	std::tuple<int, int> stride;
	
public:
	Conv2D(const std::tuple<int, int>& padding, 
	       const std::tuple<int, int>& dilation, 
		   const std::tuple<int, int>& stride):
		padding(padding), dilation(dilation), stride(stride)
	{}
	
	template <typename T, typename O>
	xt::xtensor<typename std::common_type_t<T, O>, 3> operator()(
		                                   const xt::xtensor<T, 3>& input, 
	                                       const xt::xtensor<O, 4>& filter,
										   const bool channelFirst=true) {
		using ResultType = typename std::common_type_t<T, O>;
		
		int inputChannels;
		int inputHeight;
		int inputWidth;
		
		if(channelFirst) {
			inputChannels = input.shape()[0];
			inputHeight = input.shape()[1];
			inputWidth = input.shape()[2];
		} else {
			inputHeight = input.shape()[0];
			inputWidth = input.shape()[1];
			inputChannels = input.shape()[2];
		}
		
		int outputChannels = filter.shape()[0];
		int kernelHeight = filter.shape()[2];
		int kernelWidth = filter.shape()[3];
		
		if(inputChannels != static_cast<int>(filter.shape()[1])) {// dimension mismatch
			throw std::invalid_argument("Input channels of input and kernel do not align!");
		}
		
		// size mismatch
		if(inputHeight + 2 * std::get<0>(padding) < (kernelHeight - 1) * std::get<0>(dilation) + 1) {
			throw std::invalid_argument("Kernel height is greater than padded input height!");
		}
		
		if(inputWidth + 2 * std::get<1>(padding) < (kernelWidth - 1) * std::get<1>(dilation) + 1) {
			throw std::invalid_argument("Kernel width is greater than padded input width!");
		}
		
		
		int kernelHeightRadius = kernelHeight / 2;
		int kernelHeightMinimum = kernelHeight % 2 == 0 ? 0 : -kernelHeightRadius;
		int kernelHeightMaximum = kernelHeight % 2 == 0 ? kernelHeight : kernelHeightRadius + 1;
		
		
		int kernelWidthRadius = kernelWidth / 2;
		int kernelWidthMinimum = kernelWidth % 2 == 0 ? 0 : -kernelWidthRadius;
		int kernelWidthMaximum = kernelWidth % 2 == 0 ? kernelWidth : kernelWidthRadius + 1;
		
		auto outputSizes = calculateOutputSize(
			std::make_tuple(inputHeight, inputWidth), 
			std::make_tuple(kernelHeight, kernelWidth),
			padding,
			dilation,
			stride);
			
		int heightMinimum = 0-std::get<0>(padding) + std::get<0>(dilation)*std::abs(kernelHeight % 2 == 0 ? 0 : kernelHeightMinimum);
		int heightMaximum = inputHeight + std::get<0>(padding) - std::get<0>(dilation)*(kernelHeight % 2 == 0 ? kernelHeight - 1 : kernelHeightRadius);
		int outputHeight = std::get<0>(outputSizes);
		
		int widthMinimum = 0-std::get<1>(padding) + std::get<1>(dilation)*std::abs(kernelWidth % 2 == 0 ? 0 : kernelWidthMinimum);
		int widthMaximum = inputWidth + std::get<1>(padding) - std::get<1>(dilation)*(kernelWidth % 2 == 0 ? kernelWidth - 1 : kernelWidthRadius);
		int outputWidth = std::get<1>(outputSizes);
	
		std::vector<int> heightIndices;
		for(int heightX=heightMinimum; heightX<heightMaximum; heightX+=std::get<0>(stride)) {
			heightIndices.push_back(heightX);
		}
		
		std::vector<int> widthIndices;
		for(int widthX=widthMinimum; widthX<widthMaximum; widthX+=std::get<1>(stride)) {
			widthIndices.push_back(widthX);
		}
		
		xt::xtensor<ResultType, 3> result;
		if(channelFirst) {
			xt::xtensor<T, 5> patch = xt::zeros<T>({inputChannels, kernelHeight, kernelWidth, outputHeight, outputWidth});
			
			for(auto inputChannel=0; inputChannel < inputChannels; ++inputChannel) {
				for(auto kernelY=kernelHeightMinimum; kernelY < kernelHeightMaximum; ++kernelY) {
					for(auto kernelX=kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
						for(std::size_t outIndexY = 0; outIndexY < heightIndices.size(); ++outIndexY) {
							auto idY = heightIndices.at(outIndexY) + kernelY * std::get<0>(dilation);
                            auto outKernelY = kernelY + std::abs(kernelHeightMinimum);
							
							for(std::size_t outIndexX = 0; outIndexX < widthIndices.size(); ++outIndexX) {
								auto idX = widthIndices.at(outIndexX) + kernelX * std::get<1>(dilation);
								auto outKernelX = kernelX + std::abs(kernelWidthMinimum);
								
								if((0 <= idY && idY < inputHeight) && (0 <= idX && idX < inputWidth)) {
                                    patch(inputChannel, outKernelY, outKernelX, outIndexY, outIndexX) = input(inputChannel, idY, idX);
								}
							}
						}
					}
				}
			}
			
			auto reshapedKernel = xt::reshape_view(filter, {outputChannels, inputChannels * kernelHeight * kernelWidth});
			auto reshapedPatch = xt::reshape_view(patch, {inputChannels * kernelHeight * kernelWidth, outputHeight * outputWidth});
			result = xt::reshape_view(xt::linalg::dot(reshapedKernel, reshapedPatch), {outputChannels, outputHeight, outputWidth});
			
		} else {
			xt::xtensor<T, 5> patch = xt::zeros<T>({outputHeight, outputWidth, inputChannels, kernelHeight, kernelWidth});
			
			for(std::size_t outIndexY = 0; outIndexY < heightIndices.size(); ++outIndexY) {
				for(std::size_t outIndexX = 0; outIndexX < widthIndices.size(); ++outIndexX) {
					for(auto kernelY=kernelHeightMinimum; kernelY < kernelHeightMaximum; ++kernelY) {
						auto idY = heightIndices.at(outIndexY) + kernelY * std::get<0>(dilation);
                        auto outKernelY = kernelY + std::abs(kernelHeightMinimum);
						
						for(auto kernelX=kernelWidthMinimum; kernelX < kernelWidthMaximum; ++kernelX) {
							auto idX = widthIndices.at(outIndexX) + kernelX * std::get<1>(dilation);
							auto outKernelX = kernelX + std::abs(kernelWidthMinimum);
							
							if((0 <= idY && idY < inputHeight) && (0 <= idX && idX < inputWidth)){
								for(auto inputChannel=0; inputChannel < inputChannels; ++inputChannel) {
                                	patch(outIndexY, outIndexX, inputChannel, outKernelY, outKernelX) = input(idY, idX, inputChannel);
                                }
							}
						}
					}
				}
			}
			
			
			auto reshapedKernel = xt::transpose(xt::reshape_view(filter, {outputChannels, inputChannels * kernelHeight * kernelWidth}));
			auto reshapedPatch = xt::reshape_view(patch, {outputHeight*outputWidth, inputChannels*kernelHeight*kernelWidth});
			result = xt::reshape_view(xt::linalg::dot(reshapedPatch, reshapedKernel), {outputHeight, outputWidth, outputChannels});
		}
		
		return result;
	}
};

#endif