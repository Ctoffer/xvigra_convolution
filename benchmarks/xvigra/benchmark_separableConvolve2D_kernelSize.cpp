#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#include "xvigra/explicit_convolution.hpp"
#include "xvigra/separable_convolution.hpp"
#include "xvigra/io_util.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - begin                                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

#define INPUT_SIZE_MIN 1
#define INPUT_SIZE_MAX 21
#define INPUT_SIZE_STEP 1


#define BENCHMARK_SINGLE_VERSION(name)\
	BENCHMARK_TEMPLATE(name, float)\
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {\
    	return *(std::min_element(std::begin(v), std::end(v)));\
  	})\
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {\
    	return *(std::max_element(std::begin(v), std::end(v)));\
  	})\
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)\
	->Unit(benchmark::kMillisecond)

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - end                                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolve2D - begin                                                                            ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_separableConvolve2D_kernelSize_channelFirst(benchmark::State& state) {
	int inputHeight = 1080;
	int inputWidth = 1920;
	int inputChannels = 3;
	int kernelHeight = state.range(0);
	int kernelWidth = state.range(0);
	
	std::array<int, 3> inputShape{inputChannels, inputHeight, inputWidth};
	std::array<int, 1> kernelShapeY{kernelHeight};
	std::array<int, 1> kernelShapeX{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions2D options2D;
	options2D.setPadding(padding - 1, padding + 1);
	options2D.setStride(stride + 1, stride - 1);
	options2D.setDilation(dilation - 1, dilation + 1);
	options2D.setChannelPosition(xvigra::ChannelPosition::FIRST);   

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 1> kernelY;
	xt::xtensor<ElementType, 1> kernelX;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernelY = xt::random::rand<ElementType>(kernelShapeY);
		kernelX = xt::random::rand<ElementType>(kernelShapeX);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernelY = xt::random::randint<ElementType>(kernelShapeY);
		kernelX = xt::random::randint<ElementType>(kernelShapeX);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve2D(
		 	input, 
		 	std::array{kernelY, kernelX}, 
		 	options2D
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_separableConvolve2D_kernelSize_channelLast(benchmark::State& state) {
	int inputHeight = 1080;
	int inputWidth = 1920;
	int inputChannels = 3;
	int kernelHeight = state.range(0);
	int kernelWidth = state.range(0);
	
	std::array<int, 3> inputShape{inputHeight, inputWidth, inputChannels};
	std::array<int, 1> kernelShapeY{kernelHeight};
	std::array<int, 1> kernelShapeX{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions2D options2D;
	options2D.setPadding(padding - 1, padding + 1);
	options2D.setStride(stride + 1, stride - 1);
	options2D.setDilation(dilation - 1, dilation + 1);
	options2D.setChannelPosition(xvigra::ChannelPosition::LAST);   

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 1> kernelY;
	xt::xtensor<ElementType, 1> kernelX;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernelY = xt::random::rand<ElementType>(kernelShapeY);
		kernelX = xt::random::rand<ElementType>(kernelShapeX);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernelY = xt::random::randint<ElementType>(kernelShapeY);
		kernelX = xt::random::randint<ElementType>(kernelShapeX);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve2D(
		 	input, 
		 	std::array{kernelY, kernelX}, 
		 	options2D
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolve2D - end                                                                              ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolveND<2> - begin                                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_separableConvolveND_2D_kernelSize_channelFirst(benchmark::State& state) {
	int inputHeight = 1080;
	int inputWidth = 1920;
	int inputChannels = 3;
	int kernelHeight = state.range(0);
	int kernelWidth = state.range(0);
	
	std::array<int, 3> inputShape{inputChannels, inputHeight, inputWidth};
	std::array<int, 1> kernelShapeY{kernelHeight};
	std::array<int, 1> kernelShapeX{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions2D options2D;
	options2D.setPadding(padding - 1, padding + 1);
	options2D.setStride(stride + 1, stride - 1);
	options2D.setDilation(dilation - 1, dilation + 1);
	options2D.setChannelPosition(xvigra::ChannelPosition::FIRST);   

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 1> kernelY;
	xt::xtensor<ElementType, 1> kernelX;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernelY = xt::random::rand<ElementType>(kernelShapeY);
		kernelX = xt::random::rand<ElementType>(kernelShapeX);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernelY = xt::random::randint<ElementType>(kernelShapeY);
		kernelX = xt::random::randint<ElementType>(kernelShapeX);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolveND<2>(
		 	input, 
		 	std::array{kernelY, kernelX}, 
		 	std::array{options2D.optionsY, options2D.optionsX}
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_separableConvolveND_2D_kernelSize_channelLast(benchmark::State& state) {
	int inputHeight = 1080;
	int inputWidth = 1920;
	int inputChannels = 3;
	int kernelHeight = state.range(0);
	int kernelWidth = state.range(0);
	
	std::array<int, 3> inputShape{inputHeight, inputWidth, inputChannels};
	std::array<int, 1> kernelShapeY{kernelHeight};
	std::array<int, 1> kernelShapeX{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions2D options2D;
	options2D.setPadding(padding - 1, padding + 1);
	options2D.setStride(stride + 1, stride - 1);
	options2D.setDilation(dilation - 1, dilation + 1);
	options2D.setChannelPosition(xvigra::ChannelPosition::LAST);   

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 1> kernelY;
	xt::xtensor<ElementType, 1> kernelX;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernelY = xt::random::rand<ElementType>(kernelShapeY);
		kernelX = xt::random::rand<ElementType>(kernelShapeX);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernelY = xt::random::randint<ElementType>(kernelShapeY);
		kernelX = xt::random::randint<ElementType>(kernelShapeX);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolveND<2>(
		 	input, 
		 	std::array{kernelY, kernelX}, 
		 	std::array{options2D.optionsY, options2D.optionsX}
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolveND<2> - end                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolve<2> - begin                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_separableConvolve_2D_kernelSize_channelFirst(benchmark::State& state) {
	int inputHeight = 1080;
	int inputWidth = 1920;
	int inputChannels = 3;
	int kernelHeight = state.range(0);
	int kernelWidth = state.range(0);
	
	std::array<int, 3> inputShape{inputChannels, inputHeight, inputWidth};
	std::array<int, 1> kernelShapeY{kernelHeight};
	std::array<int, 1> kernelShapeX{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions2D options2D;
	options2D.setPadding(padding - 1, padding + 1);
	options2D.setStride(stride + 1, stride - 1);
	options2D.setDilation(dilation - 1, dilation + 1);
	options2D.setChannelPosition(xvigra::ChannelPosition::FIRST);   

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 1> kernelY;
	xt::xtensor<ElementType, 1> kernelX;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernelY = xt::random::rand<ElementType>(kernelShapeY);
		kernelX = xt::random::rand<ElementType>(kernelShapeX);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernelY = xt::random::randint<ElementType>(kernelShapeY);
		kernelX = xt::random::randint<ElementType>(kernelShapeX);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve<2>(
		 	input, 
		 	std::array{kernelY, kernelX}, 
		 	std::array{options2D.optionsY, options2D.optionsX}
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_separableConvolve_2D_kernelSize_channelLast(benchmark::State& state) {
	int inputHeight = 1080;
	int inputWidth = 1920;
	int inputChannels = 3;
	int kernelHeight = state.range(0);
	int kernelWidth = state.range(0);
	
	std::array<int, 3> inputShape{inputHeight, inputWidth, inputChannels};
	std::array<int, 1> kernelShapeY{kernelHeight};
	std::array<int, 1> kernelShapeX{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions2D options2D;
	options2D.setPadding(padding - 1, padding + 1);
	options2D.setStride(stride + 1, stride - 1);
	options2D.setDilation(dilation - 1, dilation + 1);
	options2D.setChannelPosition(xvigra::ChannelPosition::LAST);   

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 1> kernelY;
	xt::xtensor<ElementType, 1> kernelX;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernelY = xt::random::rand<ElementType>(kernelShapeY);
		kernelX = xt::random::rand<ElementType>(kernelShapeX);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernelY = xt::random::randint<ElementType>(kernelShapeY);
		kernelX = xt::random::randint<ElementType>(kernelShapeX);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve<2>(
		 	input, 
		 	std::array{kernelY, kernelX}, 
		 	std::array{options2D.optionsY, options2D.optionsX}
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolve<2> - end                                                                             ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark convolve2D - begin                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_convolve2D_kernelSize_channelFirst(benchmark::State& state) {
	int inputHeight = 1080;
	int inputWidth = 1920;
	int inputChannels = 3;
	int kernelHeight = state.range(0);
	int kernelWidth = state.range(0);
	
	std::array<int, 3> inputShape{inputChannels, inputHeight, inputWidth};
	std::array<int, 4> kernelShape{inputChannels, inputChannels, kernelHeight, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions2D options2D;
	options2D.setPadding(padding - 1, padding + 1);
	options2D.setStride(stride + 1, stride - 1);
	options2D.setDilation(dilation - 1, dilation + 1);
	options2D.setChannelPosition(xvigra::ChannelPosition::FIRST);   

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 4> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::convolve2D(
		 	input, 
		 	kernel, 
		 	options2D
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_convolve2D_kernelSize_channelLast(benchmark::State& state) {
	int inputHeight = 1080;
	int inputWidth = 1920;
	int inputChannels = 3;
	int kernelHeight = state.range(0);
	int kernelWidth = state.range(0);
	
	std::array<int, 3> inputShape{inputHeight, inputWidth, inputChannels};
	std::array<int, 4> kernelShape{inputChannels, inputChannels, kernelHeight, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions2D options2D;
	options2D.setPadding(padding - 1, padding + 1);
	options2D.setStride(stride + 1, stride - 1);
	options2D.setDilation(dilation - 1, dilation + 1);
	options2D.setChannelPosition(xvigra::ChannelPosition::LAST);   

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 4> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::convolve2D(
		 	input, 
		 	kernel, 
		 	options2D
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark convolve2D - end                                                                                       ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ run benchmarks - begin                                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

BENCHMARK_SINGLE_VERSION(benchmark_separableConvolve2D_kernelSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_separableConvolve2D_kernelSize_channelLast);

BENCHMARK_SINGLE_VERSION(benchmark_separableConvolveND_2D_kernelSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_separableConvolveND_2D_kernelSize_channelLast);

BENCHMARK_SINGLE_VERSION(benchmark_separableConvolve_2D_kernelSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_separableConvolve_2D_kernelSize_channelLast);

BENCHMARK_SINGLE_VERSION(benchmark_convolve2D_kernelSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_convolve2D_kernelSize_channelLast);


BENCHMARK_MAIN();

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ run benchmarks - end                                                                                             ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝