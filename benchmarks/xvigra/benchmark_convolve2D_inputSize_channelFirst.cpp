#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#include "xvigra/explicit_convolution.hpp"

#define INPUT_SIZE_MIN 50
#define INPUT_SIZE_MAX 2000
#define INPUT_SIZE_STEP 650

// =======================================================================================


template <typename ElementType>
void benchmark_convolve2D_v1_inputSize_channelFirst(benchmark::State& state) {
	int inputHeight = static_cast<int>(state.range(0) + 1);
	int inputWidth = static_cast<int>(state.range(0) - 1);
	int inputChannels = 3;
	int kernelHeight = 8;
	int kernelWidth = 7;
	
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
void benchmark_convolve2D_v1_inputSize_channelLast(benchmark::State& state) {
	int inputHeight = static_cast<int>(state.range(0) + 1);
	int inputWidth = static_cast<int>(state.range(0) - 1);
	int inputChannels = 3;
	int kernelHeight = 8;
	int kernelWidth = 7;
	
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


// =======================================================================================

template <typename ElementType>
void benchmark_convolve2D_v2_inputSize_channelFirst(benchmark::State& state) {
	int inputHeight = static_cast<int>(state.range(0) + 1);
	int inputWidth = static_cast<int>(state.range(0) - 1);
	int inputChannels = 3;
	int kernelHeight = 8;
	int kernelWidth = 7;
	
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
		 auto result = xvigra::convolve2D_v2(
		 	input, 
		 	kernel, 
		 	options2D
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}

// =======================================================================================

template <typename ElementType>
void benchmark_convolve2D_v3_inputSize_channelFirst(benchmark::State& state) {
	int inputHeight = static_cast<int>(state.range(0) + 1);
	int inputWidth = static_cast<int>(state.range(0) - 1);
	int inputChannels = 3;
	int kernelHeight = 8;
	int kernelWidth = 7;
	
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
		 auto result = xvigra::convolve2D_v3(
		 	input, 
		 	kernel, 
		 	options2D
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}

// =======================================================================================

template <typename ElementType>
void benchmark_convolve2D_v4_inputSize_channelFirst(benchmark::State& state) {
	int inputHeight = static_cast<int>(state.range(0) + 1);
	int inputWidth = static_cast<int>(state.range(0) - 1);
	int inputChannels = 3;
	int kernelHeight = 8;
	int kernelWidth = 7;
	
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
		 auto result = xvigra::convolve2D_v4(
		 	input, 
		 	kernel, 
		 	options2D
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}

// =======================================================================================



// =======================================================================================


BENCHMARK_TEMPLATE(benchmark_convolve2D_v1_inputSize_channelFirst, float)
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)
	->Unit(benchmark::kMillisecond);


// =======================================================================================


BENCHMARK_TEMPLATE(benchmark_convolve2D_v2_inputSize_channelFirst, float)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)
	->Unit(benchmark::kMillisecond);


// =======================================================================================


BENCHMARK_TEMPLATE(benchmark_convolve2D_v3_inputSize_channelFirst, float)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)
	->Unit(benchmark::kMillisecond);


// =======================================================================================


BENCHMARK_TEMPLATE(benchmark_convolve2D_v4_inputSize_channelFirst, float)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)
	->Unit(benchmark::kMillisecond);



// =======================================================================================


BENCHMARK_MAIN();