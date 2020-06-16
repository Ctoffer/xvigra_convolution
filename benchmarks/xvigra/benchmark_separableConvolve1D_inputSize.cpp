#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#include "xvigra/separable_convolution.hpp"

#define INPUT_SIZE_MIN 50
#define INPUT_SIZE_MAX 2000
#define INPUT_SIZE_STEP 50

template <typename ElementType>
void benchmark_separableConvolve1D_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 1> kernelShape{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::IMPLICIT;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 1> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve1D<ElementType, ElementType, xvigra::ChannelPosition::FIRST>(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_separableConvolve1D_inputSize_channelLast(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 1> kernelShape{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::LAST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 1> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve1D<ElementType, ElementType, xvigra::ChannelPosition::LAST>(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}


// =======================================================================================
// =======================================================================================
// =======================================================================================

template <typename ElementType>
void benchmark_separableConvolve_1D_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 1;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 1> kernelShape{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::IMPLICIT;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 1> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve<ElementType, ElementType, xvigra::ChannelPosition::FIRST, 1>(input, {kernel}, {options});
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_separableConvolve_1D_inputSize_channelLast(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 1> kernelShape{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::IMPLICIT;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 1> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve<ElementType, ElementType, xvigra::ChannelPosition::LAST, 1>(input, {kernel}, {options});
		 benchmark::DoNotOptimize(result.data());
	}
}


// =======================================================================================
// =======================================================================================
// =======================================================================================

BENCHMARK_TEMPLATE(benchmark_separableConvolve1D_inputSize_channelFirst, std::uint16_t)
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_separableConvolve1D_inputSize_channelLast, std::uint16_t)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);


BENCHMARK_TEMPLATE(benchmark_separableConvolve_1D_inputSize_channelFirst, std::uint16_t)
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_separableConvolve_1D_inputSize_channelLast, std::uint16_t)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);


BENCHMARK_MAIN();