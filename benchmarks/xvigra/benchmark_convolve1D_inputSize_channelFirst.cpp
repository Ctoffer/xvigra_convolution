#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#include "xvigra_legacy/explicit_convolution.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - begin                                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

#define INPUT_SIZE_MIN 50
#define INPUT_SIZE_MAX 2000
#define INPUT_SIZE_STEP 50


#define BENCHMARK_SINGLE_VERSION(name)                                        \
    BENCHMARK_TEMPLATE(name, float)                                           \
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {   \
        return *(std::min_element(std::begin(v), std::end(v)));               \
      })                                                                      \
    ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {   \
        return *(std::max_element(std::begin(v), std::end(v)));               \
      })                                                                      \
    ->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP) \
    ->Unit(benchmark::kMicrosecond)

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - end                                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark version 1 to 5 - begin                                                                                 ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_convolve1D_v1_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 3> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra_legacy::convolve1D_v1(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_convolve1D_v2_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 3> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}

	for (auto _ : state) {
		 auto result = xvigra_legacy::convolve1D_v2(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_convolve1D_v3_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 3> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	xvigra::ChannelPosition channelPosition = options.channelPosition;
    int paddingBegin = options.paddingBegin();
    int paddingEnd = options.paddingEnd();
    int paddingTotal = options.paddingTotal();
    xvigra::BorderTreatment borderTreatmentBegin = options.borderTreatmentBegin;
    xvigra::BorderTreatment borderTreatmentEnd = options.borderTreatmentEnd;

	for (auto _ : state) {
		 auto result = xvigra_legacy::convolve1D_v3(
		 	input, kernel, 
		 	channelPosition, 
		 	paddingBegin, paddingEnd, paddingTotal, 
		 	dilation, 
		 	stride, 
		 	borderTreatmentBegin, borderTreatmentEnd
		 );
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_convolve1D_v4_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 3> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra_legacy::convolve1D_v4(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_convolve1D_v5_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 3> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra_legacy::convolve1D_v5(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}

template <typename ElementType>
void benchmark_convolve1D_v6_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 3> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}
	
	for (auto _ : state) {
		 auto result = xvigra_legacy::convolve1D_v6(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark version 1 to 5 - end                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ run benchmarks - begin                                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

BENCHMARK_SINGLE_VERSION(benchmark_convolve1D_v1_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_convolve1D_v2_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_convolve1D_v3_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_convolve1D_v4_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_convolve1D_v5_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_convolve1D_v6_inputSize_channelFirst);


BENCHMARK_MAIN();

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ run benchmarks - end                                                                                             ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
