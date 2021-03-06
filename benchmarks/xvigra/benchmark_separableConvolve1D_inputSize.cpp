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
    ->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)             \
    ->Unit(benchmark::kMicrosecond)

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ define - end                                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolve1D - begin                                                                            ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_separableConvolve1D_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 1> kernelShape{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

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
		 auto result = xvigra::separableConvolve1D(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_separableConvolve1D_inputSize_channelLast(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
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
		 auto result = xvigra::separableConvolve1D(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolve1D - end                                                                              ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolveND<1> - begin                                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_separableConvolveND_1D_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 1> kernelShape{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 1> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}

	auto kernelArray = std::array{kernel};
	auto optionsArray = std::array{options};
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolveND<1>(input, kernelArray, optionsArray);
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_separableConvolveND_1D_inputSize_channelLast(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
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

	auto kernelArray = std::array{kernel};
	auto optionsArray = std::array{options};
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolveND<1>(input, kernelArray, optionsArray);
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolveND<1> - end                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolve<1> - begin                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_separableConvolve_1D_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 1> kernelShape{kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::FIRST;   

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 1> kernel;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernel = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernel = xt::random::randint<ElementType>(kernelShape);
	}

	auto kernelArray = std::array{kernel};
	auto optionsArray = std::array{options};
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve<1>(input, kernelArray, optionsArray);
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_separableConvolve_1D_inputSize_channelLast(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
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

	auto kernelArray = std::array{kernel};
	auto optionsArray = std::array{options};
	
	for (auto _ : state) {
		 auto result = xvigra::separableConvolve<1>(input, kernelArray, optionsArray);
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark separableConvolve<1> - end                                                                             ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark convolve1D - begin                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename ElementType>
void benchmark_convolve1D_inputSize_channelFirst(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 3> kernelShape{inputChannels, inputChannels, kernelWidth};

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
		 auto result = xvigra::convolve1D(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}


template <typename ElementType>
void benchmark_convolve1D_inputSize_channelLast(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 3> kernelShape{inputChannels, inputChannels, kernelWidth};

	int padding = 3;
	int stride = 4;
	int dilation = 2;

	xvigra::KernelOptions options(padding, stride, dilation);
	options.channelPosition = xvigra::ChannelPosition::LAST;   

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
		 auto result = xvigra::convolve1D(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ benchmark convolve1D - end                                                                                       ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝


// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ run benchmarks - begin                                                                                           ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

BENCHMARK_SINGLE_VERSION(benchmark_separableConvolve1D_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_separableConvolve1D_inputSize_channelLast);

BENCHMARK_SINGLE_VERSION(benchmark_separableConvolveND_1D_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_separableConvolveND_1D_inputSize_channelLast);

BENCHMARK_SINGLE_VERSION(benchmark_separableConvolve_1D_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_separableConvolve_1D_inputSize_channelLast);

BENCHMARK_SINGLE_VERSION(benchmark_convolve1D_inputSize_channelFirst);
BENCHMARK_SINGLE_VERSION(benchmark_convolve1D_inputSize_channelLast);


BENCHMARK_MAIN();

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ run benchmarks - end                                                                                             ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
