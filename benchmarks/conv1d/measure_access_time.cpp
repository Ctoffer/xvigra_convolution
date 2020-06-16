#include <benchmark/benchmark.h>

#include <iostream>
#include "conv1d_v2.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#define INPUT_SIZE_MIN 50
#define INPUT_SIZE_MAX 2000
#define INPUT_SIZE_STEP 50

template <typename ElementType>
void benchmark_conv1d_accessTime_channelOuterDimension(benchmark::State& state) {
	int dynamic = static_cast<int>(state.range(0));

	int inputWidth = dynamic;
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputChannels, inputWidth};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 3> kernels;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernels = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernels = xt::random::randint<ElementType>(kernelShape);
	}

	Conv1D convolution(3, 2, 4);
	auto result = convolution(input, kernels, true);
	
	for (auto _ : state) {
		 xt::xtensor<ElementType, 2> copiedResult(result.shape());
		 std::copy(result.begin(), result.end(), copiedResult.begin());
		 benchmark::DoNotOptimize(copiedResult.data());
	}
	
}

template <typename ElementType>
void benchmark_conv1d_accessTime_channelInnerDimension(benchmark::State& state) {
	int dynamic = static_cast<int>(state.range(0));

	int inputWidth = dynamic;
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

	xt::xtensor<ElementType, 2> input;
	xt::xtensor<ElementType, 3> kernels;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernels = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernels = xt::random::randint<ElementType>(kernelShape);
	}
	
	Conv1D convolution(3, 2, 4);
	auto result = convolution(input, kernels, false);
	
	for (auto _ : state) {
		 xt::xtensor<ElementType, 2> copiedResult(result.shape());
		 std::copy(result.begin(), result.end(), copiedResult.begin());
		 benchmark::DoNotOptimize(copiedResult.data());
	}
	
}

BENCHMARK_TEMPLATE(benchmark_conv1d_accessTime_channelOuterDimension, std::uint16_t)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_accessTime_channelOuterDimension, int)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_accessTime_channelOuterDimension, float)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_accessTime_channelOuterDimension, double)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

BENCHMARK_TEMPLATE(benchmark_conv1d_accessTime_channelInnerDimension, std::uint16_t)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_accessTime_channelInnerDimension, int)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_accessTime_channelInnerDimension, float)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_accessTime_channelInnerDimension, double)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

BENCHMARK_MAIN();