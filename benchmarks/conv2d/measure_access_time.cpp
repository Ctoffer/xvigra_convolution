#include <benchmark/benchmark.h>

#include <iostream>
#include "conv2d_v2.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#define INPUT_SIZE_MIN 50
#define INPUT_SIZE_MAX 2000
#define INPUT_SIZE_STEP 50

template <typename ElementType>
void benchmark_conv2d_accessTime_channelOuterDimension(benchmark::State& state) {
	int dynamic = static_cast<int>(state.range(0));

	int inputHeight = dynamic + 1;
	int inputWidth = dynamic;
	int inputChannels = 3;
	int outputChannels = inputChannels - 1;
	int kernelHeight = 6;
	int kernelWidth = 7;

	std::tuple<int, int> padding(4, 3);
	std::tuple<int, int> dilation(3, 2);
	std::tuple<int, int> stride(3, 4);
	
	std::array<int, 3> inputShape{inputChannels, inputHeight, inputWidth};
	std::array<int, 4> kernelShape{outputChannels, inputChannels, kernelHeight, kernelWidth};

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 4> kernels;

	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernels = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernels = xt::random::randint<ElementType>(kernelShape);
	}
	
	Conv2D convolution(padding, dilation, stride);
	auto result = convolution(input, kernels);

	for (auto _ : state) {
		 xt::xtensor<ElementType, 3> copiedResult(result.shape());
		 std::copy(result.begin(), result.end(), copiedResult.begin());
		 benchmark::DoNotOptimize(copiedResult.data());
	}
	
}

template <typename ElementType>
void benchmark_conv2d_accessTime_channelInnerDimension(benchmark::State& state) {
	int dynamic = static_cast<int>(state.range(0));

	int inputHeight = dynamic + 1;
	int inputWidth = dynamic;
	int inputChannels = 3;
	int outputChannels = inputChannels - 1;
	int kernelHeight = 6;
	int kernelWidth = 7;

	std::tuple<int, int> padding(4, 3);
	std::tuple<int, int> dilation(3, 2);
	std::tuple<int, int> stride(3, 4);
	
	std::array<int, 3> inputShape{inputHeight, inputWidth, inputChannels};
	std::array<int, 4> kernelShape{outputChannels, inputChannels, kernelHeight, kernelWidth};

	xt::xtensor<ElementType, 3> input;
	xt::xtensor<ElementType, 4> kernels;
	
	if constexpr (std::is_floating_point<ElementType>::value) {
		input = xt::random::rand<ElementType>(inputShape);
		kernels = xt::random::rand<ElementType>(kernelShape);
	} else {
		input = xt::random::randint<ElementType>(inputShape);
		kernels = xt::random::randint<ElementType>(kernelShape);
	}
	
	Conv2D convolution(padding, dilation, stride);
	auto result = convolution(input, kernels, false);
	
	for (auto _ : state) {
		 xt::xtensor<ElementType, 3> copiedResult(result.shape());
		 std::copy(result.begin(), result.end(), copiedResult.begin());
		 benchmark::DoNotOptimize(copiedResult.data());
	}
	
}

BENCHMARK_TEMPLATE(benchmark_conv2d_accessTime_channelOuterDimension, std::uint16_t)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_accessTime_channelOuterDimension, int)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_accessTime_channelOuterDimension, float)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_accessTime_channelOuterDimension, double)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

BENCHMARK_TEMPLATE(benchmark_conv2d_accessTime_channelInnerDimension, std::uint16_t)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_accessTime_channelInnerDimension, int)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_accessTime_channelInnerDimension, float)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_accessTime_channelInnerDimension, double)->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

BENCHMARK_MAIN();