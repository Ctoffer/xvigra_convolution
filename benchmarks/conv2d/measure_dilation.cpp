#include <benchmark/benchmark.h>

#include <iostream>
#include "conv2d_v2.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#define DILATION_MIN 2
#define DILATION_MAX 20
#define DILATION_STEP 1

template <typename ElementType>
void benchmark_conv2d_dilation_channelOuterDimension(benchmark::State& state) {
	int dynamic = static_cast<int>(state.range(0));

	int inputHeight = 501;
	int inputWidth = 500;
	int inputChannels = 3;
	int outputChannels = inputChannels - 1;
	int kernelHeight = 6;
	int kernelWidth = 7;

	std::tuple<int, int> padding(4, 3);
	std::tuple<int, int> dilation(dynamic + 1, dynamic);
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

	for (auto _ : state) {
		 auto result = convolution(input, kernels);
		 benchmark::DoNotOptimize(result.data());
	}
	
}

template <typename ElementType>
void benchmark_conv2d_dilation_channelInnerDimension(benchmark::State& state) {
	int dynamic = static_cast<int>(state.range(0));

	int inputHeight = 501;
	int inputWidth = 500;
	int inputChannels = 3;
	int outputChannels = inputChannels - 1;
	int kernelHeight = 6;
	int kernelWidth = 7;
	
	std::tuple<int, int> padding(4, 3);
	std::tuple<int, int> dilation(dynamic + 1, dynamic);
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
	
	for (auto _ : state) {
		 auto result = convolution(input, kernels, false);
		 benchmark::DoNotOptimize(result.data());
	}
	
}

BENCHMARK_TEMPLATE(benchmark_conv2d_dilation_channelOuterDimension, std::uint16_t)->DenseRange(DILATION_MIN, DILATION_MAX, DILATION_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_dilation_channelOuterDimension, int)->DenseRange(DILATION_MIN, DILATION_MAX, DILATION_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_dilation_channelOuterDimension, float)->DenseRange(DILATION_MIN, DILATION_MAX, DILATION_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_dilation_channelOuterDimension, double)->DenseRange(DILATION_MIN, DILATION_MAX, DILATION_STEP);

BENCHMARK_TEMPLATE(benchmark_conv2d_dilation_channelInnerDimension, std::uint16_t)->DenseRange(DILATION_MIN, DILATION_MAX, DILATION_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_dilation_channelInnerDimension, int)->DenseRange(DILATION_MIN, DILATION_MAX, DILATION_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_dilation_channelInnerDimension, float)->DenseRange(DILATION_MIN, DILATION_MAX, DILATION_STEP);
BENCHMARK_TEMPLATE(benchmark_conv2d_dilation_channelInnerDimension, double)->DenseRange(DILATION_MIN, DILATION_MAX, DILATION_STEP);

BENCHMARK_MAIN();