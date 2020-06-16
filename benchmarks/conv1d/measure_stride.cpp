#include <benchmark/benchmark.h>

#include <iostream>
#include "conv1d_v2.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#define STRIDE_MIN 1
#define STRIDE_MAX 8
#define STRIDE_STEP 1

template <typename ElementType>
void benchmark_conv1d_stride_channelOuterDimension(benchmark::State& state) {
	int inputWidth = 500;
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

	Conv1D convolution(3, 2, static_cast<int>(state.range(0)));
	
	for (auto _ : state) {
		 auto result = convolution(input, kernels);
		 benchmark::DoNotOptimize(result.data());
	}
	
}

template <typename ElementType>
void benchmark_conv1d_stride_channelInnerDimension(benchmark::State& state) {
	int inputWidth = 500;
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
	
	Conv1D convolution(3, 2, static_cast<int>(state.range(0)));
	
	for (auto _ : state) {
		 auto result = convolution(input, kernels, false);
		 benchmark::DoNotOptimize(result.data());
	}
	
}

BENCHMARK_TEMPLATE(benchmark_conv1d_stride_channelOuterDimension, std::uint16_t)->DenseRange(STRIDE_MIN, STRIDE_MAX, STRIDE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_stride_channelOuterDimension, int)->DenseRange(STRIDE_MIN, STRIDE_MAX, STRIDE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_stride_channelOuterDimension, float)->DenseRange(STRIDE_MIN, STRIDE_MAX, STRIDE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_stride_channelOuterDimension, double)->DenseRange(STRIDE_MIN, STRIDE_MAX, STRIDE_STEP);

BENCHMARK_TEMPLATE(benchmark_conv1d_stride_channelInnerDimension, std::uint16_t)->DenseRange(STRIDE_MIN, STRIDE_MAX, STRIDE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_stride_channelInnerDimension, int)->DenseRange(STRIDE_MIN, STRIDE_MAX, STRIDE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_stride_channelInnerDimension, float)->DenseRange(STRIDE_MIN, STRIDE_MAX, STRIDE_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_stride_channelInnerDimension, double)->DenseRange(STRIDE_MIN, STRIDE_MAX, STRIDE_STEP);

BENCHMARK_MAIN();