#include <benchmark/benchmark.h>

#include <cstdint>
#include <iostream>

#include "conv1d_v2.hpp"

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#define CHANNELS_MIN 2
#define CHANNELS_MAX 10
#define CHANNELS_STEP 1

template <typename ElementType>
void benchmark_conv1d_channels_channelOuterDimension(benchmark::State& state) {
	int inputWidth = 500;
	int inputChannels = static_cast<int>(state.range(0));
	int outputChannels = inputChannels - 1;
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
	
	for (auto _ : state) {
		 auto result = convolution(input, kernels, true);
		 benchmark::DoNotOptimize(result.data());
	}
	
}

template <typename ElementType>
void benchmark_conv1d_channels_channelInnerDimension(benchmark::State& state) {
	int inputWidth = 500;
	int inputChannels = static_cast<int>(state.range(0));
	int outputChannels = inputChannels - 1;
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
	
	for (auto _ : state) {
		 auto result = convolution(input, kernels, false);
		 benchmark::DoNotOptimize(result.data());
	}
	
}

BENCHMARK_TEMPLATE(benchmark_conv1d_channels_channelOuterDimension, std::uint16_t)->DenseRange(CHANNELS_MIN, CHANNELS_MAX, CHANNELS_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_channels_channelOuterDimension, int)->DenseRange(CHANNELS_MIN, CHANNELS_MAX, CHANNELS_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_channels_channelOuterDimension, float)->DenseRange(CHANNELS_MIN, CHANNELS_MAX, CHANNELS_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_channels_channelOuterDimension, double)->DenseRange(CHANNELS_MIN, CHANNELS_MAX, CHANNELS_STEP);

BENCHMARK_TEMPLATE(benchmark_conv1d_channels_channelInnerDimension, std::uint16_t)->DenseRange(CHANNELS_MIN, CHANNELS_MAX, CHANNELS_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_channels_channelInnerDimension, int)->DenseRange(CHANNELS_MIN, CHANNELS_MAX, CHANNELS_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_channels_channelInnerDimension, float)->DenseRange(CHANNELS_MIN, CHANNELS_MAX, CHANNELS_STEP);
BENCHMARK_TEMPLATE(benchmark_conv1d_channels_channelInnerDimension, double)->DenseRange(CHANNELS_MIN, CHANNELS_MAX, CHANNELS_STEP);

BENCHMARK_MAIN();