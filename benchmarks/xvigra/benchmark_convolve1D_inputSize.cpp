#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>

#include "xtensor/xtensor.hpp"
#include "xtensor/xrandom.hpp"

#include "xvigra_legacy/explicit_convolution.hpp"

#define INPUT_SIZE_MIN 50
#define INPUT_SIZE_MAX 2000
#define INPUT_SIZE_STEP 150
#define INPUT_TYPE float

// =======================================================================================

template <typename ElementType>
void benchmark_convolve1D_inputSize_channelFirst_v1(benchmark::State& state) {
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
void benchmark_convolve1D_inputSize_channelLast_v1(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

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
		 auto result = xvigra_legacy::convolve1D_v1(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}

// =======================================================================================

template <typename ElementType>
void benchmark_convolve1D_inputSize_channelFirst_v2(benchmark::State& state) {
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
void benchmark_convolve1D_inputSize_channelLast_v2(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

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
		 auto result = xvigra_legacy::convolve1D_v2(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}

// =======================================================================================

template <typename ElementType>
void benchmark_convolve1D_inputSize_channelFirst_v3(benchmark::State& state) {
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
void benchmark_convolve1D_inputSize_channelLast_v3(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

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

// =======================================================================================

template <typename ElementType>
void benchmark_convolve1D_inputSize_channelFirst_v4(benchmark::State& state) {
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
void benchmark_convolve1D_inputSize_channelLast_v4(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

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
		 auto result = xvigra_legacy::convolve1D_v4(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}

// =======================================================================================

template <typename ElementType>
void benchmark_convolve1D_inputSize_channelFirst_v5(benchmark::State& state) {
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
void benchmark_convolve1D_inputSize_channelLast_v5(benchmark::State& state) {
	int inputWidth = static_cast<int>(state.range(0));
	int inputChannels = 3;
	int outputChannels = 3;
	int kernelWidth = 7;
	
	std::array<int, 2> inputShape{inputWidth, inputChannels};
	std::array<int, 3> kernelShape{outputChannels, inputChannels, kernelWidth};

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
		 auto result = xvigra_legacy::convolve1D_v5(input, kernel, options);
		 benchmark::DoNotOptimize(result.data());
	}
}

// =======================================================================================

BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelFirst_v1, INPUT_TYPE)
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelLast_v1, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

// =======================================================================================

BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelFirst_v2, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelLast_v2, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

// =======================================================================================

BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelFirst_v3, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelLast_v3, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

// =======================================================================================

BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelFirst_v4, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelLast_v4, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

// =======================================================================================


BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelFirst_v5, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);
BENCHMARK_TEMPLATE(benchmark_convolve1D_inputSize_channelLast_v5, INPUT_TYPE)
	->ComputeStatistics("min", [](const std::vector<double>& v) -> double {
    	return *(std::min_element(std::begin(v), std::end(v)));
  	})
	->ComputeStatistics("max", [](const std::vector<double>& v) -> double {
    	return *(std::max_element(std::begin(v), std::end(v)));
  	})
	->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP);

// =======================================================================================

BENCHMARK_MAIN();