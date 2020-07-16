#include <benchmark/benchmark.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xmanipulation.hpp"
#include "xtensor/xrandom.hpp"

#define INPUT_SIZE_MIN 50
#define INPUT_SIZE_MAX 1100
#define INPUT_SIZE_STEP 100

#define BENCHMARK_SINGLE_VERSION_XARRAY(name)                                 \
    BENCHMARK_TEMPLATE(name, xt::xarray<float>)                               \
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {   \
        return *(std::min_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {   \
        return *(std::max_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)             \
    ->Unit(benchmark::kMicrosecond)

#define BENCHMARK_SINGLE_VERSION_XTENSOR(name)                                \
    BENCHMARK_TEMPLATE(name, xt::xtensor<float, 2>)                           \
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {   \
        return *(std::min_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {   \
        return *(std::max_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)             \
    ->Unit(benchmark::kMicrosecond)

template <typename ContainerType>
void benchmark_transpose_complete(benchmark::State& state) {
    using T = typename ContainerType::value_type;
    using ShapeType = typename ContainerType::shape_type;
    using ShapeValueType = typename ShapeType::value_type;

    ShapeValueType size = static_cast<ShapeValueType>(state.range(0));

    ShapeType inputShape{size - 25, size + 25};
    ShapeType resultShape{size + 25, size - 25};

    ContainerType input;

    if constexpr (std::is_floating_point_v<T>) {
        input = xt::random::rand<T>(inputShape);
    } else {
        input = xt::random::randint<T>(inputShape);
    }

    ContainerType result(resultShape);

    for (auto _ : state) {
        result = xt::transpose(input);
    }
}


template <typename ContainerType>
void benchmark_transpose_complete_operatorCallAgainstCache(benchmark::State& state) {
    using T = typename ContainerType::value_type;
    using ShapeType = typename ContainerType::shape_type;
    using ShapeValueType = typename ShapeType::value_type;

    ShapeValueType size = static_cast<ShapeValueType>(state.range(0));

    ShapeType inputShape{size - 25, size + 25};
    ShapeType resultShape{size + 25, size - 25};

    ContainerType input;

    if constexpr (std::is_floating_point_v<T>) {
        input = xt::random::rand<T>(inputShape);
    } else {
        input = xt::random::randint<T>(inputShape);
    }

    ContainerType result(resultShape);

    for (auto _ : state) {
        for(std::size_t y = 0; y < inputShape[0]; ++y) {
            for(std::size_t x = 0; x < inputShape[1]; ++x) {
                result(x, y) = input(y, x);
            }
        }
    }
}

template <typename ContainerType>
void benchmark_transpose_complete_operatorCallCacheAligned(benchmark::State& state) {
    using T = typename ContainerType::value_type;
    using ShapeType = typename ContainerType::shape_type;
    using ShapeValueType = typename ShapeType::value_type;

    ShapeValueType size = static_cast<ShapeValueType>(state.range(0));

    ShapeType inputShape{size - 25, size + 25};
    ShapeType resultShape{size + 25, size - 25};

    ContainerType input;

    if constexpr (std::is_floating_point_v<T>) {
        input = xt::random::rand<T>(inputShape);
    } else {
        input = xt::random::randint<T>(inputShape);
    }

    ContainerType result(resultShape);

    for (auto _ : state) {
        for(std::size_t x = 0; x < resultShape[0]; ++x) {
            for(std::size_t y = 0; y < resultShape[1]; ++y) {
                result(x, y) = input(y, x);
            }
        }
    }
}

BENCHMARK_SINGLE_VERSION_XARRAY(benchmark_transpose_complete);
BENCHMARK_SINGLE_VERSION_XARRAY(benchmark_transpose_complete_operatorCallAgainstCache);
BENCHMARK_SINGLE_VERSION_XARRAY(benchmark_transpose_complete_operatorCallCacheAligned);

BENCHMARK_SINGLE_VERSION_XTENSOR(benchmark_transpose_complete);
BENCHMARK_SINGLE_VERSION_XTENSOR(benchmark_transpose_complete_operatorCallAgainstCache);
BENCHMARK_SINGLE_VERSION_XTENSOR(benchmark_transpose_complete_operatorCallCacheAligned);

BENCHMARK_MAIN();
