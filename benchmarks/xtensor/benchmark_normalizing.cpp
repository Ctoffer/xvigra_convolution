#include <benchmark/benchmark.h>

#include "raw/array_view_3d.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xrandom.hpp"

#define INPUT_SIZE_MIN 1
#define INPUT_SIZE_MAX 50
#define INPUT_SIZE_STEP 3

#define BENCHMARK_SINGLE_VERSION(name)                                        \
    BENCHMARK_TEMPLATE(name, float)                                           \
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {   \
        return *(std::min_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {   \
        return *(std::max_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)             \
    ->Unit(benchmark::kNanosecond)

template <typename T>
void benchmark_normalizing_amax(benchmark::State& state) {
    using InputTensor = typename xt::xtensor<T, 3>;
    using ShapeType = typename InputTensor::shape_type;
    using ShapeValueType = typename ShapeType::value_type;

    ShapeValueType size = static_cast<ShapeValueType>(state.range(0));

    InputTensor input;
    InputTensor result;
    ShapeType inputShape{size - 1, size, size + 1};

    if constexpr (std::is_floating_point_v<T>) {
        input = xt::random::rand<T>(inputShape);
    } else {
        input = xt::random::randint<T>(inputShape);
    }

    for (auto _ : state) {
        result = input / xt::amax(input);
    }
}

template <typename T>
void benchmark_normalizing_amax_unpacked(benchmark::State& state) {
    using InputTensor = typename xt::xtensor<T, 3>;
    using ShapeType = typename InputTensor::shape_type;
    using ShapeValueType = typename ShapeType::value_type;

    ShapeValueType size = static_cast<ShapeValueType>(state.range(0));

    InputTensor input;
    InputTensor result;
    ShapeType inputShape{size - 1, size, size + 1};

    if constexpr (std::is_floating_point_v<T>) {
        input = xt::random::rand<T>(inputShape);
    } else {
        input = xt::random::randint<T>(inputShape);
    }

    for (auto _ : state) {
        result = input / xt::amax(input)[0];
    }
}

BENCHMARK_SINGLE_VERSION(benchmark_normalizing_amax);
BENCHMARK_SINGLE_VERSION(benchmark_normalizing_amax_unpacked);

BENCHMARK_MAIN();
