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
    BENCHMARK_TEMPLATE(name, xt::xarray<float>, xt::xarray<float>)                               \
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {   \
        return *(std::min_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {   \
        return *(std::max_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)             \
    ->Unit(benchmark::kMillisecond)

#define BENCHMARK_SINGLE_VERSION_XTENSOR(name)                                \
    BENCHMARK_TEMPLATE(name, xt::xtensor<float, 2>, xt::xtensor<float, 4>)                           \
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {   \
        return *(std::min_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {   \
        return *(std::max_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)             \
    ->Unit(benchmark::kMillisecond)

template <typename InputContainerType, typename ResultContainerType>
void benchmark_reshape_complete(benchmark::State& state) {
    using T = typename InputContainerType::value_type;
    using InputShapeType = typename InputContainerType::shape_type;
    using ResultShapeType = typename ResultContainerType::shape_type;
    using ShapeValueType = typename InputShapeType::value_type;

    ShapeValueType size = static_cast<ShapeValueType>(state.range(0));

    InputShapeType inputShape{size - 25, size + 25};
    ResultShapeType resultShape{5, (size - 25)/5, (size + 25)/5, 5};

    InputContainerType input;

    if constexpr (std::is_floating_point_v<T>) {
        input = xt::random::rand<T>(inputShape);
    } else {
        input = xt::random::randint<T>(inputShape);
    }

    ResultContainerType result(resultShape);

    for (auto _ : state) {
        result = xt::reshape_view(input, resultShape);
    }
}


template <typename InputContainerType, typename ResultContainerType>
void benchmark_reshape_complete_operatorCallAgainstCache(benchmark::State& state) {
    using T = typename InputContainerType::value_type;
    using InputShapeType = typename InputContainerType::shape_type;
    using ResultShapeType = typename ResultContainerType::shape_type;
    using ShapeValueType = typename InputShapeType::value_type;

    ShapeValueType size = static_cast<ShapeValueType>(state.range(0));

    InputShapeType inputShape{size - 25, size + 25};
    ResultShapeType resultShape{5, (size - 25)/5, (size + 25)/5, 5};

    InputContainerType input;

    if constexpr (std::is_floating_point_v<T>) {
        input = xt::random::rand<T>(inputShape);
    } else {
        input = xt::random::randint<T>(inputShape);
    }

    ResultContainerType result(resultShape);

    for (auto _ : state) {
        for (std::size_t w = 0; w < resultShape[3]; ++w) {
            for (std::size_t x = 0; x < resultShape[2]; ++x) {
                for (std::size_t y = 0; y < resultShape[1]; ++y) {
                    for (std::size_t z = 0; z < resultShape[0]; ++z) {
                        result(z, y, x, w) = input(z * resultShape[1] + y, x * resultShape[3] + w);
                    }
                }
            }
        }
    }
}

template <typename InputContainerType, typename ResultContainerType>
void benchmark_reshape_complete_operatorCallCacheAligned(benchmark::State& state) {
    using T = typename InputContainerType::value_type;
    using InputShapeType = typename InputContainerType::shape_type;
    using ResultShapeType = typename ResultContainerType::shape_type;
    using ShapeValueType = typename InputShapeType::value_type;

    ShapeValueType size = static_cast<ShapeValueType>(state.range(0));

    InputShapeType inputShape{size - 25, size + 25};
    ResultShapeType resultShape{5, (size - 25)/5, (size + 25)/5, 5};

    InputContainerType input;

    if constexpr (std::is_floating_point_v<T>) {
        input = xt::random::rand<T>(inputShape);
    } else {
        input = xt::random::randint<T>(inputShape);
    }

    ResultContainerType result(resultShape);

    for (auto _ : state) {
        for (std::size_t z = 0; z < resultShape[0]; ++z) {
            for (std::size_t y = 0; y < resultShape[1]; ++y) {
                for (std::size_t x = 0; x < resultShape[2]; ++x) {
                    for (std::size_t w = 0; w < resultShape[3]; ++w) {
                        result(z, y, x, w) = input(z * resultShape[1] + y, x * resultShape[3] + w);
                    }
                }
            }
        }
    }
}

BENCHMARK_SINGLE_VERSION_XARRAY(benchmark_reshape_complete);
BENCHMARK_SINGLE_VERSION_XARRAY(benchmark_reshape_complete_operatorCallAgainstCache);
BENCHMARK_SINGLE_VERSION_XARRAY(benchmark_reshape_complete_operatorCallCacheAligned);

BENCHMARK_SINGLE_VERSION_XTENSOR(benchmark_reshape_complete);
BENCHMARK_SINGLE_VERSION_XTENSOR(benchmark_reshape_complete_operatorCallAgainstCache);
BENCHMARK_SINGLE_VERSION_XTENSOR(benchmark_reshape_complete_operatorCallCacheAligned);

BENCHMARK_MAIN();
