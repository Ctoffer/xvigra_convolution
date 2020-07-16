#include <benchmark/benchmark.h>

#include "raw/array_view_3d.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#define INPUT_SIZE_MIN 50
#define INPUT_SIZE_MAX 1100
#define INPUT_SIZE_STEP 100

#define SETUP_VARIABLES                                                                                                   \
    int size = state.range(0);                                                                                            \
                                                                                                                          \
    int inputZ = size - 1;                                                                                                \
    int inputY = size;                                                                                                    \
    int inputX = size + 1;                                                                                                \
                                                                                                                          \
    int paddingStartY = 0;                                                                                                \
    int paddingEndY = 0;                                                                                                  \
    int paddingStartX = 0;                                                                                                \
    int paddingEndX = 0;                                                                                                  \
                                                                                                                          \
    int strideY = 1;                                                                                                      \
    int strideX = 1;                                                                                                      \
                                                                                                                          \
    int outputY = static_cast<int>(std::ceil(static_cast<double>((inputY - paddingStartY - paddingEndY)) / (strideY)));   \
    int outputX = static_cast<int>(std::ceil(static_cast<double>((inputX - paddingStartX - paddingEndX)) / (strideX)));

#define BENCHMARK_SINGLE_VERSION(name)                                        \
    BENCHMARK_TEMPLATE(name, float)                                           \
    ->ComputeStatistics("min", [](const std::vector<double>& v) -> double {   \
        return *(std::min_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->ComputeStatistics("max", [](const std::vector<double>& v) -> double {   \
        return *(std::max_element(std::begin(v), std::end(v)));               \
    })                                                                        \
    ->DenseRange(INPUT_SIZE_MIN, INPUT_SIZE_MAX, INPUT_SIZE_STEP)             \
    ->Unit(benchmark::kMillisecond)

template <typename ElementType>
void benchmark_tensor_copy_complete_Z_againstCache(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});

    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int z = inputZ / 2;

    for (auto _ : state) {
        for(int x = 0; x < outputX; ++x) {
            for(int y = 0; y < outputY; ++y) {
                res(y, x) = data(z, y, x);
            }
        }
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_tensor_copy_complete_Z_cacheAligned(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});

    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int z = inputZ / 2;

    for (auto _ : state) {
        for(int y = 0; y < outputY; ++y) {
            for(int x = 0; x < outputX; ++x) {
                res(y, x) = data(z, y, x);
            }
        }
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_tensor_copy_complete_Z_rawAgainstCache(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int z = inputZ / 2;
    raw::ArrayView3D rawView{
        {inputZ, inputY, inputX}
        , {0, paddingStartY, paddingStartX}
        , {1, strideY, strideX}
    };
    auto rawData = data.data();
    auto resData = res.data();

    for (auto _ : state) {
        for(int x = 0; x < outputX; ++x) {
            for(int y = 0; y < outputY; ++y) {
                *raw::access_direct(resData, outputX, y, x) = *rawView.access(rawData, z, y, x);
            }
        }
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_tensor_copy_complete_Z_rawCacheAligned(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int z = inputZ / 2;
    raw::ArrayView3D rawView{
        {inputZ, inputY, inputX}
        , {0, paddingStartY, paddingStartX}
        , {1, strideY, strideX}
    };
    auto rawData = data.data();
    auto resData = res.data();

    for (auto _ : state) {
        for(int y = 0; y < outputY; ++y) {
            for(int x = 0; x < outputX; ++x) {
                *raw::access_direct(resData, outputX, y, x) = *rawView.access(rawData, z, y, x);
            }
        }
        benchmark::DoNotOptimize(res.data());
    }
}

BENCHMARK_SINGLE_VERSION(benchmark_tensor_copy_complete_Z_againstCache);
BENCHMARK_SINGLE_VERSION(benchmark_tensor_copy_complete_Z_cacheAligned);
BENCHMARK_SINGLE_VERSION(benchmark_tensor_copy_complete_Z_rawAgainstCache);
BENCHMARK_SINGLE_VERSION(benchmark_tensor_copy_complete_Z_rawCacheAligned);

BENCHMARK_MAIN();