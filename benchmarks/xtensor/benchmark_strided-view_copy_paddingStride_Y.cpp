#include <benchmark/benchmark.h>

#include "raw/array_view_3d.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xstrided_view.hpp"

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
    int paddingStartZ = 2;                                                                                                \
    int paddingEndZ = 1;                                                                                                  \
    int paddingStartX = 0;                                                                                                \
    int paddingEndX = 0;                                                                                                  \
                                                                                                                          \
    int strideZ = 2;                                                                                                      \
    int strideX = 1;                                                                                                      \
                                                                                                                          \
    int outputY = static_cast<int>(std::ceil(static_cast<double>((inputZ - paddingStartZ - paddingEndZ)) / (strideZ)));   \
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
void benchmark_xstrided_view_copy_paddingStride_Y_assign(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int y = inputY / 2;
    auto view = xt::strided_view(data, {xt::range(paddingStartZ, inputZ - paddingEndZ, strideZ), y, xt::range(paddingStartX, inputX - paddingEndX, strideX)});

    for (auto _ : state) {
        res = view;
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_xstrided_view_copy_paddingStride_Y_iterator(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int y = inputY / 2;
    auto view = xt::strided_view(data, {xt::range(paddingStartZ, inputZ - paddingEndZ, strideZ), y, xt::range(paddingStartX, inputX - paddingEndX, strideX)});

    for (auto _ : state) {
        std::copy(view.begin(), view.end(), res.begin());
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_xstrided_view_copy_paddingStride_Y_operatorCallAgainstCache(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int y = inputY / 2;
    auto view = xt::strided_view(data, {xt::range(paddingStartZ, inputZ - paddingEndZ, strideZ), y, xt::range(paddingStartX, inputX - paddingEndX, strideX)});

    for (auto _ : state) {
        for(int x = 0; x < outputX; ++x) {
            for(int z = 0; z < outputY; ++z) {
                res(z, x) = view(z, x);
            }
        }

        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_xstrided_view_copy_paddingStride_Y_operatorCallCacheAligned(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int y = inputY / 2;
    auto view = xt::strided_view(data, {xt::range(paddingStartZ, inputZ - paddingEndZ, strideZ), y, xt::range(paddingStartX, inputX - paddingEndX, strideX)});

    for (auto _ : state) {
         for(int z = 0; z < outputY; ++z) {
            for(int x = 0; x < outputX; ++x) {
                res(z, x) = view(z, x);
            }
        }

        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_xstrided_view_copy_paddingStride_Y_rawAgainstCache(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int y = inputY / 2;
    raw::ArrayView3D rawView{
        {inputZ, inputY, inputX}
        , {paddingStartZ, 0, paddingStartX}
        , {strideZ, 1, strideX}
    };
    auto rawData = data.data();
    auto resData = res.data();

    for (auto _ : state) {
        for(int x = 0; x < outputX; ++x) {
            for(int z = 0; z < outputY; ++z) {
                *raw::access_direct(resData, outputX, z, x) = *rawView.access(rawData, z, y, x);
            }
        }
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_xstrided_view_copy_paddingStride_Y_rawCacheAligned(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int y = inputY / 2;
    raw::ArrayView3D rawView{
        {inputZ, inputY, inputX}
        , {paddingStartZ, 0, paddingStartX}
        , {strideZ, 1, strideX}
    };
    auto rawData = data.data();
    auto resData = res.data();

    for (auto _ : state) {
        for(int z = 0; z < outputY; ++z) {
            for(int x = 0; x < outputX; ++x) {
                *raw::access_direct(resData, outputX, z, x) = *rawView.access(rawData, z, y, x);
            }
        }
        benchmark::DoNotOptimize(res.data());
    }
}

BENCHMARK_SINGLE_VERSION(benchmark_xstrided_view_copy_paddingStride_Y_assign);
BENCHMARK_SINGLE_VERSION(benchmark_xstrided_view_copy_paddingStride_Y_iterator);
BENCHMARK_SINGLE_VERSION(benchmark_xstrided_view_copy_paddingStride_Y_operatorCallAgainstCache);
BENCHMARK_SINGLE_VERSION(benchmark_xstrided_view_copy_paddingStride_Y_operatorCallCacheAligned);
BENCHMARK_SINGLE_VERSION(benchmark_xstrided_view_copy_paddingStride_Y_rawAgainstCache);
BENCHMARK_SINGLE_VERSION(benchmark_xstrided_view_copy_paddingStride_Y_rawCacheAligned);

BENCHMARK_MAIN();
