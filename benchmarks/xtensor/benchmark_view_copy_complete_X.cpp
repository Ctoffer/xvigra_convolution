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
    int paddingStartZ = 0;                                                                                                \
    int paddingEndZ = 0;                                                                                                  \
    int paddingStartY = 0;                                                                                                \
    int paddingEndY = 0;                                                                                                  \
                                                                                                                          \
    int strideZ = 1;                                                                                                      \
    int strideY = 1;                                                                                                      \
                                                                                                                          \
    int outputY = static_cast<int>(std::ceil(static_cast<double>((inputZ - paddingStartZ - paddingEndZ)) / (strideZ)));   \
    int outputX = static_cast<int>(std::ceil(static_cast<double>((inputY - paddingStartY - paddingEndY)) / (strideY)));

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
void benchmark_view_copy_complete_X_assign(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int x = inputX / 2;
    auto view = xt::view(data, xt::all(), xt::all(), x);

    for (auto _ : state) {
        res = view;
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_view_copy_complete_X_iterator(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int x = inputX / 2;
    auto view = xt::view(data, xt::all(), xt::all(), x);

    for (auto _ : state) {
        std::copy(view.begin(), view.end(), res.begin());
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_view_copy_complete_X_operatorCallAgainstCache(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int x = inputX / 2;
    auto view = xt::view(data, xt::all(), xt::all(), xt::range(x,x+1)); // HOTFIX: need range instead of pure x due to compiler errors
 
    for (auto _ : state) {
        for(int y = 0; y < outputX; ++y) {
            for(int z = 0; z < outputY; ++z) {
                res(z, y) = view(z, y);
            }
        }

        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_view_copy_complete_X_operatorCallCacheAligned(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int x = inputX / 2;
    auto view = xt::view(data, xt::all(), xt::all(), xt::range(x,x+1)); // HOTFIX: need range instead of pure x due to compiler errors

    for (auto _ : state) {
         for(int z = 0; z < outputY; ++z) {
            for(int y = 0; y < outputX; ++y) {
                 res(z, y) = view(z, y);
            }
        }

        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_view_copy_complete_X_rawAgainstCache(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int x = inputX / 2;
    raw::ArrayView3D rawView{
        {inputZ, inputY, inputX}
        , {paddingStartZ, paddingStartY, 0}
        , {strideZ, strideY, 1}
    };
    auto rawData = data.data();
    auto resData = res.data();

    for (auto _ : state) {
        for(int y = 0; y < outputX; ++y) {
            for(int z = 0; z < outputY; ++z) {
                *raw::access_direct(resData, outputX, z, y) = *rawView.access(rawData, z, y, x);
            }
        }
        benchmark::DoNotOptimize(res.data());
    }
}

template <typename ElementType>
void benchmark_view_copy_complete_X_rawCacheAligned(benchmark::State& state) {
    SETUP_VARIABLES

    xt::xtensor<ElementType, 3> data = xt::ones<ElementType>({inputZ, inputY, inputX});
    xt::xtensor<ElementType, 2> res = xt::ones<ElementType>({outputY, outputX});
    int i = 0;
    for(auto& elem: data) {
        elem = i++;
    }

    int x = inputX / 2;
    raw::ArrayView3D rawView{
        {inputZ, inputY, inputX}
        , {paddingStartZ, paddingStartY, 0}
        , {strideZ, strideY, 1}
    };
    auto rawData = data.data();
    auto resData = res.data();

    for (auto _ : state) {
        for(int z = 0; z < outputY; ++z) {
            for(int y = 0; y < outputX; ++y) {
                *raw::access_direct(resData, outputX, z, y) = *rawView.access(rawData, z, y, x);
            }
        }
        benchmark::DoNotOptimize(res.data());
    }
}

BENCHMARK_SINGLE_VERSION(benchmark_view_copy_complete_X_assign);
BENCHMARK_SINGLE_VERSION(benchmark_view_copy_complete_X_iterator);
BENCHMARK_SINGLE_VERSION(benchmark_view_copy_complete_X_operatorCallAgainstCache);
BENCHMARK_SINGLE_VERSION(benchmark_view_copy_complete_X_operatorCallCacheAligned);
BENCHMARK_SINGLE_VERSION(benchmark_view_copy_complete_X_rawAgainstCache);
BENCHMARK_SINGLE_VERSION(benchmark_view_copy_complete_X_rawCacheAligned);

BENCHMARK_MAIN();
