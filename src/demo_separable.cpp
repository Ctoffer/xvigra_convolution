#include <array>
#include <stdexcept>
#include <iostream>
#include <ostream>
#include <type_traits>
#include <vector>

#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xview.hpp"

#include "xvigra/separable_convolution.hpp"
#include "xvigra/convolution_util.hpp"

template <typename T, std::size_t Dim>
std::ostream& operator<<(std::ostream& out, const std::array<T, Dim> data) {
    if constexpr (Dim == 0) {
        return out << "[]";
    } else if constexpr (Dim == 1) {
        return out << "[" << data[0] << "]";
    } else {
        out << "[" << data[0];
        for (std::size_t idx = 1; idx < Dim; ++idx) {
            out << ", " << data[idx];
        }    
        return out << "]";
    }
    
}


template <typename T>
std::ostream& operator<<(std::ostream& out, const std::vector<T> data) {
    if (data.size() == 0) {
        return out << "[]";
    } else if (data.size() == 1) {
        return out << "[" << data[0] << "]";
    } else {
        out << "[" << data[0];
        for (std::size_t idx = 1; idx < data.size(); ++idx) {
            out << ", " << data[idx];
        }    
        return out << "]";
    }
    
}


template <typename T>
void printTensor(const xt::xtensor<T, 3>& tensor) {
    auto tensorShape = tensor.shape();
    std::cout << "Shape: " << tensorShape << std::setprecision(2) << std::fixed << "\n";

    for (std::size_t y = 0; y < tensorShape[0]; ++y) {
        std::cout << "[";
        for (std::size_t x = 0; x < tensorShape[1]; ++x) {
            std::vector<T> channelValues;
            for (std::size_t c = 0; c < tensorShape[2]; ++c) {
                channelValues.push_back(tensor(y, x, c));
            }
            std::cout << channelValues << " ";
        }
        std::cout << "]\n";
    }

    std::cout << std::endl;
}


void runFirstWithSeparableConvolve1D() {
    std::cout << "runFirstWithSeparableConvolve1D" << std::endl;
    std::cout << "================================================================================" << std::endl;
    using InputType = double;
    using KernelType = double;

    xt::xtensor<InputType, 2> arr = {{1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}, {1, 2, 3, 4, 5}};
    xt::xtensor<KernelType, 1> rawKernel{1.0, 1.3, 1.7};
    xvigra::KernelOptions options;

    auto result = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(arr, rawKernel, options);

    std::cout << std::endl;
    std::cout << result.shape() << std::endl;
    std::cout << result << std::endl;

    auto result2 = xvigra::separableConvolve<InputType, KernelType, xvigra::ChannelPosition::FIRST, 1>(
        arr, 
        {rawKernel}, 
        {options}
    );
    std::cout << result2.shape() << std::endl;
    std::cout << result2 << std::endl;
    std::cout << std::endl;
}


void runLastWithSeparableConvolve1D() {
    std::cout << "runLastWithSeparableConvolve1D" << std::endl;
    std::cout << "================================================================================" << std::endl;
    using InputType = double;
    using KernelType = double;

    xt::xtensor<InputType, 2> arr = {{1, 1, 1}, {2, 2, 2}, {3, 3, 3}, {4, 4, 4}, {5, 5, 5}};
    xt::xtensor<KernelType, 1> rawKernel{1.0, 1.3, 1.7};
    xvigra::KernelOptions options;

    auto result = xvigra::separableConvolve1D<InputType, KernelType, xvigra::ChannelPosition::LAST>(arr, rawKernel, options);

    std::cout << std::endl;
    std::cout << result.shape() << std::endl;
    std::cout << result << std::endl;

    auto result2 = xvigra::separableConvolve<InputType, KernelType, xvigra::ChannelPosition::LAST, 1>(
        arr, 
        {rawKernel}, 
        {options}
    );
    std::cout << result2.shape() << std::endl;
    std::cout << result2 << std::endl;
    std::cout << std::endl;
}


void runFirstWithSeparableConvolve2D() {
    std::cout << "runFirstWithSeparableConvolve2D" << std::endl;
    std::cout << "================================================================================" << std::endl;
    using InputType = double;
    using KernelType = double;
    using ResultType = double;

    xt::xtensor<InputType, 3> arr = 
    {
        {
            { 1,  2,  3,  4,  5},
            { 6,  7,  8,  9, 10},
            {11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20},
            {21, 22, 23, 24, 25},
            {26, 27, 28, 29, 30}
        },
        {
            {31, 32, 33, 34, 35},
            {36, 37, 38, 39, 40},
            {41, 42, 43, 44, 45},
            {46, 47, 48, 49, 50},
            {51, 52, 53, 54, 55},
            {56, 57, 58, 59, 60}
        },
        {
            {30, 29, 28, 27, 26},
            {25, 24, 23, 22, 21},
            {20, 19, 18, 17, 16},
            {15, 14, 13, 12, 11},
            {10,  9,  8,  7,  6},
            { 5,  4,  3,  2,  1}
        }
    };

    xt::xtensor<KernelType, 1> rawKernelY{1.0, 1.3, 1.7};
    xt::xtensor<KernelType, 1> rawKernelX{1.0, 1.3, 1.7};
    xvigra::KernelOptions optionsY;
    xvigra::KernelOptions optionsX;

    auto result = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::FIRST>(
        arr, 
        {rawKernelY, rawKernelX}, 
        {optionsY, optionsX}
    );


    std::cout << std::endl;
    printTensor<ResultType>(result);

    auto result2 = xvigra::separableConvolve<InputType, KernelType, xvigra::ChannelPosition::FIRST, 2>(
        arr, 
        {rawKernelY, rawKernelX}, 
        {optionsY, optionsX}
    );
    printTensor<ResultType>(result2);
}


void runLastWithSeparableConvolve2D() {
    std::cout << "runLastWithSeparableConvolve2D" << std::endl;
    std::cout << "================================================================================" << std::endl;
    using InputType = double;
    using KernelType = double;
    using ResultType = double;

    xt::xtensor<InputType, 3> arr = 
    {
        {{ 1, 31, 30}, { 2, 32, 29}, { 3, 33, 28}, { 4, 34, 27}, { 5, 35, 26}},
        {{ 6, 36, 25}, { 7, 37, 24}, { 8, 38, 23}, { 9, 39, 22}, {10, 40, 21}},
        {{11, 41, 20}, {12, 42, 19}, {13, 43, 18}, {14, 44, 17}, {15, 45, 16}},
        {{16, 46, 15}, {17, 47, 14}, {18, 48, 13}, {19, 49, 12}, {20, 50, 11}},
        {{21, 51, 10}, {22, 52,  9}, {23, 53,  8}, {24, 54,  7}, {25, 55,  6}},
        {{26, 56,  5}, {27, 57,  4}, {28, 58,  3}, {29, 59,  2}, {30, 60,  1}},
     };

    xt::xtensor<KernelType, 1> rawKernelY{1.0, 1.3, 1.7};
    xt::xtensor<KernelType, 1> rawKernelX{1.0, 1.3, 1.7};
    xvigra::KernelOptions optionsY;
    xvigra::KernelOptions optionsX;

    auto result = xvigra::separableConvolve2D<InputType, KernelType, xvigra::ChannelPosition::LAST>(
        arr, 
        {rawKernelY, rawKernelX}, 
        {optionsY, optionsX}
    );


    std::cout << std::endl;
    printTensor<ResultType>(result);

    auto result2 = xvigra::separableConvolve<InputType, KernelType, xvigra::ChannelPosition::LAST, 2>(
        arr, 
        {rawKernelY, rawKernelX}, 
        {optionsY, optionsX}
    );
    printTensor<ResultType>(result2);
}


int main() {
    runFirstWithSeparableConvolve1D();
    runLastWithSeparableConvolve1D();
    
    runFirstWithSeparableConvolve2D();
    runLastWithSeparableConvolve2D();

    return 0;
}