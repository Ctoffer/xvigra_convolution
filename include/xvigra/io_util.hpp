#ifndef XVIGRA_IO_UTIL_HPP
#define XVIGRA_IO_UTIL_HPP

#include <array>
#include <iostream>
#include <ostream>
#include <vector>

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xtensor.hpp"

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


#endif // XVIGRA_IO_UTIL_HPP
