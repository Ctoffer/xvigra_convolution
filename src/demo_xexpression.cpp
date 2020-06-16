#include <iostream>
#include <iomanip>
#include <ostream>
#include <stdexcept>

#include "xtensor/xarray.hpp"
#include "xtensor/xexpression.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"


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
std::ostream& operator<<(std::ostream& out, const xt::svector<T> vector) {
    std::ostream_iterator<T> outIterator(out, ", ");
    out << "[";
    std::copy(vector.begin(), vector.end(), outIterator);
    return out << "]";
}


template <typename E>
void simpleCall(const xt::xexpression<E>& array) {
    using DerivedType = typename xt::xexpression<E>::derived_type;
    DerivedType container = array.derived_cast();
    using ElementType = typename DerivedType::value_type;

    std::cout << "Type : " << typeid(DerivedType).name() << std::endl;
    std::cout << "E    : " << typeid(ElementType).name() << std::endl;
    std::cout << "Shape: " << container.shape() << std::endl;
    std::cout << "Dim  : " << container.dimension() << std::endl;
    std::cout << "[0]  : " << container[0] << std::endl;
}


int main() {
    xt::xarray<int> array = {
        {1, 2, 3}, {4, 5, 6}
    };
    simpleCall(array);
    std::cout << std::endl;

    xt::xtensor<double, 3> tensor = 
    {
        {
            { 1.0,  2.0,  3.0,  4.0,  5.0},
            { 6.0,  7.0,  8.0,  9.0, 10.0},
            {11.0, 12.0, 13.0, 14.0, 15.0},
            {16.0, 17.0, 18.0, 19.0, 20.0},
            {21.0, 22.0, 23.0, 24.0, 25.0},
            {26.0, 27.0, 28.0, 29.0, 30.0}
        },
        {
            {31.0, 32.0, 33.0, 34.0, 35.0},
            {36.0, 37.0, 38.0, 39.0, 40.0},
            {41.0, 42.0, 43.0, 44.0, 45.0},
            {46.0, 47.0, 48.0, 49.0, 50.0},
            {51.0, 52.0, 53.0, 54.0, 55.0},
            {56.0, 57.0, 58.0, 59.0, 60.0}
        },
        {
            {30.0, 29.0, 28.0, 27.0, 26.0},
            {25.0, 24.0, 23.0, 22.0, 21.0},
            {20.0, 19.0, 18.0, 17.0, 16.0},
            {15.0, 14.0, 13.0, 12.0, 11.0},
            {10.0,  9.0,  8.0,  7.0,  6.0},
            { 5.0,  4.0,  3.0,  2.0,  1.0}
        }
    };
    simpleCall(tensor);
    std::cout << std::endl;

    auto view = xt::view(tensor, xt::range(0, 3, 2), 3, xt::all());
    simpleCall(view);
    std::cout << std::endl;

    auto stridedView = xt::strided_view(tensor, {xt::range(0, 3, 2), xt::range(3, 6), xt::all()});
    simpleCall(stridedView);

    return 0;
}