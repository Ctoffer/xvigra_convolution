#include <iostream>
#include <string>
#include <typeinfo>

#include "xtensor/xio.hpp"

#include "xvigra/image_io.hpp"


template <typename E>
void printImage(const E&);

template <typename ElementType>
void loadAndPrintImage(bool useTensor=true) {
    if(useTensor) {
        auto image = xvigra::loadImageAsXTensor<ElementType>("./resources/tests/Piercing-The-Ocean_Small.pgm");
        printImage(image);
    } else {
        auto image = xvigra::loadImageAsXArray<ElementType>("./resources/src/demo_small.png");
        printImage(image);
    }
}

template <typename E>
void printImage(const E& image) {
    std::cout << image.shape().size() << std::endl;
    std::cout << typeid(image).name() << std::endl;
    assert(image.shape().size() == 3);
    std::cout << image.shape()[0] << "x" << image.shape()[1] << "x" << image.shape()[2] << std::endl;
    std::cout << "for" << std::endl;
   
    for(std::size_t y = 0; y < image.shape()[0]; ++y) {
        for(std::size_t x = 0; x < image.shape()[1]; ++x) {
            std::cout << std::fixed << std::setprecision(3) << std::setfill(' ') << std::setw(3) << image(y, x, 0) << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    loadAndPrintImage<int>();
    loadAndPrintImage<float>();

    loadAndPrintImage<int>(false);
    loadAndPrintImage<float>(false);


    xvigra::saveImage(".\\resources\\src\\demo_small_copy-float.pgm", xvigra::loadImageAsXTensor<float>(".\\resources\\src\\demo_small.png"));
    xvigra::saveImage(".\\resources\\src\\demo_small_copy-int.png", xvigra::loadImageAsXTensor<int>(".\\resources\\src\\demo_small.pbm"));
    xvigra::saveImage("./resources/tests/demo.ppm", xt::xtensor<int, 1>({1, 2, 3, 4, 5}));

    return 0;
}