#include <stdexcept>

#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"

#include "xvigra/explicit_convolution.hpp"
#include "xvigra/convolution_util.hpp"


void runDemo(const xt::xtensor<int, 1>& data, const xt::xtensor<double, 3>& kernel, const xvigra::KernelOptions& options) {
    std::cout << "Padding: " << options.getPadding() 
              << ", Stride: " << options.stride 
              << ", Dilation: " << options.dilation 
              << std::endl;
    std::cout << "Data: " << data << std::endl;
    std::cout << "Kernel: " << kernel << std::endl;
    try {
        std::cout << "Result: " << xvigra::convolve1DImplicit(data, kernel, options) << std::endl; 
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    std::cout << std::endl;
}


void runDemo(const xt::xtensor<int, 2>& data, const xt::xtensor<double, 3>& kernel, const xvigra::KernelOptions& options) {
    std::cout << "Padding: " << options.getPadding() 
              << ", Stride: " << options.stride 
              << ", Dilation: " << options.dilation 
              << std::endl;
    std::cout << "Data: " << data << std::endl;
    std::cout << "Kernel: " << kernel << std::endl;
    try {
        std::cout << "Result: " << xvigra::convolve1D(data, kernel, options) << std::endl; 
    } catch (const std::exception& e) {
        std::cout << e.what() << std::endl;
    }
    std::cout << std::endl;
}


void runDemos(const xt::xtensor<int, 1>& data, const xt::xtensor<double, 3>& kernel, xvigra::KernelOptions& options) {
    std::cout << "======================================================================" << std::endl;
    options.setPadding(0);
    runDemo(data, kernel, options);

    options.setPadding(1);
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());
    std::cout << "asymmetricReflect" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::avoid());
    std::cout << "avoid" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::repeat());
    std::cout << "repeat" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());
    std::cout << "symmetricReflect" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::wrap());
    std::cout << "wrap" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::constant(2));
    std::cout << "constant(2)" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatmentBegin(xvigra::BorderTreatment::avoid());
    options.setBorderTreatmentEnd(xvigra::BorderTreatment::asymmetricReflect());
    std::cout << "avoid - asymmetricReflect" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatmentBegin(xvigra::BorderTreatment::asymmetricReflect());
    options.setBorderTreatmentEnd(xvigra::BorderTreatment::avoid());
    std::cout << "asymmetricReflect - avoid" << std::endl;
    runDemo(data, kernel, options);
    std::cout << "======================================================================" << std::endl;
}


void runDemos(const xt::xtensor<int, 2>& data, const xt::xtensor<double, 3>& kernel, xvigra::KernelOptions& options) {
    std::cout << "======================================================================" << std::endl;
    options.setPadding(0);
    runDemo(data, kernel, options);

    options.setPadding(1);
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::asymmetricReflect());
    std::cout << "asymmetricReflect" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::avoid());
    std::cout << "avoid" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::repeat());
    std::cout << "repeat" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::symmetricReflect());
    std::cout << "symmetricReflect" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::wrap());
    std::cout << "wrap" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatment(xvigra::BorderTreatment::constant(2));
    std::cout << "constant(2)" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatmentBegin(xvigra::BorderTreatment::avoid());
    options.setBorderTreatmentEnd(xvigra::BorderTreatment::asymmetricReflect());
    std::cout << "avoid - asymmetricReflect" << std::endl;
    runDemo(data, kernel, options);

    options.setBorderTreatmentBegin(xvigra::BorderTreatment::asymmetricReflect());
    options.setBorderTreatmentEnd(xvigra::BorderTreatment::avoid());
    std::cout << "asymmetricReflect - avoid" << std::endl;
    runDemo(data, kernel, options);
    std::cout << "======================================================================" << std::endl;
}


void runWithOptions(xvigra::KernelOptions& options, bool symmetric=false) {
    xt::xtensor<double, 3> kernel;
    if (symmetric) {
        kernel = xt::xtensor<double, 3>{{{1, 1.3, 1.7, 2.11}}};
    } else {
        kernel = xt::xtensor<double, 3>{{{1, 1.3, 1.7}}};
    }
    
    {
        options.channelPosition = xvigra::ChannelPosition::FIRST;
        xt::xtensor<int, 2> data{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
        runDemos(data, kernel, options);
    }
   
    {
        options.channelPosition = xvigra::ChannelPosition::LAST;
        xt::xtensor<int, 2> data{{1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}};
        runDemos(data, kernel, options);
    }

    {
        options.channelPosition = xvigra::ChannelPosition::IMPLICIT;
        xt::xtensor<int, 1> data{1, 2, 3, 4, 5, 6, 7, 8, 9};
        runDemos(data, kernel, options);
    }
}


void completeDemoRun(bool symmetric) {
    xvigra::KernelOptions options;
   
    runWithOptions(options, symmetric);

    options.stride = 2;
    runWithOptions(options, symmetric);
    options.stride = 1;

    options.dilation = 2;
    runWithOptions(options, symmetric);
    options.dilation = 1;

    options.stride = 2;
    options.dilation = 2;
    runWithOptions(options, symmetric);
}


int main() {
    completeDemoRun(false);
    completeDemoRun(true);

    return 0;
}