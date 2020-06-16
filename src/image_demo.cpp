#include <iostream>
#include <tuple>
#include <string>

#include "conv1d_v2.hpp"
#include "conv2d_v2.hpp"
#include "image-io.hpp"

#include "xtensor/xarray.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor-io/ximage.hpp"

int main() {
	try {
		auto image = loadImage(".\\test.png");
		std::cout << image.shape().at(0) 
				  << "x"  << image.shape().at(1)
				  << "x"  << image.shape().at(2)
				  << std::endl;

		Conv2D convolution(std::tuple<int, int>(0, 0), std::tuple<int, int>(1, 1), std::tuple<int, int>(1, 1));
		xt::xtensor<int, 4> filter{
			{
				{{0,-1,0},{-1,5,-1},{0,-1,0}}, 
				{{0,1,0},{1,-4,1},{0,1,0}}, 
				{{-2, -1, 0}, {-1,1,1}, {0, 1, 2}}
			},
			
			{
				{{0,-1,0},{-1,5,-1},{0,-1,0}}, 
				{{0,1,0},{1,-4,1},{0,1,0}}, 
				{{-2, -1, 0}, {-1,1,1}, {0, 1, 2}}
			},
			
			{ 
				{{0,-1,0},{-1,5,-1},{0,-1,0}}, 
				{{0,1,0},{1,-4,1},{0,1,0}}, 
				{{-2, -1, 0}, {-1,1,1}, {0, 1, 2}}
			}
		};
		
		auto result = convolution(image, filter, false);
		saveImage(".\\test_conv.png", result);
	} catch (const std::runtime_error& e) {
	    // your error handling code here
	    std::cout << e.what() << "\n";
	} catch(const std::invalid_argument& e) {
		std::cout << e.what() << "\n";
	}catch(...) {
		std::cout << "Something went wrong" << std::endl;
	}
	

	return 0;
}