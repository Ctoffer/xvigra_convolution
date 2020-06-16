#ifndef XVIGRA_IMAGE_IO_HPP
#define XVIGRA_IMAGE_IO_HPP

#include "xtensor/xtensor.hpp"
#include "xtensor-io/ximage.hpp"

xt::xtensor<int, 3> loadImage(const std::string&);
void saveImage(const std::string&, const xt::xtensor<int, 3>&);
xt::xtensor<uint8_t, 3> normalizeAfterConvolution(const xt::xtensor<int, 3>&);


xt::xtensor<int, 3> loadImage(const std::string& path) {
	auto image = xt::load_image<uint8_t>(path);
	return xt::xtensor<int, 3>(xt::cast<int>(image));
}

void saveImage(const std::string& path, const xt::xtensor<int, 3>& image){
	xt::dump_image(path, normalizeAfterConvolution(image));
}

xt::xtensor<uint8_t, 3> normalizeAfterConvolution(const xt::xtensor<int, 3>& image) {
	int minimum = xt::amin(image)[0];
	int maximum = xt::amax(image)[0];
	xt::xtensor<uint8_t,3> result(image.shape());

	int i = 0;
	auto resultIter = result.begin();
	for(auto imageIter = image.begin(); imageIter < image.end(); ++imageIter, ++resultIter, ++i) {
		int value = *imageIter;
		double normedValue = static_cast<double>(value - minimum) / (maximum - minimum);
		double scaledValue = 255 * normedValue;
		uint8_t resultValue = static_cast<uint8_t>(static_cast<int>(scaledValue));
		*resultIter = resultValue;
	}

	return result;
}

#endif