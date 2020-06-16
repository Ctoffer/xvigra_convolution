#ifndef XVIGRA_IMAGE_IO_HPP
#define XVIGRA_IMAGE_IO_HPP

#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xarray.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xtensor.hpp"

#include "xtensor-io/ximage.hpp"

using StdPath = std::filesystem::path;

namespace xvigra {
    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ forward declaration - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename ElementType=int>
    xt::xtensor<ElementType, 3> loadImageAsXTensor(const StdPath&);

    template <typename ElementType=int>
    xt::xarray<ElementType> loadImageAsXArray(const StdPath&);

    template <typename ElementType=int>
    xt::xarray<ElementType> internalLoadImage(const std::string&);

    template <typename Container>
    void saveImage(const StdPath&, const Container&);

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ forward declaration - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ function definitions - begin                                                                                 ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    /**
     * <p>
     *    Loads image data from a given path into a xt::xtensor with the desired ElementType.
     *    The result has always 3 dimension H x W x C.
     * </p>
     * 
     * @see xvigra::internalLoadImage
     * @param path Path from which the image should be read
     * @return The loaded image as xt::xtensor
     */
    template <typename ElementType>
    xt::xtensor<ElementType, 3> loadImageAsXTensor(const StdPath& path) {
        return xt::xtensor<ElementType, 3>(internalLoadImage<ElementType>(path.u8string()));
    }

    /**
     * <p>
     *    Loads image data from a given path into a xt::xarray with the desired ElementType.
     *    The result has always 3 dimension H x W x C.
     * </p>
     * 
     * @see xvigra::internalLoadImage
     * @param path Path from which the image should be read
     * @return The loaded image as xt::xarray
     */
    template <typename ElementType>
    xt::xarray<ElementType> loadImageAsXArray(const StdPath& path) {
        return xt::xarray<ElementType>(internalLoadImage<ElementType>(path.u8string()));
    }

    /**
     * <p>
     *    Internal function to load the load an image from the given path.
     * </p>
     * <p>
     *    If the specified ElementType is a floating point type, then the values in the
     *    result will be between 0.0 and 1.0 (both included).<br>
     *    If the specified ElementType ist an integer type, the the values in the result
     *    will be between 0 and 255 (both included).<br>
     * </p>
     *
     * @tparam ElementType Type of the elements of the loaded image
     * @param path Path from which the image should be read
     * @return The loaded image
     */
    template <typename ElementType>
    xt::xarray<ElementType> internalLoadImage(const std::string& path) {
        auto image = xt::cast<ElementType>(xt::load_image<uint8_t>(path));

        if constexpr (std::is_floating_point_v<ElementType>) {
            return image / static_cast<ElementType>(xt::amax(image)[0]);
        } 

        return image;
    }

    /**
     * <p>
     *    This functions saves the given container based on xt::dump_image to an image file.
     *    Only 2 or 3 dimensional container are allowed, 2 dimensional will be interpreted
     *    as H x W x C with C = 1.
     * </p>
     * <p>
     *    The container needs the matching number of channels for the desired output file
     *    format. Example: You want to save to a pgm-file - so you need an input with the
     *    dimensions H x W or H x W x 1.
     * </p>
     *
     * @tparam Container Type of the container (e.g. xt::xarray, xt::xtensor or some view)
     * @param path The path where the image should be saved
     * @param container The container containing the image data
     * @throws std::invalid_argument If the given container has not 2 or 3 dimensions
     */
    template <typename Container>
    void saveImage(const StdPath& path, const Container& container) {
        using ElementType = typename Container::value_type;
        auto inputShape = container.shape();
        auto numberOfDimensions = inputShape.size();
      
        if (numberOfDimensions < 2 || 3 < numberOfDimensions) {
            throw std::invalid_argument("The number of dimensions should be 2 (H x W) or 3 (H x W x C).");
        }

        xt::xtensor<ElementType, 3> reshapedData;
        if (numberOfDimensions == 2) {
            reshapedData = xt::xtensor<ElementType, 3>(xt::reshape_view(container, std::vector<std::size_t>{inputShape[0], inputShape[1], 1}));
        } else {
            reshapedData = xt::xtensor<ElementType, 3>(container);
        }

        xt::xtensor<uint8_t, 3> normedData;
        if constexpr (std::is_floating_point_v<ElementType>) {
            normedData = xt::cast<uint8_t>(xt::round(255 * reshapedData));
        } else {
            normedData = xt::cast<uint8_t>(reshapedData);
        }

        xt::dump_image(path.u8string(), normedData);
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ function definitions - end                                                                                   ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
} // namespace xvigra

#endif // XVIGRA_IMAGE_IO_HPP