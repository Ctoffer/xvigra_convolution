#include <filesystem>
#include <stdexcept>
#include <type_traits>

#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#include "xtensor/xarray.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xtensor.hpp"
#include "xtensor/xio.hpp"

#include "xvigra/image_io.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ external namespace - begin                                                                                   ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

namespace fs = std::filesystem;

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ external namespace - end                                                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ type definition - begin                                                                                      ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

using StdPath = fs::path;
template <typename T>
using ImageTensor = typename xt::xtensor<T, 3>;
template <typename T>
using ImageArray = typename xt::xarray<T>;

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ type definition - end                                                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ constant definition - begin                                                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

constexpr auto TEST_DEMO_PATH = "./resources/tests/test_demo.ppm";

constexpr auto TEST_SMALL_IMAGE_PGM = "./resources/tests/Piercing-The-Ocean_Small.pgm";
constexpr auto TEST_IMAGE_PBM = "./resources/tests/Piercing-The-Ocean.pbm";
constexpr auto TEST_IMAGE_PGM = "./resources/tests/Piercing-The-Ocean.pgm";
constexpr auto TEST_IMAGE_PPM = "./resources/tests/Piercing-The-Ocean.ppm";
constexpr auto TEST_IMAGE_PNG = "./resources/tests/Piercing-The-Ocean.png";

constexpr auto TEST_IMAGE_COPY_PBM = "./resources/tests/Piercing-The-Ocean_Copy.pbm";
constexpr auto TEST_IMAGE_COPY_PGM = "./resources/tests/Piercing-The-Ocean_Copy.pgm";
constexpr auto TEST_IMAGE_COPY_PPM = "./resources/tests/Piercing-The-Ocean_Copy.ppm";
constexpr auto TEST_IMAGE_COPY_PNG = "./resources/tests/Piercing-The-Ocean_Copy.png";

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ constant definition - end                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ forward declaration - begin                                                                                  ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

template <typename IntegerContainer, typename FloatContainer>
void checkIntegerVsFloatContainer(const IntegerContainer&, const FloatContainer&);

template <typename ElementType>
void checkSaveImageFromTensor(const StdPath&, const StdPath&);

template <typename ElementType>
void checkSaveImageFromArray(const StdPath&, const StdPath&);

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ forward declaration - end                                                                                    ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test loadImage - begin                                                                                       ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE("Test loadImage") {
    SUBCASE("Small Input Image") {
        SUBCASE("As xTensor") {
            ImageTensor<int> expectedImage{
                {{217}, {213}, {222}, {219}, {217}, {223}, {214}, {210}, {199}, {209}, {207}, {204}, {197}, {192}, {198}, {200}, {191}, {180}, {177}, {174}, {168}, {150}, {148}, {147}, {151}, {138}},
                {{209}, {214}, {216}, {229}, {221}, {217}, {219}, {212}, {210}, {202}, {203}, {202}, {215}, {225}, {223}, {220}, {209}, {206}, {192}, {185}, {177}, {173}, {147}, {138}, {147}, {148}},
                {{207}, {214}, {213}, {221}, {214}, {214}, {219}, {209}, {217}, {221}, {219}, {207}, {164}, {164}, {158}, {153}, {171}, {216}, {212}, {192}, {173}, {167}, {169}, {162}, {147}, {151}},
                {{214}, {212}, {206}, {215}, {219}, {216}, {227}, {225}, {205}, {218}, {186}, {153}, {169}, {186}, {175}, {187}, {153}, {144}, {200}, {211}, {194}, {162}, {163}, {167}, {158}, {164}},
                {{218}, {202}, {207}, {222}, {225}, {225}, {189}, {166}, {176}, {191}, {204}, {172}, {176}, {162}, {174}, {168}, {177}, {171}, {137}, {194}, {205}, {174}, {170}, {162}, {173}, {162}},
                {{217}, {223}, {219}, {211}, {231}, {243}, {148}, {208}, {195}, {159}, {153}, {117}, {114}, {113}, {105}, {136}, {164}, {175}, {169}, {150}, {210}, {204}, {186}, {167}, {180}, {179}},
                {{213}, {218}, {219}, {224}, {246}, {246}, {199}, {191}, {139}, {114}, {138}, {158}, {156}, {122}, {125}, {107}, {103}, {116}, {148}, {151}, {162}, {197}, {190}, {181}, {179}, {173}},
                {{234}, {218}, {224}, {243}, {247}, {248}, {224}, {137}, {130}, {129}, {119}, {149}, {130}, {124}, {128}, {113}, {118}, { 99}, {103}, {152}, {149}, {131}, {205}, {194}, {178}, {178}},
                {{239}, {246}, {248}, {248}, {249}, {246}, {183}, {162}, {160}, {146}, {119}, {171}, {183}, {178}, {179}, {195}, {160}, {126}, {102}, {101}, {158}, {160}, {212}, {211}, {207}, {199}},
                {{238}, {246}, {248}, {249}, {249}, {249}, {144}, {176}, {185}, {162}, {143}, {158}, {139}, {144}, {126}, {163}, {181}, {188}, {165}, {115}, {131}, {169}, {215}, {214}, {190}, {209}},
                {{238}, {247}, {248}, {249}, {250}, {250}, {145}, {175}, {180}, {167}, {171}, {161}, {238}, {221}, {157}, {181}, {165}, {165}, {153}, {161}, {118}, {137}, {211}, {203}, {191}, {196}},
                {{238}, {238}, {244}, {249}, {250}, {250}, {209}, {179}, {179}, {168}, {156}, {180}, {237}, {240}, {215}, {175}, {201}, {168}, {153}, {169}, {120}, {180}, {221}, {193}, {192}, {157}},
                {{217}, {221}, {242}, {249}, {249}, {249}, {205}, {173}, {185}, {201}, {220}, {176}, {193}, {239}, {231}, {188}, {175}, {167}, {173}, {180}, {153}, {204}, {224}, {200}, {165}, {127}},
                {{165}, {207}, {248}, {241}, {249}, {231}, {182}, {194}, {246}, {248}, {213}, {156}, {171}, {234}, {239}, {178}, {163}, {191}, {187}, {176}, {153}, {224}, {212}, {192}, {203}, {155}},
                {{182}, {178}, {197}, {201}, {247}, {212}, {185}, {237}, {250}, {249}, {248}, {228}, {234}, {243}, {241}, {232}, {234}, {187}, {178}, {186}, {167}, {228}, {212}, {180}, {169}, {159}},
                {{194}, {176}, {181}, {231}, {230}, {210}, {218}, {251}, {251}, {245}, {215}, {223}, {233}, {245}, {244}, {241}, {194}, {187}, {190}, {180}, {214}, {229}, {226}, {164}, {158}, {149}},
                {{184}, {222}, {192}, {237}, {234}, {204}, {249}, {246}, {231}, {193}, {146}, {180}, {172}, {240}, {246}, {219}, {186}, {193}, {190}, {208}, {205}, {181}, {187}, {213}, {151}, {173}},
                {{185}, {177}, {217}, {235}, {239}, {242}, {233}, {212}, {183}, {152}, {144}, {146}, {202}, {248}, {226}, {152}, {170}, {199}, {161}, {194}, {128}, {158}, {161}, {164}, {141}, {197}},
                {{203}, {164}, {226}, {242}, {238}, {242}, {223}, {135}, {120}, {103}, {197}, {206}, {198}, {161}, {111}, {210}, {243}, {236}, {218}, {201}, {147}, {202}, {145}, {142}, {149}, {180}},
                {{212}, {175}, {253}, {226}, {221}, {195}, {154}, {132}, {183}, {210}, {199}, {165}, {154}, {176}, {215}, {246}, {232}, {184}, {176}, {165}, {200}, {219}, {183}, {195}, {173}, {178}},
                {{204}, {170}, {243}, {216}, {194}, {194}, {169}, {124}, {159}, {188}, {156}, {196}, {213}, {190}, {244}, {219}, {160}, {178}, {209}, {137}, {132}, {217}, {181}, {115}, {157}, {153}},
                {{190}, {171}, {239}, {145}, {172}, {166}, {131}, {195}, {218}, {201}, {213}, {216}, {212}, {204}, {228}, {154}, {143}, {216}, {162}, {118}, {184}, {188}, {137}, {150}, {153}, {176}},
                {{192}, {151}, {205}, {190}, {170}, {174}, {204}, {223}, {218}, {217}, {188}, {221}, {185}, {195}, {198}, {159}, {169}, {210}, {181}, {197}, {160}, {147}, {151}, {160}, {179}, {162}},
                {{185}, {185}, {201}, {193}, {181}, {148}, {182}, {183}, {175}, {171}, {124}, {165}, {191}, {187}, {190}, {161}, {166}, {198}, {187}, {175}, {166}, {181}, {167}, {171}, {172}, {175}},
                {{197}, {172}, {212}, {203}, {196}, {204}, {200}, {183}, {191}, {168}, {167}, {163}, {187}, {203}, {210}, {181}, {171}, {164}, {154}, {159}, {180}, {194}, {180}, {182}, {173}, {176}},
            };
            ImageTensor<int> actualImage = xvigra::loadImageAsXTensor(TEST_SMALL_IMAGE_PGM);

            CHECK(xt::amin(expectedImage)[0] == xt::amin(actualImage)[0]);
            CHECK(xt::amax(expectedImage)[0] == xt::amax(actualImage)[0]);
            CHECK(expectedImage.shape() == actualImage.shape());
            CHECK(expectedImage == actualImage);
        }

        SUBCASE("As xArray") {
            ImageArray<int> expectedImage =  ImageArray<int>(ImageTensor<int>({
                {{217}, {213}, {222}, {219}, {217}, {223}, {214}, {210}, {199}, {209}, {207}, {204}, {197}, {192}, {198}, {200}, {191}, {180}, {177}, {174}, {168}, {150}, {148}, {147}, {151}, {138}},
                {{209}, {214}, {216}, {229}, {221}, {217}, {219}, {212}, {210}, {202}, {203}, {202}, {215}, {225}, {223}, {220}, {209}, {206}, {192}, {185}, {177}, {173}, {147}, {138}, {147}, {148}},
                {{207}, {214}, {213}, {221}, {214}, {214}, {219}, {209}, {217}, {221}, {219}, {207}, {164}, {164}, {158}, {153}, {171}, {216}, {212}, {192}, {173}, {167}, {169}, {162}, {147}, {151}},
                {{214}, {212}, {206}, {215}, {219}, {216}, {227}, {225}, {205}, {218}, {186}, {153}, {169}, {186}, {175}, {187}, {153}, {144}, {200}, {211}, {194}, {162}, {163}, {167}, {158}, {164}},
                {{218}, {202}, {207}, {222}, {225}, {225}, {189}, {166}, {176}, {191}, {204}, {172}, {176}, {162}, {174}, {168}, {177}, {171}, {137}, {194}, {205}, {174}, {170}, {162}, {173}, {162}},
                {{217}, {223}, {219}, {211}, {231}, {243}, {148}, {208}, {195}, {159}, {153}, {117}, {114}, {113}, {105}, {136}, {164}, {175}, {169}, {150}, {210}, {204}, {186}, {167}, {180}, {179}},
                {{213}, {218}, {219}, {224}, {246}, {246}, {199}, {191}, {139}, {114}, {138}, {158}, {156}, {122}, {125}, {107}, {103}, {116}, {148}, {151}, {162}, {197}, {190}, {181}, {179}, {173}},
                {{234}, {218}, {224}, {243}, {247}, {248}, {224}, {137}, {130}, {129}, {119}, {149}, {130}, {124}, {128}, {113}, {118}, { 99}, {103}, {152}, {149}, {131}, {205}, {194}, {178}, {178}},
                {{239}, {246}, {248}, {248}, {249}, {246}, {183}, {162}, {160}, {146}, {119}, {171}, {183}, {178}, {179}, {195}, {160}, {126}, {102}, {101}, {158}, {160}, {212}, {211}, {207}, {199}},
                {{238}, {246}, {248}, {249}, {249}, {249}, {144}, {176}, {185}, {162}, {143}, {158}, {139}, {144}, {126}, {163}, {181}, {188}, {165}, {115}, {131}, {169}, {215}, {214}, {190}, {209}},
                {{238}, {247}, {248}, {249}, {250}, {250}, {145}, {175}, {180}, {167}, {171}, {161}, {238}, {221}, {157}, {181}, {165}, {165}, {153}, {161}, {118}, {137}, {211}, {203}, {191}, {196}},
                {{238}, {238}, {244}, {249}, {250}, {250}, {209}, {179}, {179}, {168}, {156}, {180}, {237}, {240}, {215}, {175}, {201}, {168}, {153}, {169}, {120}, {180}, {221}, {193}, {192}, {157}},
                {{217}, {221}, {242}, {249}, {249}, {249}, {205}, {173}, {185}, {201}, {220}, {176}, {193}, {239}, {231}, {188}, {175}, {167}, {173}, {180}, {153}, {204}, {224}, {200}, {165}, {127}},
                {{165}, {207}, {248}, {241}, {249}, {231}, {182}, {194}, {246}, {248}, {213}, {156}, {171}, {234}, {239}, {178}, {163}, {191}, {187}, {176}, {153}, {224}, {212}, {192}, {203}, {155}},
                {{182}, {178}, {197}, {201}, {247}, {212}, {185}, {237}, {250}, {249}, {248}, {228}, {234}, {243}, {241}, {232}, {234}, {187}, {178}, {186}, {167}, {228}, {212}, {180}, {169}, {159}},
                {{194}, {176}, {181}, {231}, {230}, {210}, {218}, {251}, {251}, {245}, {215}, {223}, {233}, {245}, {244}, {241}, {194}, {187}, {190}, {180}, {214}, {229}, {226}, {164}, {158}, {149}},
                {{184}, {222}, {192}, {237}, {234}, {204}, {249}, {246}, {231}, {193}, {146}, {180}, {172}, {240}, {246}, {219}, {186}, {193}, {190}, {208}, {205}, {181}, {187}, {213}, {151}, {173}},
                {{185}, {177}, {217}, {235}, {239}, {242}, {233}, {212}, {183}, {152}, {144}, {146}, {202}, {248}, {226}, {152}, {170}, {199}, {161}, {194}, {128}, {158}, {161}, {164}, {141}, {197}},
                {{203}, {164}, {226}, {242}, {238}, {242}, {223}, {135}, {120}, {103}, {197}, {206}, {198}, {161}, {111}, {210}, {243}, {236}, {218}, {201}, {147}, {202}, {145}, {142}, {149}, {180}},
                {{212}, {175}, {253}, {226}, {221}, {195}, {154}, {132}, {183}, {210}, {199}, {165}, {154}, {176}, {215}, {246}, {232}, {184}, {176}, {165}, {200}, {219}, {183}, {195}, {173}, {178}},
                {{204}, {170}, {243}, {216}, {194}, {194}, {169}, {124}, {159}, {188}, {156}, {196}, {213}, {190}, {244}, {219}, {160}, {178}, {209}, {137}, {132}, {217}, {181}, {115}, {157}, {153}},
                {{190}, {171}, {239}, {145}, {172}, {166}, {131}, {195}, {218}, {201}, {213}, {216}, {212}, {204}, {228}, {154}, {143}, {216}, {162}, {118}, {184}, {188}, {137}, {150}, {153}, {176}},
                {{192}, {151}, {205}, {190}, {170}, {174}, {204}, {223}, {218}, {217}, {188}, {221}, {185}, {195}, {198}, {159}, {169}, {210}, {181}, {197}, {160}, {147}, {151}, {160}, {179}, {162}},
                {{185}, {185}, {201}, {193}, {181}, {148}, {182}, {183}, {175}, {171}, {124}, {165}, {191}, {187}, {190}, {161}, {166}, {198}, {187}, {175}, {166}, {181}, {167}, {171}, {172}, {175}},
                {{197}, {172}, {212}, {203}, {196}, {204}, {200}, {183}, {191}, {168}, {167}, {163}, {187}, {203}, {210}, {181}, {171}, {164}, {154}, {159}, {180}, {194}, {180}, {182}, {173}, {176}},
            }));
            ImageArray<int> actualImage = xvigra::loadImageAsXArray(TEST_SMALL_IMAGE_PGM);

            CHECK(xt::amin(expectedImage)[0] == xt::amin(actualImage)[0]);
            CHECK(xt::amax(expectedImage)[0] == xt::amax(actualImage)[0]);
            CHECK(expectedImage.shape() == actualImage.shape());
            CHECK(expectedImage == actualImage);
        }
    }

    SUBCASE("As xTensor") {
        SUBCASE("From PBM File") {
           ImageTensor<int> integerImage = xvigra::loadImageAsXTensor<int>(TEST_IMAGE_PBM);
           ImageTensor<float> floatImage = xvigra::loadImageAsXTensor<float>(TEST_IMAGE_PBM);

           checkIntegerVsFloatContainer
        (integerImage, floatImage);
        }

        SUBCASE("From PGM File") {
           ImageTensor<int> integerImage = xvigra::loadImageAsXTensor<int> (TEST_IMAGE_PGM);
           ImageTensor<float> floatImage = xvigra::loadImageAsXTensor<float>(TEST_IMAGE_PGM);

           checkIntegerVsFloatContainer
        (integerImage, floatImage);
        }

        SUBCASE("From PPM File") {
           ImageTensor<int> integerImage = xvigra::loadImageAsXTensor<int>(TEST_IMAGE_PPM);
           ImageTensor<float> floatImage = xvigra::loadImageAsXTensor<float>(TEST_IMAGE_PPM);

           checkIntegerVsFloatContainer
        (integerImage, floatImage);
        }

        SUBCASE("From PNG File") {
           ImageTensor<int> integerImage = xvigra::loadImageAsXTensor<int>(TEST_IMAGE_PNG);
           ImageTensor<float> floatImage = xvigra::loadImageAsXTensor<float>(TEST_IMAGE_PNG);

           checkIntegerVsFloatContainer
        (integerImage, floatImage);
        }
    }

    SUBCASE("As xArray") {
        SUBCASE("From PBM File") {
           ImageTensor<int> integerImage = xvigra::loadImageAsXArray<int>(TEST_IMAGE_PBM);
           ImageTensor<float> floatImage = xvigra::loadImageAsXArray<float>(TEST_IMAGE_PBM);

           checkIntegerVsFloatContainer
        (integerImage, floatImage);
        }

        SUBCASE("From PGM File") {
           ImageTensor<int> integerImage = xvigra::loadImageAsXArray<int>(TEST_IMAGE_PGM);
           ImageTensor<float> floatImage = xvigra::loadImageAsXArray<float>(TEST_IMAGE_PGM);

           checkIntegerVsFloatContainer
        (integerImage, floatImage);
        }

        SUBCASE("From PPM File") {
           ImageTensor<int> integerImage = xvigra::loadImageAsXArray<int>(TEST_IMAGE_PPM);
           ImageTensor<float> floatImage = xvigra::loadImageAsXArray<float>(TEST_IMAGE_PPM);

           checkIntegerVsFloatContainer
        (integerImage, floatImage);
        }

        SUBCASE("From PNG File") {
           ImageTensor<int> integerImage = xvigra::loadImageAsXArray<int>(TEST_IMAGE_PNG);
           ImageTensor<float> floatImage = xvigra::loadImageAsXArray<float>(TEST_IMAGE_PNG);

           checkIntegerVsFloatContainer
        (integerImage, floatImage);
        }
    }
}


template <typename IntegerContainer, typename FloatContainer>
void checkIntegerVsFloatContainer(const IntegerContainer& integerImage, const FloatContainer& floatImage) {
    int minimalIntegerValue = xt::amin(integerImage)[0];
    int maximalIntegerValue = xt::amax(integerImage)[0];
    float minimalFloatValue = xt::amin(floatImage)[0];
    float maximalFloatValue = xt::amax(floatImage)[0];

    CHECK(0 <= minimalIntegerValue);
    CHECK(maximalIntegerValue <= 255);
    CHECK(doctest::Approx(0.0f).epsilon(1e-7f) <= minimalFloatValue);
    CHECK(maximalFloatValue <= doctest::Approx(1.0f).epsilon(1e-7f));

    CHECK(minimalIntegerValue == static_cast<int>(minimalFloatValue * 255));
    CHECK(maximalIntegerValue == static_cast<int>(maximalFloatValue * 255));

    CHECK(integerImage.shape() == floatImage.shape());
    CHECK(integerImage == xt::cast<int>(255 * floatImage));
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test loadImage - end                                                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test saveImage - begin                                                                                       ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE_TEMPLATE("Test saveImage", ElementType, short, int, float, double) {
    SUBCASE("From xTensor") {
        SUBCASE("To PBM File") {
           checkSaveImageFromTensor<ElementType>(TEST_IMAGE_PBM, TEST_IMAGE_COPY_PBM);
        }

        SUBCASE("To PGM File") {
            checkSaveImageFromTensor<ElementType>(TEST_IMAGE_PGM, TEST_IMAGE_COPY_PGM);
        }

        SUBCASE("To PPM File") {
            checkSaveImageFromTensor<ElementType>(TEST_IMAGE_PPM, TEST_IMAGE_COPY_PPM);
        }

        SUBCASE("To PNG File") {
            checkSaveImageFromTensor<ElementType>(TEST_IMAGE_PNG, TEST_IMAGE_COPY_PNG);
        }
    }

    SUBCASE("From xArray") {
        SUBCASE("To PBM File") {
           checkSaveImageFromArray<ElementType>(TEST_IMAGE_PBM, TEST_IMAGE_COPY_PBM);
        }

        SUBCASE("To PGM File") {
            checkSaveImageFromArray<ElementType>(TEST_IMAGE_PGM, TEST_IMAGE_COPY_PGM);
        }

        SUBCASE("To PPM File") {
            checkSaveImageFromArray<ElementType>(TEST_IMAGE_PPM, TEST_IMAGE_COPY_PPM);
        }

        SUBCASE("To PNG File") {
            checkSaveImageFromArray<ElementType>(TEST_IMAGE_PNG, TEST_IMAGE_COPY_PNG);
        }
    }

    SUBCASE("Invalid number of dimensions") {
        SUBCASE("Less than 2") {
            xt::xtensor<ElementType, 1> data;
            if constexpr (std::is_floating_point_v<ElementType>) {
                data = xt::random::rand<ElementType>(
                    {11}, 
                    static_cast<ElementType>(0.0f), 
                    static_cast<ElementType>(1.0f)
                );
            } else {
                data = xt::random::randint<ElementType>(
                    {11}, 
                    0, 
                    255
                );
            }

            CHECK_THROWS_WITH_AS(
                xvigra::saveImage(TEST_DEMO_PATH, data), 
                "The number of dimensions should be 2 (H x W) or 3 (H x W x C).",
                 std::invalid_argument
            );
        }

        SUBCASE("More than 3") {
            xt::xarray<ElementType> data;
            if constexpr (std::is_floating_point_v<ElementType>) {
                data = xt::random::rand<ElementType>(
                    {11, 7, 13, 5}, 
                    static_cast<ElementType>(0.0f), 
                    static_cast<ElementType>(1.0f)
                );
            } else {
                data = xt::random::randint<ElementType>(
                    {11, 7, 13, 5},
                    0, 
                    255
                );
            }

            CHECK_THROWS_WITH_AS(
                xvigra::saveImage(TEST_DEMO_PATH, data), 
                "The number of dimensions should be 2 (H x W) or 3 (H x W x C).",
                 std::invalid_argument
            );
        }
    }
}


template <typename ElementType>
void checkSaveImageFromTensor(const StdPath& orginalPath, const StdPath& copyPath) {
    if(fs::exists(copyPath)) {
       fs::remove(copyPath);
    }
    CHECK(!fs::exists(copyPath));

    ImageTensor<ElementType> originalImage = xvigra::loadImageAsXTensor<ElementType>(orginalPath);
    xvigra::saveImage(copyPath, originalImage);
    CHECK(fs::exists(copyPath));
    ImageTensor<ElementType> copiedImage = xvigra::loadImageAsXTensor<ElementType>(copyPath);

    CHECK(originalImage.shape() == copiedImage.shape());
    CHECK(originalImage == copiedImage);
}


template <typename ElementType>
void checkSaveImageFromArray(const StdPath& orginalPath, const StdPath& copyPath) {
    if(fs::exists(copyPath)) {
       fs::remove(copyPath);
    }
    CHECK(!fs::exists(copyPath));

    ImageTensor<ElementType> originalImage = xvigra::loadImageAsXArray<ElementType>(orginalPath);
    xvigra::saveImage(copyPath, originalImage);
    CHECK(fs::exists(copyPath));
    ImageTensor<ElementType> copiedImage = xvigra::loadImageAsXArray<ElementType>(copyPath);

    CHECK(originalImage.shape() == copiedImage.shape());
    CHECK(originalImage == copiedImage);
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test saveImage - end                                                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝