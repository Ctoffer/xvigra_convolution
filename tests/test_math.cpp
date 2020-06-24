#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#include "xvigra/math.hpp"

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test roundValue - begin                                                                                          ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

TEST_CASE("Test roundValue") {
    SUBCASE("Positive numbers") {
        SUBCASE("Round 5.291 to 0 decimals") {
            CHECK(xvigra::roundValue(5.291, 0) == 5.0);
        }

        SUBCASE("Round 5.91 to 0 decimals") {
            CHECK(xvigra::roundValue(5.91, 0) == 6.0);
        }

        SUBCASE("Round 5.291 to 1 decimals") {
            CHECK(xvigra::roundValue(5.291, 1) == 5.3);
        }

        SUBCASE("Round 5.291 to 2 decimals") {
            CHECK(xvigra::roundValue(5.291, 2) == 5.29);
        }

        SUBCASE("Round π to 11 decimals") {
            CHECK(xvigra::roundValue(3.14159265358979323846, 11) == 3.14159265359);
        }
    }
}

// ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
// ║ Test roundValue - end                                                                                            ║
// ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
