#ifndef XVIGRA_MATH_HPP
#define XVIGRA_MATH_HPP

#include <cmath>
#include <type_traits>

#include <xtensor/xtensor.hpp>

namespace xvigra {
    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ forward declaration - begin                                                                                  ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T>
    T roundValue(const T, int);

    template <typename E>
    E roundExpression(E&&, int);

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ forward declaration - end                                                                                    ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ round - begin                                                                                                ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝

    template <typename T>
    T roundValue(
        const T value, 
        int decimals
    ) {
        if constexpr (std::is_floating_point_v<T>) {
            int d = 0;
            std::size_t factor = std::pow(10, decimals);

            if ( (std::abs(value * factor * 10) - std::abs(static_cast<int>(value * factor) * 10) ) > 4) {
                d = 1;
            }

            return static_cast<T>( (std::floor(value * factor) + d) / static_cast<double>(factor) );
        } else {
            return value;
        }
    }

    template <typename E>
    E roundExpression(
        E&& expression, 
        int decimals
    ) {
        E copiedExpression(expression);
        auto sourceIter = expression.begin();
        auto sourceEnd = expression.end();
        auto targetIter = copiedExpression.begin();

        for (; sourceIter < sourceEnd; ++sourceIter, ++targetIter) {
            *targetIter = xvigra::roundValue(*sourceIter, decimals);
        }

        return copiedExpression;
    }

    // ╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
    // ║ round - begin                                                                                                ║
    // ╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
}

#endif // XVIGRA_MATH_HPP