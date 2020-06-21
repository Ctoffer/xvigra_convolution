#ifndef XVIGRA_ITER_UTIL_HPP
#define XVIGRA_ITER_UTIL_HPP

#include <tuple>
#include <utility>

namespace xvigra {
    template <typename T>
    constexpr auto enumerate(T&& iterable){
        using TIterator = decltype(std::begin(std::declval<T>()));
        struct EnumerateIterator {
            std::size_t i;
            TIterator iter;

            bool operator!=(const EnumerateIterator& other) const { 
                return iter != other.iter; 
            }

            void operator++() {
                ++i;
                ++iter;
            }

            auto operator*() const { 
                return std::make_tuple(i, *iter); 
            }
        };

        struct Enumerate {
            T iterable;

            auto begin() { 
                return EnumerateIterator{0, std::begin(iterable)}; 
            }

            auto end() { 
                return EnumerateIterator{0, std::end(iterable)}; 
            }
        };
        return Enumerate{std::forward<T>(iterable)};
    }
}


#endif // XVIGRA_ITER_UTIL_HPP
