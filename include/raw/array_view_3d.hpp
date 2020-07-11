#ifndef RAW_ARRAY_VIEW_3D_HPP
#define RAW_ARRAY_VIEW_3D_HPP

#include <array>

namespace raw {
	inline int dotprod(int x1, int y1, int x2, int y2) {
		return x1*x2 + y1*y2;
	}	

	inline int dotprod(int x1, int y1, int z1, int x2, int y2, int z2) {
		return x1*x2 + y1*y2 + z1*z2;
	}	

	class ArrayView3D {
	private:
		std::array<int, 3> dimensions;
		std::array<int, 3> offset;
		std::array<int, 3> stride;
		
		int baseOffset;
		int strideX;
		int strideY;
		int strideZ;
		
		void calculateConstants() {
			baseOffset = offset.at(0) * dimensions.at(2) * dimensions.at(1)
						 + offset.at(1) * dimensions.at(2) 
					     + offset.at(2);
			
			strideZ = stride.at(0) * dimensions.at(2) * dimensions.at(1);
			strideY = stride.at(1) * dimensions.at(2);
			strideX = stride.at(2);
		}
		
	public:	
		ArrayView3D(const std::array<int, 3>& dimensions)
		: dimensions{dimensions}, offset{0, 0, 0}, stride{1, 1, 1} {
			calculateConstants();
		}
		
		ArrayView3D(const std::array<int, 3>& dimensions, const std::array<int, 3>& offset)
		: dimensions{dimensions}, offset{offset}, stride{1, 1, 1} {
			calculateConstants();
		}

		ArrayView3D(const std::array<int, 3>& dimensions, const std::array<int, 3>& offset, const std::array<int, 3>& stride)
		: dimensions{dimensions}, offset{offset}, stride{stride} {
			calculateConstants();
		}
		
		template <typename T>
		inline T* access(T* array, int z, int y, int x) {
			return &array[baseOffset + dotprod(z, y, x, strideZ, strideY, strideX)];
		}
	};

	template <typename T>
	inline T* access_direct(T* array, int X, int y, int x) {
		return &array[dotprod(y, x, X, 1)];
	}
}

#endif // RAW_ARRAY_VIEW_3D_HPP