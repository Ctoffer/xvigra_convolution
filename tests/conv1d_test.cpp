#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"

#ifdef VOID
#undef VOID
#endif

#include <string>

#include "conv1d_v2.hpp"

#include "xtensor/xtensor.hpp"

TEST_CASE("Input Layout: Channel x Width")
{
	SUBCASE("Even Input, Odd Kernel") 
	{
		xt::xtensor<int, 2> inputBlackAndWhite{{832, 924, 929, 919, 847, 551, 425, 876, 817, 285}};
		xt::xtensor<int, 3> kernelsBlackAndWhite{{{ 96, 182, 17}}};
		
		xt::xtensor<int, 2> inputColor{
			{575, 658, 181, 778, 205, 281, 101, 294, 997, 759},
			{133, 873, 656, 355, 756, 740, 373,99, 703, 285},
			{429, 136, 541, 347, 449, 913, 979, 568, 497, 519}
		};
		xt::xtensor<int, 3> kernelsColor{
			{{536, 116, 378}, { 79, 388, 582}, {878,  86, 754}},
			{{731, 941, 825}, {205, 571, 262}, {147, 149, 339}},
			{{786, 797, 736}, {426, 931, 725}, {248, 114, 392}}
		};
		
		SUBCASE("Default padding, dilation, stride") 
		{
			int padding = 0;
			int dilation = 1;
			int stride = 1;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White") 
			{
				xt::xtensor<int, 2> expectedResult{
					{263833, 273405, 270841, 251745, 188819, 145138, 214121, 237635}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{1980241, 1625445, 1737696, 2330741, 1955490, 1848346, 2225279, 1939365},
					{2153174, 2157954, 1852232, 2119196, 1706768, 1406257, 1887218, 2609134},
					{2788581, 2705488, 2421007, 2868413, 2340046, 1812711, 2310606, 2888579}
				};
				
				auto actualResult = convolution(inputColor, kernelsColor);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}
		
		SUBCASE("Custom padding Default dilation, stride")
		{
			int padding = 3;
			int dilation = 1;
			int stride = 1;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White")
			{
				xt::xtensor<int, 2> expectedResult{
					{ 0,14144, 167132, 263833, 273405, 270841, 251745, 188819, 145138, 214121, 237635, 130302,27360,0}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
			
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{0,618222, 1014552, 1980241, 1625445, 1737696, 2330741, 1955490, 1848346, 2225279, 1939365, 1269553,885021, 0},
					{0,654652, 1498619, 2153174, 2157954, 1852232, 2119196, 1706768, 1406257, 1887218, 2609134, 1900266,689547, 0},
					{0,687793, 1801529, 2788581, 2705488, 2421007, 2868413, 2340046, 1812711, 2310606, 2888579, 2135800,846696, 0}
				};
			
				auto actualResult = convolution(inputColor, kernelsColor);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}
		
		SUBCASE("Custom dilation Default padding, stride")
		{
			int padding = 0;
			int dilation = 2;
			int stride = 1;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White") 
			{
				xt::xtensor<int, 2> expectedResult{
					{263349, 265329, 250563, 203398, 172551, 217173}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
			
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{1873447, 2024193, 1972990, 1744975, 1965210, 1976152},
					{1655567, 2401673, 1550732, 1976540, 1999927, 1785923},
					{2306655, 3014155, 2202826, 2376588, 2572195, 2122466}
				};
			
				auto actualResult = convolution(inputColor, kernelsColor);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}
	
		SUBCASE("Custom stride Default padding, dilation")
		{
			int padding = 0;
			int dilation = 1;
			int stride = 4;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White") 
			{
				xt::xtensor<int, 2> expectedResult{
					{263833, 188819}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
			
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{1980241, 1955490},
					{2153174, 1706768},
					{2788581, 2340046}
				};
			
				auto actualResult = convolution(inputColor, kernelsColor);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}
		
		SUBCASE("Custom padding, dilation, stride")
		{
			int padding = 3;
			int dilation = 2;
			int stride = 4;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White") 
			{
				xt::xtensor<int, 2> expectedResult{
					{ 15708, 265329, 217173}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
			
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{ 859354, 2024193, 1976152},
					{ 817680, 2401673, 1785923},
					{1170525, 3014155, 2122466}
				};
			
				auto actualResult = convolution(inputColor, kernelsColor);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}

		SUBCASE("Mismatch in number of input channels")
		{
			
		}
		
		SUBCASE("Oversized Kernel no input padding")
		{
			
		}
		
		SUBCASE("Oversized Kernel and input padding")
		{
			
		}
	}	
}

TEST_CASE("Input Layout: Width x Channel")
{
	SUBCASE("Even Input, Odd Kernel") 
	{
		xt::xtensor<int, 2> inputBlackAndWhite{{832}, {924}, {929}, {919}, {847}, {551}, {425}, {876}, {817}, {285}};
		xt::xtensor<int, 3> kernelsBlackAndWhite{{{ 96, 182, 17}}};
		
		xt::xtensor<int, 2> inputColor{
			{575, 133, 429},
			{658, 873, 136},
			{181, 656, 541},
			{778, 355, 347},
			{205, 756, 449},
			{281, 740, 913},
			{101, 373, 979},
			{294,  99, 568},
			{997, 703, 497},
			{759, 285, 519}
		};
		xt::xtensor<int, 3> kernelsColor{
			{{536, 116, 378}, { 79, 388, 582}, {878,  86, 754}},
			{{731, 941, 825}, {205, 571, 262}, {147, 149, 339}},
			{{786, 797, 736}, {426, 931, 725}, {248, 114, 392}}
		};
		
		SUBCASE("Default padding, dilation, stride") 
		{
			int padding = 0;
			int dilation = 1;
			int stride = 1;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White") 
			{
				xt::xtensor<int, 2> expectedResult{
					{263833}, 
					{273405}, 
					{270841}, 
					{251745}, 
					{188819}, 
					{145138}, 
					{214121}, 
					{237635}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite, false);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{1980241, 2153174, 2788581},
					{1625445, 2157954, 2705488},
					{1737696, 1852232, 2421007},
					{2330741, 2119196, 2868413},
					{1955490, 1706768, 2340046},
					{1848346, 1406257, 1812711},
					{2225279, 1887218, 2310606},
					{1939365, 2609134, 2888579}
				};
				
				auto actualResult = convolution(inputColor, kernelsColor, false);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}
		
		SUBCASE("Custom padding Default dilation, stride")
		{
			int padding = 3;
			int dilation = 1;
			int stride = 1;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White")
			{
				xt::xtensor<int, 2> expectedResult{
					{0},
					{14144}, 
					{167132}, 
					{263833}, 
					{273405}, 
					{270841}, 
					{251745}, 
					{188819}, 
					{145138}, 
					{214121}, 
					{237635}, 
					{130302},
					{27360},
					{0}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite, false);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
			
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{      0,       0,       0}, 
					{ 618222,  654652,  687793}, 
					{1014552, 1498619, 1801529}, 
					{1980241, 2153174, 2788581},
					{1625445, 2157954, 2705488}, 
					{1737696, 1852232, 2421007}, 
					{2330741, 2119196, 2868413}, 
					{1955490, 1706768, 2340046},
					{1848346, 1406257, 1812711}, 
					{2225279, 1887218, 2310606}, 
					{1939365, 2609134, 2888579}, 
					{1269553, 1900266, 2135800},
					{ 885021,  689547,  846696},
					{      0,       0,       0}
				};
			
				auto actualResult = convolution(inputColor, kernelsColor, false);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}
		
		SUBCASE("Custom dilation Default padding, stride")
		{
			int padding = 0;
			int dilation = 2;
			int stride = 1;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White") 
			{
				xt::xtensor<int, 2> expectedResult{
					{263349}, 
					{265329}, 
					{250563}, 
					{203398}, 
					{172551}, 
					{217173}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite, false);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
			
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{1873447, 1655567, 2306655}, 
					{2024193, 2401673, 3014155},
					{1972990, 1550732, 2202826}, 
					{1744975, 1976540, 2376588}, 
					{1965210, 1999927, 2572195}, 
					{1976152, 1785923, 2122466}
				};
			
				auto actualResult = convolution(inputColor, kernelsColor, false);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}
	
		SUBCASE("Custom stride Default padding, dilation")
		{
			int padding = 0;
			int dilation = 1;
			int stride = 4;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White") 
			{
				xt::xtensor<int, 2> expectedResult{
					{263833}, 
					{188819}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite, false);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
			
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{1980241, 2153174, 2788581},
					{1955490, 1706768, 2340046}
				};
			
				auto actualResult = convolution(inputColor, kernelsColor, false);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}
		
		SUBCASE("Custom padding, dilation, stride")
		{
			int padding = 3;
			int dilation = 2;
			int stride = 4;
			Conv1D convolution(padding, dilation, stride);
			
			SUBCASE("Black'n'White") 
			{
				xt::xtensor<int, 2> expectedResult{
					{15708}, 
					{265329}, 
					{217173}
				};
				
				auto actualResult = convolution(inputBlackAndWhite, kernelsBlackAndWhite, false);
				
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
			
			SUBCASE("Color")
			{
				xt::xtensor<int, 2> expectedResult{
					{ 859354,  817680, 1170525}, 
					{2024193, 2401673, 3014155}, 
					{1976152, 1785923, 2122466}
				};
			
				auto actualResult = convolution(inputColor, kernelsColor, false);
				CHECK(expectedResult.shape() == actualResult.shape());
				CHECK(expectedResult == actualResult);
			}
		}

		SUBCASE("Mismatch in number of input channels")
		{
			
		}
		
		SUBCASE("Oversized Kernel no input padding")
		{
			
		}
		
		SUBCASE("Oversized Kernel and input padding")
		{

		}
	}
}
