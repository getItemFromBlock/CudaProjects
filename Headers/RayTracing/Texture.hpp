#pragma once

#include "Maths/Maths.hpp"

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#else
#define CUDA_FUNCTION
#endif

namespace RayTracing
{
	class Texture
	{
	public:
		Texture() {}

		~Texture() {}

		void Load(const char* path);
		void Unload();

		u8* data = nullptr;
		Maths::IVec2 resolution;
		u8 mipmap = 0;

		CUDA_FUNCTION Maths::Vec4 Sample(Maths::Vec2 uv, bool linear = true);
		CUDA_FUNCTION Maths::Vec4 TexelFetch(Maths::IVec2 pos, u8 mipmap = 0);
	private:
	};

}