#pragma once

#include "Texture.cuh"

namespace RayTracing
{
	class CubeMap : public Texture
	{
	public:
		CubeMap() {}

		~CubeMap() {}

		__device__ Maths::Vec4 Sample(Maths::Vec3 uv) const;
	private:
		// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cube#cubemap-textures
	};
}