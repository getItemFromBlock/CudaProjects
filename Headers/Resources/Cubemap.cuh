#pragma once

#include "Texture.cuh"

namespace Resources
{
	class Cubemap : public Texture
	{
	public:
		Cubemap() {}

		~Cubemap() {}

		__host__ __device__ Maths::IVec2 GetResolution() const;
		__device__ Maths::Vec4 Sample(Maths::Vec3 uv) const;

		cudaArray_t device_data = nullptr;
		cudaTextureObject_t device_tex = 0;
		Maths::IVec2 resolution;
	private:
		// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html?highlight=cube#cubemap-textures
	};
}