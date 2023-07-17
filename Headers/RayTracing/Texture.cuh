#pragma once

#include "cuda_runtime.h"

#include "Maths/Maths.cuh"

namespace RayTracing
{
	class Texture
	{
	public:
		Texture() {}

		~Texture() {}

		__host__ __device__ Maths::IVec2 GetResolution() const;
		__device__ Maths::Vec4 Sample(Maths::Vec2 uv) const;

		cudaArray_t device_data = nullptr;
		cudaTextureObject_t device_tex = 0;
		Maths::IVec2 resolution;
	private:
	};
}