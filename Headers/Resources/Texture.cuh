#pragma once

#ifdef __INTELLISENSE__
#define __CUDACC__
#endif
#include <cuda_runtime.h>

#include "Maths/Maths.cuh"

namespace Resources
{
	class Texture
	{
	public:
		Texture() {}

		~Texture() {}

		__host__ __device__ Maths::IVec2 GetResolution() const;
		__device__ Maths::Vec4 Sample(Maths::Vec2 uv) const;
		__device__ Maths::Vec4 Sample(Maths::Vec2 uv, Maths::Vec2 dX, Maths::Vec2 dY) const;
		static __host__ u32 GetMaxLOD(Maths::IVec2 res);

		cudaArray_t device_data = nullptr;
		cudaTextureObject_t device_tex = 0;
		Maths::IVec2 resolution;
		u32 maxLOD = 0;
	private:
	};
}