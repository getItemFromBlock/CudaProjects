#pragma once

#include "cuda_runtime.h"

#include "Maths/Maths.cuh"

namespace RayTracing
{
	class FrameBuffer
	{
	public:
		FrameBuffer() {}

		~FrameBuffer() {}

		__host__ __device__ Maths::IVec2 GetResolution() const;
		__device__ u32 Sample(Maths::IVec2 pos);
		__device__ Maths::Vec4 SampleVec(Maths::IVec2 pos);
		__device__ void Write(Maths::IVec2 pos, u32 color);
		__device__ void Write(Maths::IVec2 pos, Maths::Vec4 color);

		cudaArray_t device_data = nullptr;
		cudaSurfaceObject_t device_surf = 0;
		Maths::IVec2 resolution;
	private:
	};
}