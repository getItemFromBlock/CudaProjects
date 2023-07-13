#include "RayTracing/FrameBuffer.cuh"

using namespace RayTracing;
using namespace Maths;

__host__ __device__ IVec2 FrameBuffer::GetResolution() const
{
	return resolution;
}

__device__ Vec4 FrameBuffer::Sample(IVec2 pos)
{

	float4 res;
	surf2Dread(&res, device_surf, pos.x * sizeof(float4), pos.y);
	return Vec4(res.x, res.y, res.z, res.w);
}

__device__ void FrameBuffer::Write(Maths::IVec2 pos, Maths::Vec4 color)
{
	float4 out;
	out.x = color.x;
	out.y = color.y;
	out.z = color.z;
	out.w = color.w;
	surf2Dwrite(out, device_surf, pos.x * sizeof(float4), pos.y);
	// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#surface-object-api
}
