#include "RayTracing/FrameBuffer.cuh"

using namespace RayTracing;
using namespace Maths;

__host__ __device__ IVec2 FrameBuffer::GetResolution() const
{
	return resolution;
}

__device__ u32 FrameBuffer::Sample(IVec2 pos)
{
	uchar4 res;
	surf2Dread(&res, device_surf, pos.x * sizeof(uchar4), pos.y);
	return res.w << 24 | res.z << 16 | res.y << 8 | res.x;
}

__device__ Vec4 FrameBuffer::SampleVec(IVec2 pos)
{
	uchar4 res;
	surf2Dread(&res, device_surf, pos.x * sizeof(uchar4), pos.y);
	Maths::Vec4 output = Maths::Vec4(res.x, res.y, res.z, res.w) / 255;
	return output;
}

__device__ void FrameBuffer::Write(Maths::IVec2 pos, u32 color)
{
	uchar4 out;
	out.x = color & 0xff;
	out.y = (color >> 8) & 0xff;
	out.z = (color >> 16) & 0xff;
	out.w = (color >> 24) & 0xff;
	surf2Dwrite(out, device_surf, pos.x * sizeof(uchar4), pos.y);
}

__device__ void FrameBuffer::Write(IVec2 pos, Vec4 color)
{
	uchar4 out;
	color = Util::Clamp(color) * 255.0f + 0.5f;
	out.x = static_cast<u8>(color.z);
	out.y = static_cast<u8>(color.y);
	out.z = static_cast<u8>(color.x);
	out.w = static_cast<u8>(color.w);
	surf2Dwrite(out, device_surf, pos.x * sizeof(uchar4), pos.y);
}