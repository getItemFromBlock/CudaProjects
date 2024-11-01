#include "Resources/Texture.cuh"

using namespace Maths;
using namespace Resources;

__host__ __device__ IVec2 Texture::GetResolution() const
{
	return resolution;
}

__device__ Vec4 Texture::Sample(Vec2 uv) const
{
	float4 res = tex2D<float4>(device_tex, uv.x, uv.y);
	return Vec4(res.x, res.y, res.z, res.w);
}

__device__ Vec4 Texture::Sample(Vec2 uv, Vec2 dX, Vec2 dY) const
{
	float2 a = {dX.x, dX.y};
	float2 b = {dY.x, dY.y};
	float4 res = tex2DGrad<float4>(device_tex, uv.x, uv.y, a, b);
	return Vec4(res.x, res.y, res.z, res.w);
}


__host__ u32 Texture::GetMaxLOD(Maths::IVec2 res)
{
	return 1 + (u32)(floor(log2(max(res.x, res.y))));
}