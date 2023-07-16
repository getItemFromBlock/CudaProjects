#include "RayTracing/Texture.cuh"

using namespace Maths;
using namespace RayTracing;

__host__ __device__ IVec2 Texture::GetResolution() const
{
	return resolution;
}

__device__ Vec4 Texture::Sample(Vec2 uv)
{
	float4 res = tex2D<float4>(device_tex, uv.x, uv.y);
	for (u8 i = 0; i < 4; ++i)
	{
		*(&res.x + i) = powf(*(&res.x + i), 1/2.3f);
	}
	return Vec4(res.x, res.y, res.z, res.w);
}
