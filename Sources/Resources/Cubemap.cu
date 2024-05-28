#include "Resources/Cubemap.cuh"

using namespace Maths;
using namespace Resources;

__host__ __device__ IVec2 Cubemap::GetResolution() const
{
	return resolution;
}

__device__ Vec4 Cubemap::Sample(Vec3 uv) const
{
	float4 res = texCubemap<float4>(device_tex, uv.x, uv.y, uv.z);
	return Vec4(res.x, res.y, res.z, res.w);
}
