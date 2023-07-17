#include "RayTracing/CubeMap.cuh"

using namespace Maths;
using namespace RayTracing;

__device__ Vec4 CubeMap::Sample(Vec3 uv) const
{
	float4 res = texCubemap<float4>(device_tex, uv.x, uv.y, uv.z);
	return Vec4(res.x, res.y, res.z, res.w);
}
