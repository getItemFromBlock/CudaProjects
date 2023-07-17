#pragma once

#include "Texture.cuh"

namespace RayTracing
{
	class CubeMap : public Texture
	{
	public:
		CubeMap() {}

		~CubeMap() {}

		__device__ Maths::Vec4 Sample(Maths::Vec3 uv) const;
	private:
	};
}