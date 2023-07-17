#pragma once

#include "Maths/Maths.cuh"

namespace RayTracing
{
	class Material
	{
	public:
		Material() {}

		~Material() {}

		Maths::Vec3 diffuseColor = Maths::Vec3(1);
		Maths::Vec3 transmittanceColor;
		Maths::Vec3 emissionColor;
		f32 roughness = 1;
		f32 metallic = 0;
		f32 ior = 1;

		u32 diffuseTex = ~0;
		u32 metallicTex = ~0;
		u32 roughnessTex = ~0;
		u32 normalTex = ~0;
	private:

	};

}