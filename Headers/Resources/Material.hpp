#pragma once

#include "Maths/Maths.cuh"

namespace Resources
{
	class Material
	{
	public:
		Material() {}

		~Material() {}

		Maths::Vec3 diffuseColor = Maths::Vec3(1);
		f32 roughness = 1;
		Maths::Vec3 transmittanceColor;
		f32 metallic = 0;
		Maths::Vec3 emissionColor;
		f32 ior = 1;

		u32 diffuseTex = ~0;
		u32 metallicTex = ~0;
		u32 roughnessTex = ~0;
		u32 normalTex = ~0;
	private:

	};

}