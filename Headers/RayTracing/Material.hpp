#pragma once

#include "Maths/Maths.hpp"
#include "Texture.hpp"

namespace RayTracing
{
	class Material
	{
	public:
		Material() {}

		~Material() {}

		Maths::Vec3 ambientColor;
		Maths::Vec3 diffuseColor;
		Maths::Vec3 specularColor;
		Maths::Vec3 transmittanceColor;
		Maths::Vec3 emissionColor;
		f32 shininess = 1;
		f32 ior = 1;
		f32 transparency = 0;

		Texture* ambientTex = nullptr;
		Texture* diffuseTex = nullptr;
		Texture* specularTex = nullptr;
	private:

	};

}