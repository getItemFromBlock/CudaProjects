#pragma once

#include <vector>

#include "Mesh.cuh"
#include "Material.hpp"

namespace RayTracing
{
	class Texture;

	struct TextureAlias
	{
		std::string path;
		u32 tex = ~0;
	};

	namespace ModelLoader
	{
		bool LoadModel(std::vector<Mesh>& meshes, std::vector<Material>& materials, std::vector<Texture>& textures, const std::string& path);
	}
}