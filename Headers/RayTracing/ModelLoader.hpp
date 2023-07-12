#pragma once

#include <vector>

#include "Mesh.hpp"
#include "Material.hpp"

namespace RayTracing::ModelLoader
{
	bool LoadModel(std::vector<Mesh>& meshes, std::vector<Material>& materials, std::vector<Texture>& textures, const char* path);
}