#pragma once

#include <vector>

#include "Mesh.cuh"
#include "Cubemap.cuh"
#include "Material.hpp"

namespace Resources
{
	class Texture;

	struct TextureAlias
	{
		std::string path;
		u32 tex = ~0;
	};

	struct VertexData
	{
		s32 pos;
		s32 norm;
		s32 uv;
		s32 tang;
		s32 cotang;

		bool operator==(const VertexData& other) const
		{
			return pos == other.pos && norm == other.norm && uv == other.uv && tang == other.tang && cotang == other.cotang;
		}
	};

	struct TriangleData
	{
		VertexData data[3];
	};

	namespace ModelLoader
	{
		bool LoadModel(std::vector<Mesh>& meshes, std::vector<Material>& materials, std::vector<Texture>& textures, const std::string& path);
		bool LoadCubemap(std::vector<Cubemap>& cubemaps, const std::string& path);
	}
}