#include "RayTracing/ModelLoader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.hpp"

#include <vector>
#include <iostream>
#include <filesystem>
#include <unordered_map>

#include "CudaUtil.hpp"
#include "RayTracing/Texture.cuh"

using namespace Maths;
using namespace RayTracing;

void LoadTexture(const std::string& tex, std::vector<TextureAlias>& texNames, std::vector<Texture>& textures, u32& dest, const std::string& file)
{
    if (tex.empty()) return;
    std::filesystem::path texture_filename = tex;
    if (!std::filesystem::exists(texture_filename))
    {
        // Append base dir.
        texture_filename = std::filesystem::path(file).parent_path().append(tex);
        if (!std::filesystem::exists(texture_filename))
        {
            std::cerr << "Unable to find file: " << tex << std::endl;
            return;
        }
    }
    std::string correct_path = texture_filename.string();
    for (auto& str : texNames)
    {
        if (str.path == correct_path)
        {
            dest = str.tex;
            return;
        }
    }

    dest = static_cast<u32>(textures.size());
    textures.push_back(Texture());
    if (!CudaUtil::LoadTexture(textures.back(), correct_path)) return;

    texNames.push_back(TextureAlias());
    texNames.back().path = correct_path;
    texNames.back().tex = dest;
}

void LoadMaterials(std::vector<Material>& materials, std::vector<Texture>& textures, std::vector<tinyobj::material_t> matsIn, const std::string& file)
{
    std::vector<TextureAlias> texNames;

    for (u64 m = 0; m < matsIn.size(); ++m)
    {
        materials.push_back(Material());
        tinyobj::material_t* mp = &matsIn[m];
        materials.back().ambientColor = Vec3(mp->ambient[0], mp->ambient[1], mp->ambient[2]);
        materials.back().diffuseColor = Vec3(mp->diffuse[0], mp->diffuse[1], mp->diffuse[2]);
        materials.back().specularColor = Vec3(mp->specular[0], mp->specular[1], mp->specular[2]);

        LoadTexture(mp->diffuse_texname, texNames, textures, materials.back().diffuseTex, file);
        LoadTexture(mp->ambient_texname, texNames, textures, materials.back().ambientTex, file);
        LoadTexture(mp->specular_texname, texNames, textures, materials.back().specularTex, file);
    }
}

template <>
struct std::hash<IVec3>
{
    std::size_t operator()(const IVec3& k) const
    {

        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:

        return ((hash<s32>()(k.x)
            ^ (hash<s32>()(k.y) << 1)) >> 1)
            ^ (hash<s32>()(k.z) << 1);
    }
};

struct MeshData
{
    std::unordered_map<IVec3, u32> bufferedVertices;
    std::vector<Vertice> vertices;
    std::vector<u32> indices;
    Maths::Vec3 min = INFINITY;
    Maths::Vec3 max = -INFINITY;
    u32 matIndex = 0;
};

void LoadMeshes(std::vector<Mesh>& meshes, u64 matDelta, std::vector<tinyobj::shape_t> meshesIn, const tinyobj::attrib_t& attributes)
{
    std::vector<MeshData> meshesD;
    for (u64 m = 0; m < meshesIn.size(); ++m)
    {
        tinyobj::mesh_t* mp = &meshesIn[m].mesh;
        std::vector<s32> usedMats;
        s32 lastID = -1;
        u64 meshDelta = meshesD.size();
        u64 meshIndex = 0;
        for (u64 i = 0; i < mp->material_ids.size(); ++i)
        {
            if (mp->material_ids[i] != lastID)
            {
                u64 n;
                for (n = 0; n < usedMats.size(); ++n)
                {
                    if (usedMats[n] == mp->material_ids[i])
                    {
                        break;
                    }
                }
                meshIndex = n;
                if (n == usedMats.size())
                {
                    usedMats.push_back(mp->material_ids[i]);
                    meshesD.push_back(MeshData());
                    meshesD.back().matIndex = mp->material_ids[i];
                }
                lastID = mp->material_ids[i];
            }
            auto& data = meshesD[meshIndex + meshDelta];
            for (u8 j = 0; j < 3; ++j)
            {
                Maths::IVec3 vert = Maths::IVec3(mp->indices[i * 3 + j].vertex_index, mp->indices[i * 3 + j].normal_index, mp->indices[i * 3 + j].texcoord_index);
                auto result = data.bufferedVertices.find(vert);
                if (result != data.bufferedVertices.end())
                {
                    data.indices.push_back(result->second);
                }
                else
                {
                    data.indices.push_back(static_cast<u32>(data.vertices.size()));
                    data.vertices.push_back(Vertice());
                    data.vertices.back().pos = Maths::Vec3(attributes.vertices[vert.x * 3], attributes.vertices[vert.x * 3 + 1], attributes.vertices[vert.x * 3 + 2]);
                    data.vertices.back().normal = Maths::Vec3(attributes.normals[vert.y * 3], attributes.normals[vert.y * 3 + 1], attributes.normals[vert.y * 3 + 2]);
                    data.vertices.back().uv = Maths::Vec2(attributes.texcoords[vert.z * 2], attributes.texcoords[vert.z * 2 + 1]);
                    for (u8 h = 0; h < 3; ++h)
                    {
                        data.min[h] = Util::MinF(data.min[h], data.vertices.back().pos[h]);
                        data.max[h] = Util::MaxF(data.max[h], data.vertices.back().pos[h]);
                    }
                    data.bufferedVertices[vert] = data.indices.back();
                }
            }
        }
    }
    for (u64 i = 0; i < meshesD.size(); ++i)
    {
        if (meshesD[i].indices.empty()) continue;
        meshes.push_back(Mesh());
        auto& mesh = meshes.back();
        mesh.matIndex = static_cast<u32>(matDelta + meshesD[i].matIndex);
        mesh.verticeCount = static_cast<u32>(meshesD[i].vertices.size());
        mesh.indiceCount = static_cast<u32>(meshesD[i].indices.size());
        mesh.indices = CudaUtil::Allocate<u32>(mesh.indiceCount);
        mesh.sourceVertices = CudaUtil::Allocate<Vertice>(mesh.verticeCount);
        mesh.transformedVertices = CudaUtil::Allocate<Vertice>(mesh.verticeCount);
        CudaUtil::Copy(meshesD[i].indices.data(), mesh.indices, mesh.indiceCount * sizeof(u32), CudaUtil::CopyType::HToD);
        CudaUtil::Copy(meshesD[i].vertices.data(), mesh.sourceVertices, mesh.verticeCount * sizeof(Vertice), CudaUtil::CopyType::HToD);
        mesh.sourceBox.center = (meshesD[i].min + meshesD[i].max) / 2;
        mesh.sourceBox.radius = (meshesD[i].max - meshesD[i].min) / 2;
    }
}

bool RayTracing::ModelLoader::LoadModel(std::vector<Mesh>& meshes, std::vector<Material>& materials, std::vector<Texture>& textures, const std::string& path)
{
	tinyobj::attrib_t attributes = {};
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> mats;
	std::string warn;
	std::string err;
    const std::string parent = std::filesystem::path(path).parent_path().string();
	bool ret = tinyobj::LoadObj(&attributes, &shapes, &mats, &warn, &err, path.c_str(), parent.c_str());
	if (!warn.empty())
	{
		std::cout << "WARN: " << warn << std::endl;
	}
	if (!err.empty())
	{
		std::cerr << err << std::endl;
	}
	if (!ret)
	{
		std::cerr << "Failed to load " << path << std::endl;
		return false;
	}
	mats.push_back(tinyobj::material_t());
	u64 matdelta = materials.size();
    LoadMaterials(materials, textures, mats, path);
    LoadMeshes(meshes, matdelta, shapes, attributes);
	return true;
}