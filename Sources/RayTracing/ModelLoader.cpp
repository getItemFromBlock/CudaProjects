#include "RayTracing/ModelLoader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.hpp"

#include <vector>
#include <iostream>
#include <filesystem>

using namespace RayTracing;

void LoadMaterials(std::vector<Material>& materials, std::vector<Texture>& textures, std::vector<tinyobj::material_t> matsIn, const std::string& file)
{
    std::vector<std::string> texNames;

    for (u64 m = 0; m < matsIn.size(); ++m)
    {
        materials.push_back(Material());
        tinyobj::material_t* mp = &matsIn[m];
        materials.back().ambientColor = Maths::Vec3(mp->ambient[0], mp->ambient[1], mp->ambient[2]);
        materials.back().diffuseColor = Maths::Vec3(mp->diffuse[0], mp->diffuse[1], mp->diffuse[2]);
        materials.back().specularColor = Maths::Vec3(mp->specular[0], mp->specular[1], mp->specular[2]);

        if (mp->diffuse_texname.empty()) continue;
        bool found = false;
        for (auto& str : texNames)
        {
            if (str == mp->diffuse_texname)
            {
                found = true;
                break;
            }
        }
        if (found) continue;

        s32 w, h, comp;
        std::filesystem::path texture_filename = mp->diffuse_texname;
        if (!std::filesystem::exists(texture_filename))
        {
            // Append base dir.
            texture_filename = std::filesystem::path(file).parent_path().append(mp->diffuse_texname);
            if (!std::filesystem::exists(texture_filename))
            {
                std::cerr << "Unable to find file: " << mp->diffuse_texname << std::endl;
                continue;
            }
        }

        // TODO

        texNames.push_back(mp->diffuse_texname);
    }
}

bool RayTracing::ModelLoader::LoadModel(std::vector<Mesh>& meshes, std::vector<Material>& materials, std::vector<Texture>& textures, const char* path)
{
	tinyobj::attrib_t attributes = {};
	std::vector<tinyobj::shape_t> shapes;
	std::vector<tinyobj::material_t> mats;
	std::string warn;
	std::string err;
	bool ret = tinyobj::LoadObj(&attributes, &shapes, &mats, &warn, &err, path);
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
	u64 meshDelta = meshes.size();

	return true;
}