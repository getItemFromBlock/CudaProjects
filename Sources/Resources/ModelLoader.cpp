#include "Resources/ModelLoader.hpp"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.hpp"

#include <vector>
#include <iostream>
#include <filesystem>
#include <unordered_map>

#include "Compute/CudaUtil.hpp"
#include "Resources/Texture.cuh"

using namespace Maths;
using namespace Resources;
using namespace Compute;

void TryLoadTexture(const std::string& tex, std::vector<TextureAlias>& texNames, std::vector<Texture>& textures, u32& dest, const std::string& file)
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
    Texture t;
    if (!CudaUtil::LoadTexture(t, correct_path)) return;
    textures.push_back(t);
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
        materials.back().diffuseColor = Vec3(mp->diffuse[0], mp->diffuse[1], mp->diffuse[2]);
        materials.back().emissionColor = Vec3(mp->emission[0], mp->emission[1], mp->emission[2]);
        materials.back().transmittanceColor = Vec3(mp->transmittance[0], mp->transmittance[1], mp->transmittance[2]);
        materials.back().ior = mp->ior;
        materials.back().roughness = mp->roughness;
        materials.back().metallic = mp->metallic;

        TryLoadTexture(mp->diffuse_texname, texNames, textures, materials.back().diffuseTex, file);
        TryLoadTexture(mp->metallic_texname, texNames, textures, materials.back().metallicTex, file);
        TryLoadTexture(mp->roughness_texname, texNames, textures, materials.back().roughnessTex, file);
        TryLoadTexture(mp->bump_texname, texNames, textures, materials.back().normalTex, file);
    }
}

template <>
struct std::hash<VertexData>
{
    std::size_t operator()(const VertexData& k) const
    {
        return ((((((hash<s32>()(k.pos) ^ (hash<s32>()(k.norm) << 1)) >> 1) ^ (hash<s32>()(k.uv) << 1)) << 1) ^ (hash<s32>()(k.tang) << 1)) >> 1) ^ (hash<s32>()(k.cotang) << 1);
    }
};

template <>
struct std::hash<Vec3>
{
    std::size_t operator()(const Vec3& k) const
    {
        return ((hash<f32>()(k.x)
            ^ (hash<f32>()(k.y) << 1)) >> 1)
            ^ (hash<f32>()(k.z) << 1);
    }
};

struct MeshData
{
    std::unordered_map<VertexData, u32> bufferedVertices;
    std::unordered_map<Vec3, u32> bufferedTangent;
    std::unordered_map<Vec3, u32> bufferedCotangent;
    std::vector<Vec3> tangents;
    std::vector<Vec3> cotangents;
    std::vector<Vertice> vertices;
    std::vector<u32> indices;
    Maths::Vec3 min = INFINITY;
    Maths::Vec3 max = -INFINITY;
    u32 matIndex = 0;
};

Vec3 GetVec3(s32 index, const std::vector<tinyobj::real_t>& vals)
{
    if (index < 0) return Vec3();
    return Vec3(vals[index * 3llu], vals[index * 3llu + 1], vals[index * 3llu + 2]);
}

Vec2 GetVec2(s32 index, const std::vector<tinyobj::real_t>& vals)
{
    if (index < 0) return Vec2();
    return Vec2(vals[index * 2llu], vals[index * 2llu + 1]);
}

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
            TriangleData tr;
            for (u8 j = 0; j < 3; ++j)
            {
                tr.data[j].pos = mp->indices[i * 3 + j].vertex_index;
                tr.data[j].norm = mp->indices[i * 3 + j].normal_index;
                tr.data[j].uv = mp->indices[i * 3 + j].texcoord_index;
            }
            Vec3 tangent;
            Vec3 bitangent;
            Vec3 edge1 = GetVec3(tr.data[1].pos, attributes.vertices) - GetVec3(tr.data[0].pos, attributes.vertices);
            Vec3 edge2 = GetVec3(tr.data[2].pos, attributes.vertices) - GetVec3(tr.data[0].pos, attributes.vertices);
            Vec2 deltaUV1 = GetVec2(tr.data[1].uv, attributes.texcoords) - GetVec2(tr.data[0].uv, attributes.texcoords);
            Vec2 deltaUV2 = GetVec2(tr.data[2].uv, attributes.texcoords) - GetVec2(tr.data[0].uv, attributes.texcoords);
            f32 f = 1.0f / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
            tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
            tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
            tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
            tangent = tangent.Normalize();
            bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
            bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
            bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
            bitangent = -bitangent.Normalize();
            s32 tIndex = -1;
            s32 cIndex = -1;
            const auto& result = data.bufferedTangent.find(tangent);
            if (result != data.bufferedTangent.end())
            {
                tIndex = result->second;
            }
            else
            {
                data.bufferedTangent[tangent] = static_cast<s32>(data.tangents.size());
                tIndex = static_cast<s32>(data.tangents.size());
                data.tangents.push_back(tangent);
            }
            const auto& result2 = data.bufferedCotangent.find(bitangent);
            if (result2 != data.bufferedCotangent.end())
            {
                cIndex = result2->second;
            }
            else
            {
                data.bufferedCotangent[tangent] = static_cast<s32>(data.cotangents.size());
                cIndex = static_cast<s32>(data.cotangents.size());
                data.cotangents.push_back(bitangent);
            }
            for (u8 j = 0; j < 3; ++j)
            {
                tr.data[j].tang = tIndex;
                tr.data[j].cotang = cIndex;
                auto result = data.bufferedVertices.find(tr.data[j]);
                if (result != data.bufferedVertices.end())
                {
                    data.indices.push_back(result->second);
                }
                else
                {
                    data.indices.push_back(static_cast<u32>(data.vertices.size()));
                    data.vertices.push_back(Vertice());
                    data.vertices.back().pos = GetVec3(tr.data[j].pos, attributes.vertices);
                    data.vertices.back().uv = GetVec2(tr.data[j].uv, attributes.texcoords);
                    if (tr.data[j].norm >= 0)
                    {
                        data.vertices.back().normal = GetVec3(tr.data[j].norm, attributes.normals);
                    }
                    else
                    {
                        data.vertices.back().normal = Vec3(0, 0, 1);
                    }
                    data.vertices.back().tangent = data.tangents[tr.data[j].tang];
                    data.vertices.back().cotangent = data.cotangents[tr.data[j].cotang];
                    for (u8 h = 0; h < 3; ++h)
                    {
                        data.min[h] = Util::MinF(data.min[h], data.vertices.back().pos[h]);
                        data.max[h] = Util::MaxF(data.max[h], data.vertices.back().pos[h]);
                    }
                    data.bufferedVertices[tr.data[j]] = data.indices.back();
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

bool ModelLoader::LoadModel(std::vector<Mesh>& meshes, std::vector<Material>& materials, std::vector<Texture>& textures, const std::string& path)
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

bool ModelLoader::LoadCubemap(std::vector<Cubemap>& cubemaps, const std::string& path)
{
    Cubemap c;
    if (!CudaUtil::LoadCubemap(c, path)) return false;
    cubemaps.push_back(c);
    return true;
}
