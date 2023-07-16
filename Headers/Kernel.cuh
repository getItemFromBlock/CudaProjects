#pragma once

#include <vector>
#include <string>
#include "CudaUtil.hpp"
#include "Maths/Maths.cuh"
#include "RayTracing/Texture.cuh"
#include "RayTracing/FrameBuffer.cuh"
#include "RayTracing/Mesh.cuh"
#include "RayTracing/Material.hpp"
#include "RayTracing/Texture.cuh"

class Kernel
{
public:
	Kernel() {}
	~Kernel() {}
	void InitKernels(Maths::IVec2 res, s32 deviceID);
	void Resize(Maths::IVec2 newRes);
	void ClearKernels();
	void RunKernels(u32* img, f64 iTime);

	// Ray tracing specific functions start here
	void LoadMeshes(const std::vector<RayTracing::Mesh> meshes);
	void LoadTextures(const std::vector<RayTracing::Texture> textures);
	void LoadMaterials(const std::vector<RayTracing::Material> materials);
	void UpdateMeshVertices(RayTracing::Mesh* mesh, u32 index, const Maths::Mat4& mvp);
	void Synchronize();
	void RenderMeshes(u32* img, u32 meshCount, Maths::Vec3 pos, Maths::Vec3 front, Maths::Vec3 up);
	void UnloadMeshes(const std::vector<RayTracing::Mesh>& meshes);
	void UnloadTextures(const std::vector<RayTracing::Texture>& textures);
	void UnloadMaterials();

private:
	RayTracing::Mesh* device_meshes = nullptr;
	RayTracing::Texture* device_textures = nullptr;
	RayTracing::Material* device_materials = nullptr;
	RayTracing::FrameBuffer fb;
	s32 deviceID = 0;
};