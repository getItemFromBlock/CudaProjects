#pragma once

#include <vector>
#include <string>
#include "curand_kernel.h"
#include "CudaUtil.hpp"
#include "Maths/Maths.cuh"
#include "RayTracing/Texture.cuh"
#include "RayTracing/FrameBuffer.cuh"
#include "RayTracing/Mesh.cuh"
#include "RayTracing/Material.hpp"
#include "RayTracing/Texture.cuh"

enum LaunchParams : u8
{
	NONE = 0,
	ADVANCED = 1,
	INVERTED_RB = 2,
};

class Kernel
{
public:
	Kernel() {}
	~Kernel() {}
	void InitKernels(Maths::IVec2 res, s32 deviceID);
	void Resize(Maths::IVec2 newRes);
	void ClearKernels();
	void RunFractalKernels(u32* img, f64 iTime);

	// Ray tracing specific functions start here
	void LoadMeshes(const std::vector<RayTracing::Mesh> meshes);
	void LoadTextures(const std::vector<RayTracing::Texture> textures);
	void LoadMaterials(const std::vector<RayTracing::Material> materials);
	void UpdateMeshVertices(RayTracing::Mesh* mesh, u32 index, const Maths::Vec3& pos, const Maths::Quat& rot, const Maths::Vec3& scale);
	void Synchronize();
	void RenderMeshes(u32* img, const u32 meshCount, const Maths::Vec3& pos, const Maths::Vec3& front, const Maths::Vec3& up, const f32 fov, const u32 quality, const LaunchParams params);
	void UnloadMeshes(const std::vector<RayTracing::Mesh>& meshes);
	void UnloadTextures(const std::vector<RayTracing::Texture>& textures);
	void UnloadMaterials();
	void SeedRNGBuffer();

private:
	RayTracing::Mesh* device_meshes = nullptr;
	RayTracing::Texture* device_textures = nullptr;
	RayTracing::Material* device_materials = nullptr;
	RayTracing::FrameBuffer surfaceFB;
	RayTracing::FrameBuffer mainFB;
	curandState* device_prngBuffer = nullptr;
	u64 rngBufferSize = 0;
	s32 deviceID = 0;

	void LaunchRTXKernels(const u32 meshCount, const Maths::Vec3& pos, const Maths::Vec3& front, const Maths::Vec3& up, const f32 fov, const u32 quality, const LaunchParams advanced);
};