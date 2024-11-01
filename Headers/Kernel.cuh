#pragma once

#include <vector>
#include <string>
#include "curand_kernel.h"
#include "Compute/CudaUtil.hpp"
#include "Maths/Maths.cuh"
#include "Resources/Texture.cuh"
#include "Resources/Cubemap.cuh"
#include "Resources/FrameBuffer.cuh"
#include "Resources/Mesh.cuh"
#include "Resources/Material.hpp"

enum LaunchParams : u8
{
	NONE =			0x0,
	ADVANCED =		0x1,
	INVERTED_RB =	0x2,
	BOXDEBUG =		0x4,
	DENOISE =		0x8,
	CLEAR =			0x10,
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
	const Resources::FrameBuffer& GetMainFrameBuffer();

	// Ray tracing specific functions start here
	void LoadMeshes(const std::vector<Resources::Mesh> meshes);
	void LoadTextures(const std::vector<Resources::Texture> textures);
	void LoadCubemaps(const std::vector<Resources::Cubemap> cubemaps);
	void LoadMaterials(const std::vector<Resources::Material> materials);
	void UpdateMeshVertices(Resources::Mesh* mesh, u32 index, const Maths::Vec3& pos, const Maths::Quat& rot, const Maths::Vec3& scale);
	void Synchronize();
	void RenderMeshes(u32* img, const u32 meshCount, const Maths::Vec3& pos, const Maths::Vec3& front, const Maths::Vec3& up, const f32 fov, const u32 quality, const f32 strength, const LaunchParams params);
	void UnloadMeshes(const std::vector<Resources::Mesh>& meshes);
	void UnloadTextures(const std::vector<Resources::Texture>& textures);
	void UnloadCubemaps(const std::vector<Resources::Cubemap>& cubemaps);
	void UnloadMaterials();
	void SeedRNGBuffer();

private:
	Resources::Mesh* device_meshes = nullptr;
	Resources::Texture* device_textures = nullptr;
	Resources::Cubemap* device_cubemaps = nullptr;
	Resources::Material* device_materials = nullptr;
	Resources::FrameBuffer surfaceFB;
	Resources::FrameBuffer mainFB;
	curandState* device_prngBuffer = nullptr;
	u64 rngBufferSize = 0;
	s32 deviceID = 0;

	void LaunchRTXKernels(const u32 meshCount, const Maths::Vec3& pos, const Maths::Vec3& front, const Maths::Vec3& up, const f32 fov, const u32 quality, const f32 strength, const LaunchParams advanced);
};