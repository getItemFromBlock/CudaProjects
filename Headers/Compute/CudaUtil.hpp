#pragma once

#include "cuda_runtime.h"
#include "Maths/Maths.cuh"
#include "RayTracing.cuh"

#include "string"

namespace Resources
{
	class Texture;
	class MipmappedTexture;
	class Cubemap;
	class FrameBuffer;
	enum class ChannelType : u8;
}

namespace Compute
{
	namespace CudaUtil
	{
		enum class CopyType : u8
		{
			HToH = 0,
			HToD = 1,
			DToH = 2,
			DToD = 3,
			Default = 4
		};

		void ResetDevice();
		s32 SelectDevice();
		void CheckError(cudaError_t val, const char* text = "checkError failed: %s");
		void SynchronizeDevice();
		void* Allocate(u64 size);
		void Free(void* ptr);
		void FreeArray(cudaArray_t ptr);
		void FreeArray(cudaMipmappedArray_t ptr);
		void Copy(const void* source, void* dest, u64 size, CopyType kind);
		void CopyFrameBuffer(const Resources::FrameBuffer& source, u32* dest, CopyType kind);
		void CopyFrameBuffer(const u32* source, Resources::FrameBuffer& dest, CopyType kind);
		s32 GetMaxThreads(s32 deviceID);
		s32 GetDevicesCount();
		void UseDevice(s32 deviceID);
		void PrintDevicesName();
		bool LoadMipmappedTexture(Resources::MipmappedTexture& tex, const std::string& path);
		bool LoadTexture(Resources::Texture& tex, const std::string& path);
		std::vector<f32*> GenerateTextureMipMaps(f32* source, Maths::IVec2 resolution, u32& maxLOD);
		bool UnloadTexture(const Resources::Texture& tex);
		bool LoadCubemap(Resources::Cubemap& tex, const std::string& path);
		bool UnloadCubemap(const Resources::Cubemap& tex);
		bool CreateFrameBuffer(Resources::FrameBuffer& tex, Maths::IVec2 res, Resources::ChannelType type);
		bool UnloadFrameBuffer(const Resources::FrameBuffer& tex);

		template<typename T>
		T* Allocate(u64 count)
		{
			T* result;
			result = static_cast<T*>(Allocate(count * sizeof(T)));
			return result;
		}
	};
}