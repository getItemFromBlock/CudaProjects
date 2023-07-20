#pragma once

#include "cuda_runtime.h"
#include "Maths/Maths.cuh"
#include "RayTracing/RayTracing.cuh"

namespace RayTracing
{
	class Texture;
	class FrameBuffer;
	enum class ChannelType : u8;
}

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
	void Copy(const void* source, void* dest, u64 size, CopyType kind);
	void CopyFrameBuffer(const RayTracing::FrameBuffer& source, u32* dest, CopyType kind);
	void CopyFrameBuffer(const u32* source, RayTracing::FrameBuffer& dest, CopyType kind);
	s32 GetMaxThreads(s32 deviceID);
	s32 GetDevicesCount();
	void UseDevice(s32 deviceID);
	void PrintDevicesName();
	bool LoadTexture(RayTracing::Texture& tex, const std::string& path);
	bool UnloadTexture(const RayTracing::Texture& tex);
	bool CreateFrameBuffer(RayTracing::FrameBuffer& tex, Maths::IVec2 res, RayTracing::ChannelType type);
	bool UnloadFrameBuffer(const RayTracing::FrameBuffer& tex);

	template<typename T>
	T* Allocate(u64 count)
	{
		T* result;
		result = static_cast<T*>(Allocate(count * sizeof(T)));
		return result;
	}
};