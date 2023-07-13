#pragma once

#include "cuda_runtime.h"
#include "Maths/Maths.cuh"
#include "RayTracing/RayTracing.cuh"

namespace RayTracing
{
	class Texture;
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
	void Copy(const void* source, void* dest, u64 size, CopyType kind);
	s32 GetMaxThreads(s32 deviceID);
	s32 GetDevicesCount();
	void UseDevice(s32 deviceID);
	void PrintDevicesName();
	bool LoadTexture(RayTracing::Texture& tex, const std::string& path);
	bool UnloadTexture(RayTracing::Texture& tex);

	template<typename T>
	T* Allocate(u64 count)
	{
		T* result;
		result = static_cast<T*>(Allocate(count * sizeof(T)));
		return result;
	}
};