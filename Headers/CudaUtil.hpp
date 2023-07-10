#pragma once

#include "cuda_runtime.h"
#include "Types.hpp"

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

	template<typename T>
	T* Allocate(u64 count)
	{
		T* result;
		result = static_cast<T*>(Allocate(count * sizeof(T)));
		return result;
	}
};