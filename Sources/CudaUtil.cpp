#include "CudaUtil.hpp"
#include <iostream>
#include <assert.h>
#include <vector>

void CudaUtil::ResetDevice()
{
	CheckError(cudaDeviceReset(), "cudaDeviceReset failed: %s");
}

s32 CudaUtil::SelectDevice()
{
    s32 count = 0;
    CheckError(cudaGetDeviceCount(&count));
    s32 selected = 0;
    std::vector<std::string> names;
    names.resize(count);
    s32 max = 0;
    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp props = { 0 };
        cudaGetDeviceProperties(&props, i);
        names[i] = props.name;
        int score = props.maxThreadsPerBlock * props.major * 10 * props.minor;
        if (score > max)
        {
            max = score;
            selected = i;
        }
    }
    for (s32 i = 0; i < count; ++i)
    {
        std::cout << "GPU id " << i << ": " << names[i];
        if (selected == i) std::cout << " (selected)";
        std::cout << std::endl;
    }
    CheckError(cudaSetDevice(selected));
    return selected;
}

void CudaUtil::UseDevice(s32 id)
{
    CheckError(cudaSetDevice(id));
}

void CudaUtil::PrintDevicesName()
{
    s32 count = GetDevicesCount();
    for (int i = 0; i < count; ++i)
    {
        cudaDeviceProp props = { 0 };
        cudaGetDeviceProperties(&props, i);
        int score = props.maxThreadsPerBlock * props.major * 10 * props.minor;
        std::cout << "GPU id " << i << ": " << props.name << " has a score of " << score << std::endl;
    }
}

void CudaUtil::CheckError(cudaError_t val, const char* text)
{
	if (val != cudaSuccess)
	{
		std::printf(text, cudaGetErrorString(val));
	}
}

void CudaUtil::SynchronizeDevice()
{
	CheckError(cudaDeviceSynchronize(), "cudaDeviceSynchronize failed: %s");
}

void* CudaUtil::Allocate(u64 size)
{
	void* res = nullptr;
	CheckError(cudaMalloc(&res, size), "cudaMalloc failed: %s");
	return res;
}

void CudaUtil::Free(void* ptr)
{
	CheckError(cudaFree(ptr), "cudaFree failed: %s");
}

void CudaUtil::Copy(const void* source, void* dest, u64 size, CopyType kind)
{
	CheckError(cudaMemcpy(dest, source, size, static_cast<cudaMemcpyKind>(kind)), "cudaMemcpy failed: %s");
}

s32 CudaUtil::GetMaxThreads(s32 deviceID)
{
    cudaDeviceProp props = { 0 };
    cudaGetDeviceProperties(&props, deviceID);
    return props.maxThreadsPerBlock;
}

s32 CudaUtil::GetDevicesCount()
{
    s32 count = 0;
    CheckError(cudaGetDeviceCount(&count));
    return count;
}
