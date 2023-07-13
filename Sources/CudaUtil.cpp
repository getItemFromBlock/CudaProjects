#include "CudaUtil.hpp"

#include <iostream>
#include <assert.h>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "RayTracing/Texture.cuh"

using namespace Maths;
using namespace RayTracing;

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
        int score = props.maxThreadsPerBlock * (props.major * 10 + props.minor);
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

bool CudaUtil::LoadTexture(Texture& tex, const std::string& path)
{
    s32 comp;
    f32* data = stbi_loadf(path.c_str(), &tex.resolution.x, &tex.resolution.y, &comp, 4);
    if (!data)
    {
        std::cerr << "Unable to load texture: " << path << " : " << stbi_failure_reason() << std::endl;
    }
    if (tex.resolution.x <= 0 || tex.resolution.y <= 0 || !data) return false;
    const s32 fsize = sizeof(f32) * 8;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(fsize, fsize, fsize, fsize, cudaChannelFormatKindFloat);
    CheckError(cudaMallocArray(&tex.device_data, &channelDesc, tex.resolution.x, tex.resolution.y));
    const size_t spitch = tex.resolution.x * sizeof(Vec4);
    // Copy data located at address h_data in host memory to device memory
    CheckError(cudaMemcpy2DToArray(tex.device_data, 0, 0, data, spitch, tex.resolution.x, tex.resolution.y, cudaMemcpyHostToDevice));
    stbi_image_free(data);
    // Dont need this anymore, and cudaMemcpy is not async
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tex.device_data;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 1;

    // Create texture object
    CheckError(cudaCreateTextureObject(&tex.device_tex, &resDesc, &texDesc, NULL));
    return true;
}

bool CudaUtil::UnloadTexture(RayTracing::Texture& tex)
{
    if (tex.resolution.x <= 0 || tex.resolution.y <= 0) return false;
    CheckError(cudaDestroyTextureObject(tex.device_tex));
    CheckError(cudaFreeArray(tex.device_data));
    return true;
}
