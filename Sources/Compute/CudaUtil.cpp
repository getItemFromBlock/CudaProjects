#include "Compute/CudaUtil.hpp"

#include <iostream>
#include <assert.h>
#include <vector>
#include <fstream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "stb_image_write.h"

#include "Resources/Texture.cuh"
#include "Resources/MipmappedTexture.cuh"
#include "Resources/Cubemap.cuh"
#include "Resources/FrameBuffer.cuh"

using namespace Maths;
using namespace Resources;
using namespace Compute;

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
        std::string err = cudaGetErrorString(val);
		std::printf(text, cudaGetErrorString(val));
        assert(0);
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

void CudaUtil::FreeArray(cudaArray_t ptr)
{
    CheckError(cudaFreeArray(ptr), "cudaFreeArray failed: %s");
}

void CudaUtil::FreeArray(cudaMipmappedArray_t ptr)
{
    CheckError(cudaFreeMipmappedArray(ptr), "cudaFreeMipmappedArray failed: %s");
}

void CudaUtil::Copy(const void* source, void* dest, u64 size, CopyType kind)
{
	CheckError(cudaMemcpy(dest, source, size, static_cast<cudaMemcpyKind>(kind)), "cudaMemcpy failed: %s");
}

void CudaUtil::CopyFrameBuffer(const FrameBuffer& source, u32* dest, CopyType kind)
{
    const u64 spitch = sizeof(u8) * 4 * source.resolution.x;
    CheckError(cudaMemcpy2DFromArray(dest, spitch, source.device_data, 0, 0, spitch, source.resolution.y, static_cast<cudaMemcpyKind>(kind)));
}

void CudaUtil::CopyFrameBuffer(const u32* source, FrameBuffer& dest, CopyType kind)
{
    const u64 spitch = sizeof(u8) * 4 * dest.resolution.x;
    CheckError(cudaMemcpy2DToArray(dest.device_data, 0, 0, source, spitch, spitch, dest.resolution.y, static_cast<cudaMemcpyKind>(kind)));
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

bool pow2Check(u32 value)
{
    return (value & (value - 1));
}

bool CudaUtil::LoadMipmappedTexture(MipmappedTexture& tex, const std::string& path)
{
    s32 comp;
    f32* data = stbi_loadf(path.c_str(), &tex.resolution.x, &tex.resolution.y, &comp, 4);
    if (!data)
    {
        std::cerr << "Unable to load texture: " << path << " - " << stbi_failure_reason() << std::endl;
        return false;
    }
    if (tex.resolution.x <= 0 || tex.resolution.y <= 0) return false;
    if (pow2Check(tex.resolution.x) || pow2Check(tex.resolution.y))
    {
        std::cerr << "Unable to load texture: " << path << " - resolution must be a power of two (current: "
            << tex.resolution.x << ", " << tex.resolution.y << ")" << std::endl;
        return false;
    }
    // manually applying gamma correction
    for (u64 i = 0; i < 4llu * tex.resolution.x * tex.resolution.y; ++i)
    {
        data[i] = powf(data[i], 1 / 2.3f);
    }
    u32 maxLOD = 1;
    auto textureData = GenerateTextureMipMaps(data, tex.resolution, maxLOD);
    const s32 fsize = sizeof(f32) * 8;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(fsize, fsize, fsize, fsize, cudaChannelFormatKindFloat);
    cudaExtent extent = make_cudaExtent(tex.resolution.x, tex.resolution.y, 0);

    CheckError(cudaMallocMipmappedArray(&tex.device_data, &channelDesc, extent, maxLOD));
    for (u32 i = 0; i < maxLOD; i++)
    {
        cudaArray_t d_levelArray;

        CheckError(cudaGetMipmappedArrayLevel(&d_levelArray, tex.device_data, i));

        f32* img = textureData[i];
        IVec2 res = IVec2(tex.resolution.x >> i, tex.resolution.y >> i);
        u64 sizeElements = res.x * res.y;
        u64 sizeBytes = sizeElements * sizeof(Vec4);

        cudaMemcpy3DParms params = { 0 };

        params.srcPtr = make_cudaPitchedPtr(img, res.x * sizeof(Vec4), res.x, res.y);
        params.dstArray = d_levelArray;
        params.extent = make_cudaExtent(res.x, res.y, 0);
        params.kind = cudaMemcpyHostToDevice;

        CheckError(cudaMemcpy3D(&params));
        stbi_image_free(img);
    }
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeMipmappedArray;
    resDesc.res.mipmap.mipmap = tex.device_data;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeWrap;
    texDesc.addressMode[1] = cudaAddressModeWrap;
    texDesc.mipmapFilterMode = cudaFilterModePoint;
    texDesc.minMipmapLevelClamp = 0;
    texDesc.maxMipmapLevelClamp = float(maxLOD);
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.sRGB = 0;
    texDesc.normalizedCoords = 1;

    // Create texture object
    CheckError(cudaCreateTextureObject(&tex.device_tex, &resDesc, &texDesc, NULL));
    return true;
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
    for (u64 i = 0; i < 4llu * tex.resolution.x * tex.resolution.y; ++i)
    {
        data[i] = powf(data[i], 1 / 2.3f);
    }
    const s32 fsize = sizeof(f32) * 8;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(fsize, fsize, fsize, fsize, cudaChannelFormatKindFloat);
    CheckError(cudaMallocArray(&tex.device_data, &channelDesc, tex.resolution.x, tex.resolution.y));
    const size_t spitch = tex.resolution.x * sizeof(Vec4);
    CheckError(cudaMemcpy2DToArray(tex.device_data, 0, 0, data, spitch, spitch, tex.resolution.y, cudaMemcpyHostToDevice));
    // Dont need this anymore, and cudaMemcpy is not async
    stbi_image_free(data);
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

bool CudaUtil::SaveFrameBuffer(const Resources::FrameBuffer& fb, std::string path)
{
    if (fb.resolution.x <= 0 || fb.resolution.y <= 0) return false;
    u64 size = fb.type == ChannelType::U8 ? 4 : 16;
    u64 len = fb.resolution.x * fb.resolution.y * size;
    u64 spitch = fb.resolution.x * size;
    void* ptr = malloc(len);
    if (!ptr) return false;
    CheckError(cudaMemcpy2DFromArray(ptr, spitch, fb.device_data, 0, 0, spitch, fb.resolution.y, cudaMemcpyDeviceToHost));
    s32 ret;
    if (size == 16)
    {
        path += ".hdr";
        ret = stbi_write_hdr(path.c_str(), fb.resolution.x, fb.resolution.y, 4, reinterpret_cast<f32*>(ptr));
    }
    else
    {
        path += ".png";
        ret = stbi_write_png(path.c_str(), fb.resolution.x, fb.resolution.y, 4, ptr, size);
    }
    free(ptr);
    return ret == 0;
}

bool CudaUtil::UnloadTexture(const Texture& tex)
{
    if (tex.resolution.x <= 0 || tex.resolution.y <= 0) return false;
    CheckError(cudaDestroyTextureObject(tex.device_tex));
    FreeArray(tex.device_data);
    return true;
}

bool CudaUtil::LoadCubemap(Cubemap& tex, const std::string& path)
{
    std::ifstream in;
    in.open(path, std::ios::in);
    if (!in.is_open())
    {
        std::cerr << "Unable to load cubemap: " << path << " : could not open file" << std::endl;
        return false;
    }
    std::string texPath = path;
    size_t decal = texPath.find_last_of('/') + 1;
    texPath = texPath.substr(0, decal);
    std::string line;
    std::string textureNames[6];
    int index = 0;
    while (std::getline(in, line))
    {
        if (!line.empty())
        {
            if (index == 6) break;
            textureNames[index] = texPath + line;
            index++;
        }
    }
    if (index != 6)
    {
        std::cerr << "Unable to load cubemap: " << path << " : not enought images in file" << std::endl;
        return false;
    }
    f32* data[6];
    for (int i = 0; i < 6; i++)
    {
        IVec2 res;
        s32 comp;
        data[i] = stbi_loadf(textureNames[i].c_str(), &res.x, &res.y, &comp, 4);
        if (!data[i])
        {
            std::cerr << "Unable to load cubemap texture: " << textureNames[i] << " : " << stbi_failure_reason() << std::endl;
            for (int j = 0; j < i; j++)
            {
                stbi_image_free(data[i]);
            }
            return false;
        }
        if (i == 0)
        {
            tex.resolution = res;
            if (tex.resolution.x <= 0 || tex.resolution.y <= 0 || tex.resolution.x != tex.resolution.y)
            {
                std::cerr << "Invalid cubemap texture resolution: " << textureNames[i] << " : " << std::endl;
                stbi_image_free(data[0]);
                return false;
            }
        }
        else if (res != tex.resolution)
        {
            std::cerr << "Invalid cubemap texture resolution: " << textureNames[i] << " : " << std::endl;
            for (int j = 0; j <= i; j++)
            {
                stbi_image_free(data[i]);
            }
            return false;
        }
    }
    for (int i = 0; i < 6; i++)
    {
        f32* ptr = data[i];
        for (u64 j = 0; j < 4llu * tex.resolution.x * tex.resolution.y; ++j)
        {
            ptr[j] = powf(ptr[j], 1 / 2.3f);
        }
    }
    const s32 fsize = sizeof(f32) * 8;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(fsize, fsize, fsize, fsize, cudaChannelFormatKindFloat);
    cudaExtent e = {};
    e.depth = 6;
    e.width = tex.resolution.x;
    e.height = tex.resolution.y;
    CheckError(cudaMalloc3DArray(&tex.device_data, &channelDesc, e, cudaArrayCubemap));
    cudaMemcpy3DParms p = {};
    p.kind = cudaMemcpyHostToDevice;
    e.depth = 1;
    p.extent = e;
    p.dstArray = tex.device_data;
    cudaPitchedPtr ptr;
    ptr.pitch = tex.resolution.x * sizeof(Vec4);
    ptr.xsize = tex.resolution.x;
    ptr.ysize = tex.resolution.y;
    for (int i = 0; i < 6; i++)
    {
        ptr.ptr = data[i];
        p.srcPtr = ptr;
        cudaPos pos = {};
        pos.x = 0;
        pos.y = 0;
        pos.z = i;
        p.dstPos = pos;
        CheckError(cudaMemcpy3D(&p));
        stbi_image_free(data[i]);
    }
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tex.device_data;

    // Specify texture object parameters
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.seamlessCubemap = 1;
    texDesc.normalizedCoords = 1;

    // Create texture object
    CheckError(cudaCreateTextureObject(&tex.device_tex, &resDesc, &texDesc, NULL));
    return true;
}

bool CudaUtil::UnloadCubemap(const Cubemap& tex)
{
    if (tex.resolution.x <= 0 || tex.resolution.y <= 0) return false;
    CheckError(cudaDestroyTextureObject(tex.device_tex));
    FreeArray(tex.device_data);
    return true;
}

bool CudaUtil::CreateFrameBuffer(FrameBuffer& tex, Maths::IVec2 res, ChannelType type)
{
    if (res.x <= 0 || res.y <= 0) return false;
    tex.resolution = res;
    const s32 fsize = (type == ChannelType::F32 ? sizeof(f32) : sizeof(u8)) * 8;
    tex.type = type;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(fsize, fsize, fsize, fsize, type == ChannelType::F32 ? cudaChannelFormatKindFloat : cudaChannelFormatKindUnsigned);
    CheckError(cudaMallocArray(&tex.device_data, &channelDesc, tex.resolution.x, tex.resolution.y, cudaArraySurfaceLoadStore));
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = tex.device_data;

    // Create texture object
    CheckError(cudaCreateSurfaceObject(&tex.device_surf, &resDesc));
    return true;
}

bool CudaUtil::UnloadFrameBuffer(const FrameBuffer& tex)
{
    if (tex.resolution.x <= 0 || tex.resolution.y <= 0) return false;
    CheckError(cudaDestroySurfaceObject(tex.device_surf));
    FreeArray(tex.device_data);
    return true;
}

std::vector<f32*> CudaUtil::GenerateTextureMipMaps(f32* source, IVec2 res, u32& maxLOD)
{
    maxLOD = Texture::GetMaxLOD(res);
    std::vector<f32*> result = std::vector<f32*>(maxLOD);
    result[0] = source;
    Vec4* ptr = reinterpret_cast<Vec4*>(source);
    for (u32 i = 1; i < maxLOD; i++)
    {
        IVec2 r = IVec2(res.x >> i, res.y >> i);
        u64 len = r.x * r.y * sizeof(float) * 4;
        f32* tmp = static_cast<f32*>(malloc(len));
        assert(tmp);
        result[i] = tmp;
        Vec4* ptr2 = reinterpret_cast<Vec4*>(tmp);
        u32 tmpRX = r.x * 2;
        for (s32 y = 0; y < r.y; y++)
        {
            for (s32 x = 0; x < r.x; x++)
            {
                Vec4 total = ptr[2 * x + 2 * y * tmpRX];
                total += ptr[2 * x + 1 + 2 * y * tmpRX];
                total += ptr[2 * x + (2 * y + 1) * tmpRX];
                total += ptr[2 * x + 1 + (2 * y + 1) * tmpRX];
                ptr2[x + y * r.x] = total / 4;
            }
        }
        ptr = ptr2;
    }
    return result;
}