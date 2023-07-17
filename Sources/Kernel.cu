#include "Kernel.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace Maths;
using namespace RayTracing;

#define MAX_ITER 2048

__device__ void HSVtoRGB(Vec3& rgb, const Vec3& hsv)
{
    f32 fC = hsv.z * hsv.y; // Chroma
    f32 fHPrime = fmodf(hsv.x / 60.0f, 6);
    f32 fX = fC * (1 - fabsf(fmodf(fHPrime, 2) - 1));
    f32 fM = hsv.z - fC;

    if (0 <= fHPrime && fHPrime < 1)
    {
        rgb = Vec3(fC, fX, 0);
    }
    else if (1 <= fHPrime && fHPrime < 2)
    {
        rgb = Vec3(fX, fC, 0);
    }
    else if (2 <= fHPrime && fHPrime < 3)
    {
        rgb = Vec3(0, fC, fX);
    }
    else if (3 <= fHPrime && fHPrime < 4)
    {
        rgb = Vec3(0, fX, fC);
    }
    else if (4 <= fHPrime && fHPrime < 5)
    {
        rgb = Vec3(fX, 0, fC);
    }
    else if (5 <= fHPrime && fHPrime < 6)
    {
        rgb = Vec3(fC, 0, fX);
    }
    else
    {
        rgb = Vec3();
    }
    rgb += fM;
}


__device__ u32 mandelbrot(f64 x0, f64 y0)
{
    u32 iteration = 0;
    f64 x = 0;
    f64 y = 0;
    f64 xold = 0;
    f64 yold = 0;
    u32 period = 0;
    while (x * x + y * y <= 4 && iteration < MAX_ITER)
    {
        f64 xtemp = x * x - y * y + x0;
        y = 2 * x * y + y0;
        x = xtemp;
        ++iteration;
        if (fabs(x - xold) < 0.0000001 && fabs(y - yold) < 0.0000001)
        {
            iteration = MAX_ITER;
            break;
        }
        ++period;
        if (period > 20)
        {
            period = 0;
            xold = x;
            yold = y;
        }
    }
    return iteration;
}

__global__ void fractalKernel(FrameBuffer fb, f64 iTime)
{
    u64 index = 1llu * threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= 1llu * fb.resolution.x * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    f64 py = (pixel.y * 2.0 - fb.resolution.y) / fb.resolution.y;
    f64 px = (pixel.x * 2.0 - fb.resolution.x) / fb.resolution.y;

    f64 tz = 0.5 - 0.5 * cos(0.09 * iTime);
    f64 zoo = pow(0.5, 50.0 * tz);
    f64 x = -0.04801030109002 + px * zoo;
    f64 y = 0.6806 + py * zoo;

    u32 val = mandelbrot(x,y);
    if (val == MAX_ITER)
    {
        fb.Write(pixel, Maths::Vec3());
        return;
    }
    f32 norm = val * 1.0f / MAX_ITER;
    Vec3 rgb;
    norm = powf(norm, 0.3f);
    HSVtoRGB(rgb, Vec3(norm*360, 1, 1));
    fb.Write(pixel, rgb);
}

__global__ void rayTracingKernel(FrameBuffer fb, Mesh* meshes, Material* mats, Texture* texs, Vec3 pos, Vec3 front, Vec3 up, u32 meshCount)
{
    u64 index = 1llu * threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= 1llu * fb.resolution.x * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    Maths::Vec2 coord = (Vec2(pixel) * 2 - fb.resolution) / fb.resolution.y;
    Vec3 right = front.Cross(up);
    Ray r = Ray(pos, right * coord.x + up * coord.y + front * 2);
    f32 far = 100000.0f;
    Material* mat = nullptr;
    HitRecord result;
    for (u32 i = 0; i < meshCount; ++i)
    {
        HitRecord hit = meshes[i].Intersect(r, Maths::Vec2(0.00001f, far));
        if (hit.dist < 0) continue;
        far = hit.dist;
        mat = mats + meshes[i].matIndex;
        result = hit;
    }
    fb.Write(pixel, mat ? (mat->diffuseTex != ~0 ? texs[mat->diffuseTex].Sample(result.uv) : mat->diffuseColor) : Vec3());
}

__global__ void verticeKernel(Mesh* meshes, u32 meshIndex, Vec3 pos, Quat rot, Vec3 scale)
{
    u64 index = 1llu * threadIdx.x + blockIdx.x * blockDim.x;
    Mesh* mesh = meshes + meshIndex;
    if (index < mesh->verticeCount)
    {
        mesh->transformedVertices[index].pos = pos + rot * mesh->sourceVertices[index].pos * scale;
        mesh->transformedVertices[index].normal = rot * mesh->sourceVertices[index].normal;
        mesh->transformedVertices[index].uv = mesh->sourceVertices[index].uv;
    }
    else if (index == mesh->verticeCount)
    {
        mesh->transformedBox.center = pos + rot * mesh->sourceBox.center * scale;
        mesh->transformedBox.radius = mesh->sourceBox.radius * scale;
        for (u8 i = 0; i < 3; ++i)
        {
            mesh->transformedBox.invRadius[i] = 1 / mesh->transformedBox.radius[i];
        }
        mesh->transformedBox.rotation = rot;
    }
}

void Kernel::InitKernels(IVec2 resIn, s32 id)
{
    if (id < 0)
    {
        deviceID = CudaUtil::SelectDevice();
    }
    else
    {
        deviceID = id;
        CudaUtil::UseDevice(id);
    }
    CudaUtil::CreateFrameBuffer(fb, resIn);
}

void Kernel::Resize(IVec2 resIn)
{
    CudaUtil::UnloadFrameBuffer(fb);
    CudaUtil::CreateFrameBuffer(fb, resIn);
}

void Kernel::ClearKernels()
{
    CudaUtil::UnloadFrameBuffer(fb);
    CudaUtil::ResetDevice();
}

void Kernel::RunKernels(u32* img, f64 iTime)
{
    u32 count = fb.resolution.x * fb.resolution.y;
    s32 M = CudaUtil::GetMaxThreads(deviceID);

    fractalKernel<<<(count + M - 1) / M, M>>>(fb, iTime);

    CudaUtil::CheckError(cudaGetLastError(), "fractalKernel launch failed: %s");
    CudaUtil::SynchronizeDevice();

    CudaUtil::CopyFrameBuffer(fb, img, CudaUtil::CopyType::DToH);
}

void Kernel::UpdateMeshVertices(Mesh* mesh, u32 index, const Maths::Vec3& pos, const Maths::Quat& rot, const Maths::Vec3& scale)
{
    u32 count = mesh->verticeCount + 1;
    s32 M = CudaUtil::GetMaxThreads(deviceID);
    verticeKernel<<<(count + M - 1) / M, M>>>(device_meshes, index, pos, rot, scale);
    CudaUtil::CheckError(cudaGetLastError(), "verticeKernel launch failed: %s");
}

void Kernel::RenderMeshes(u32* img, u32 meshCount, Vec3 pos, Vec3 front, Vec3 up)
{
    u32 count = fb.resolution.x * fb.resolution.y;
    s32 M = CudaUtil::GetMaxThreads(deviceID);
    M /= 2;

    rayTracingKernel<<<(count + M - 1) / M, M>>>(fb, device_meshes, device_materials, device_textures, pos, front, up, meshCount);

    CudaUtil::CheckError(cudaGetLastError(), "rayTracingKernel launch failed: %s");
    CudaUtil::SynchronizeDevice();

    CudaUtil::CopyFrameBuffer(fb, img, CudaUtil::CopyType::DToH);
}

void Kernel::UnloadMeshes(const std::vector<Mesh>& meshes)
{
    CudaUtil::Free(device_meshes);
    device_meshes = nullptr;
    for (auto& mesh : meshes)
    {
        CudaUtil::Free(mesh.indices);
        CudaUtil::Free(mesh.sourceVertices);
        CudaUtil::Free(mesh.transformedVertices);
    }
}

void Kernel::UnloadMaterials()
{
    CudaUtil::Free(device_materials);
    device_materials = nullptr;
}

void Kernel::UnloadTextures(const std::vector<Texture>& textures)
{
    CudaUtil::Free(device_textures);
    device_textures = nullptr;
    for (auto& tex : textures)
    {
        CudaUtil::UnloadTexture(tex);
    }
}

void Kernel::Synchronize()
{
    CudaUtil::SynchronizeDevice();
}

void Kernel::LoadMeshes(const std::vector<Mesh> meshes)
{
    device_meshes = CudaUtil::Allocate<Mesh>(meshes.size());
    CudaUtil::Copy(meshes.data(), device_meshes, sizeof(Mesh) * meshes.size(), CudaUtil::CopyType::HToD);
}

void Kernel::LoadTextures(const std::vector<Texture> textures)
{
    device_textures = CudaUtil::Allocate<Texture>(textures.size());
    CudaUtil::Copy(textures.data(), device_textures, sizeof(Texture) * textures.size(), CudaUtil::CopyType::HToD);
}

void Kernel::LoadMaterials(const std::vector<Material> materials)
{
    device_materials = CudaUtil::Allocate<Material>(materials.size());
    CudaUtil::Copy(materials.data(), device_materials, sizeof(Material) * materials.size(), CudaUtil::CopyType::HToD);
}