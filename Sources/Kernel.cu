#include "Kernel.cuh"

#include <chrono>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace Maths;
using namespace RayTracing;

#define MAX_ITER 2048

__device__ f32 RandomUniform(curandState* const globalState, const u64 index)
{
    return curand_uniform(globalState + index);
}

__device__ f32 RandomUniform2(curandState* const globalState, const u64 index)
{
    return curand_uniform(globalState + index) * 2 - 1;
}

__device__ Vec3 Deviate(curandState* const globalState, const u64 index, const Vec3& dir, const f32 amount)
{
    if (amount == 0) return dir;
    Vec3 tang = dir.GetPerpendicular().Normalize();
    Vec3 cotang = dir.Cross(tang);
    return (tang * (RandomUniform2(globalState, index) * amount) + cotang * (RandomUniform2(globalState, index) * amount) + dir * (RandomUniform(globalState, index) * (1 - amount))).Normalize();
}

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


__device__ u32 Mandelbrot(const f64 x0, const f64 y0)
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

__global__ void FractalKernel(FrameBuffer fb, const f64 iTime)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(fb.resolution.x) * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    f64 py = (pixel.y * 2.0 - fb.resolution.y) / fb.resolution.y;
    f64 px = (pixel.x * 2.0 - fb.resolution.x) / fb.resolution.y;

    f64 tz = 0.5 - 0.5 * cos(0.09 * iTime);
    f64 zoo = pow(0.5, 50.0 * tz);
    f64 x = -0.04801030109002 + px * zoo;
    f64 y = 0.6806 + py * zoo;

    u32 val = Mandelbrot(x,y);
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

__device__ HitRecord RayTrace(const Ray& r, const Mesh* meshes, const Material* mats, u32 meshCount, f32 far, const Material*& mat, bool inverted = false)
{
    HitRecord result;
    for (u32 i = 0; i < meshCount; ++i)
    {
        HitRecord hit = meshes[i].Intersect(r, Maths::Vec2(0.0f, far), inverted);
        if (hit.dist < 0) continue;
        far = hit.dist;
        mat = mats + meshes[i].matIndex;
        result = hit;
        result.mesh = i;
    }
    return result;
}

__device__ Vec3 GetColor(Ray r, curandState* const globalState, const u64 index, const Mesh* meshes, const Material* materials, const Texture* textures, const u32 meshCount)
{
    f32 far = 100000.0f;
    Material* mat = nullptr;
    Vec3 color = Vec3(1);
    bool inverted = false;
    HitRecord result;
    u8 iterator = 0;
    while (iterator < 8)
    {
        ++iterator;
        result = RayTrace(r, meshes, materials, meshCount, far, mat, inverted);
        if (result.dist < 0)
        {
            if (iterator == 1) color *= Vec3(0.2f);
            break;
        }
        Vec3 normal;
        Vec3 tangent;
        Vec3 bitangent;
        Vec2 uv;
        meshes[result.mesh].FillData(result, normal, tangent, bitangent, uv, inverted);
        if (mat->normalTex != ~0)
        {
            Vec3 col = textures[mat->normalTex].Sample(uv).GetVector() * 2 - 1;
            normal = (tangent * col.x + bitangent * col.y + normal * col.z).Normalize();
        }
        Vec3 diffuse = mat->diffuseTex != ~0 ? textures[mat->diffuseTex].Sample(uv).GetVector() : mat->diffuseColor;
        f32 metallic = mat->metallicTex != ~0 ? textures[mat->metallicTex].Sample(uv).x : mat->metallic;
        f32 roughness = mat->roughnessTex != ~0 ? textures[mat->roughnessTex].Sample(uv).x : mat->roughness;
        roughness = roughness * roughness * roughness;
        f32 transmit = (mat->transmittanceColor.x + mat->transmittanceColor.y + mat->transmittanceColor.z) / 3;
        if (RandomUniform(globalState, index) > (metallic + transmit) / 2)
        {
            r.pos = result.pos + r.dir * 0.00001f;
            if (transmit > 0 && RandomUniform(globalState, index) < transmit)
            {
                if (!inverted)
                {
                    Vec3 dir =  r.dir.Refract(normal, 1/mat->ior);
                    if (dir.x != 0 || dir.y != 0 || dir.z != 0)
                    {
                        r.dir = Deviate(globalState, index, dir, roughness);
                        inverted = !inverted;
                    }
                    else
                    {
                        r.dir = Deviate(globalState, index, r.dir.Reflect(normal), roughness);
                    }
                    color = color * diffuse * mat->transmittanceColor + mat->emissionColor;
                }
                else
                {
                    Vec3 dir = r.dir.Refract(-normal, mat->ior);
                    if (dir.x != 0 || dir.y != 0 || dir.z != 0)
                    {
                        r.dir = Deviate(globalState, index, dir, roughness);
                        inverted = !inverted;
                    }
                    else
                    {
                        r.dir = Deviate(globalState, index, r.dir.Reflect(-normal), roughness);
                    }
                }
            }
            else
            {
                r.dir = Deviate(globalState, index, r.dir.Reflect(normal), roughness);
                color = color * diffuse + mat->emissionColor;
            }
            if (mat->emissionColor.x > 0) break;
            far = 100000.0f;
        }
        else
        {
            r.pos = result.pos;
            r.dir = Deviate(globalState, index, r.dir.Reflect(normal), roughness);
            color = color * diffuse + mat->emissionColor;
            if (mat->emissionColor.x > 0) break;
            far = 100000.0f;
        }
    }
    return Util::Clamp(color);
}

__global__ void RayTracingKernel(FrameBuffer fb, curandState* const globalState, const Mesh* meshes, const Material* mats, const Texture* texs, Vec3 pos, const Vec3 front, const Vec3 up, const f32 fov, const u32 meshCount, const u32 quality)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(fb.resolution.x) * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    Maths::Vec2 coord = (Vec2(pixel) * 2 - fb.resolution) / fb.resolution.y;
    Vec3 right = front.Cross(up);
    Ray r = Ray(pos, right * coord.x - up * coord.y + front * fov);
    Vec3 color;
    for (u32 i = 0; i < quality; ++i)
    {
        color += GetColor(r, globalState, index, meshes, mats, texs, meshCount);
    }
    fb.Write(pixel, color/quality);
}

__global__ void RayTracingKernelDebug(FrameBuffer fb, const Mesh* meshes, const Material* mats, const Texture* texs, Vec3 pos, const Vec3 front, const Vec3 up, const f32 fov, u32 meshCount)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(fb.resolution.x) * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    Maths::Vec2 coord = (Vec2(pixel) * 2 - fb.resolution) / fb.resolution.y;
    Vec3 right = front.Cross(up);
    Ray r = Ray(pos, right * coord.x - up * coord.y + front * fov);
    f32 far = 100000.0f;
    Vec3 color = Vec3(1);
    Material* mat = nullptr;
    u8 iterator = 0;
    bool inverted = false;
    while (iterator < 5)
    {
        ++iterator;
        HitRecord result = RayTrace(r, meshes, mats, meshCount, far, mat, inverted);
        if (result.dist < 0)
        {
            if (iterator == 1) color *= Vec3(0.2f);
            break;
        }
        Vec3 normal;
        Vec3 tangent;
        Vec3 bitangent;
        Vec2 uv;
        meshes[result.mesh].FillData(result, normal, tangent, bitangent, uv, inverted);
        if (mat->normalTex != ~0)
        {
            Vec3 col = texs[mat->normalTex].Sample(uv).GetVector() * 2 - 1;
            normal = (tangent * col.x + bitangent * col.y + normal * col.z).Normalize();
        }
        Vec3 diffuse = mat->diffuseTex != ~0 ? texs[mat->diffuseTex].Sample(uv).GetVector() : mat->diffuseColor;
        f32 metallic = mat->metallicTex != ~0 ? texs[mat->metallicTex].Sample(uv).x : mat->metallic;
        f32 roughness = mat->roughnessTex != ~0 ? texs[mat->roughnessTex].Sample(uv).x : mat->roughness;
        f32 transmit = (mat->transmittanceColor.x + mat->transmittanceColor.y + mat->transmittanceColor.z) / 3;
        if ((metallic + transmit) > 0.8f)
        {
            r.pos = result.pos + r.dir * 0.00001f;
            if (transmit > 0)
            {
                if (!inverted)
                {
                    Vec3 dir = r.dir.Refract(normal, 1 / mat->ior);
                    if (dir.x != 0 || dir.y != 0 || dir.z != 0)
                    {
                        r.dir = dir;
                        inverted = !inverted;
                    }
                    else
                    {
                        r.dir = r.dir.Reflect(normal);
                    }
                    color *= diffuse * mat->transmittanceColor + mat->emissionColor;
                }
                else
                {
                    Vec3 dir = r.dir.Refract(-normal, mat->ior);
                    if (dir.x != 0 || dir.y != 0 || dir.z != 0)
                    {
                        r.dir = dir;
                        inverted = !inverted;
                    }
                    else
                    {
                        r.dir = r.dir.Reflect(-normal);
                    }
                }
            }
            else
            {
                r.dir = r.dir.Reflect(normal);
                color *= diffuse + mat->emissionColor;
            }
            if (mat->emissionColor.x > 0) break;
            far = 100000.0f;
        }
        else
        {
            const Vec3 lightDir = Vec3(1, -2, 0.5f).Normalize();
            f32 pr = Util::Clamp(-lightDir.Dot(normal));
            pr += 0.2f;
            diffuse *= pr;
            r.pos = result.pos;
            r.dir = r.dir.Reflect(normal);
            color *= (diffuse*pr) + mat->emissionColor;
            if (mat->emissionColor.x > 0) break;
            far = 100000.0f;
        }
    }
    fb.Write(pixel, color);
}

__global__ void VerticeKernel(Mesh* meshes, u32 meshIndex, Vec3 pos, Quat rot, Vec3 scale)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    Mesh* mesh = meshes + meshIndex;
    if (index < mesh->verticeCount)
    {
        mesh->transformedVertices[index].pos = pos + rot * mesh->sourceVertices[index].pos * scale;
        mesh->transformedVertices[index].normal = rot * mesh->sourceVertices[index].normal;
        mesh->transformedVertices[index].uv = mesh->sourceVertices[index].uv;
        mesh->transformedVertices[index].tangent = rot * mesh->sourceVertices[index].tangent;
        mesh->transformedVertices[index].cotangent = rot * mesh->sourceVertices[index].cotangent;
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

__global__ void SeedKernel(curandState* states, u64 seed)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    curand_init(seed + index, 0, 0, states + index);
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
#ifdef RAY_TRACING
    u64 newSize = static_cast<u64>(resIn.x) * resIn.y;
    device_prngBuffer = CudaUtil::Allocate<curandState>(newSize);
    rngBufferSize = newSize;
    SeedRNGBuffer();
#endif

}

void Kernel::Resize(IVec2 resIn)
{
    CudaUtil::UnloadFrameBuffer(fb);
    CudaUtil::CreateFrameBuffer(fb, resIn);
    u64 newSize = static_cast<u64>(resIn.x) * resIn.y;
    if (newSize > rngBufferSize)
    {
        CudaUtil::Free(device_prngBuffer);
        device_prngBuffer = CudaUtil::Allocate<curandState>(newSize);
        rngBufferSize = newSize;
        SeedRNGBuffer();
    }
}

void Kernel::ClearKernels()
{
#ifdef RAY_TRACING
    CudaUtil::Free(device_prngBuffer);
#endif
    CudaUtil::UnloadFrameBuffer(fb);
    CudaUtil::ResetDevice();
}

void Kernel::RunKernels(u32* img, f64 iTime)
{
    u32 count = fb.resolution.x * fb.resolution.y;
    s32 M = CudaUtil::GetMaxThreads(deviceID);

    FractalKernel<<<(count + M - 1) / M, M>>>(fb, iTime);

    CudaUtil::CheckError(cudaGetLastError(), "FractalKernel launch failed: %s");
    CudaUtil::SynchronizeDevice();

    CudaUtil::CopyFrameBuffer(fb, img, CudaUtil::CopyType::DToH);
}

void Kernel::UpdateMeshVertices(Mesh* mesh, u32 index, const Maths::Vec3& pos, const Maths::Quat& rot, const Maths::Vec3& scale)
{
    u32 count = mesh->verticeCount + 1;
    s32 M = CudaUtil::GetMaxThreads(deviceID);
    VerticeKernel<<<(count + M - 1) / M, M>>>(device_meshes, index, pos, rot, scale);
    CudaUtil::CheckError(cudaGetLastError(), "VerticeKernel launch failed: %s");
}

s32 M = 0;
void Kernel::LaunchRTXKernels(const u32 meshCount, const Vec3& pos, const Vec3& front, const Vec3& up, const f32 fov, const bool advanced)
{
    const u32 count = fb.resolution.x * fb.resolution.y;
    if (advanced)
    {
        RayTracingKernel<<<(count + M - 1) / M, M>>>(fb, device_prngBuffer, device_meshes, device_materials, device_textures, pos, front, up, fov, meshCount, 16);
    }
    else
    {
        RayTracingKernelDebug<<<(count + M - 1) / M, M>>>(fb, device_meshes, device_materials, device_textures, pos, front, up, fov, meshCount);
    }
}

void Kernel::RenderMeshes(u32* img, const u32 meshCount, const Vec3& pos, const Vec3& front, const Vec3& up, const f32 fov, const bool advanced)
{
    if (M)
    {
        LaunchRTXKernels(meshCount, pos, front, up, fov, advanced);
        CudaUtil::CheckError(cudaGetLastError(), "RayTracingKernelDebug launch failed: %s");
    }
    else
    {
        M = CudaUtil::GetMaxThreads(deviceID);
        LaunchRTXKernels(meshCount, pos, front, up, fov, advanced);
        cudaError_t result = cudaGetLastError();
        u32 counter = 0;
        while (result != cudaSuccess && M > 16 && counter < 10)
        {
            M /= 2;
            ++counter;
            LaunchRTXKernels(meshCount, pos, front, up, fov, advanced);
            result = cudaGetLastError();
        }
        CudaUtil::CheckError(result, "Could not find adequate core number: %s");
    }
    CudaUtil::SynchronizeDevice();
    CudaUtil::CopyFrameBuffer(fb, img, CudaUtil::CopyType::DToH);
}

void Kernel::SeedRNGBuffer()
{
    s32 M = CudaUtil::GetMaxThreads(deviceID);
    u64 seed = std::chrono::system_clock::now().time_since_epoch().count();
    SeedKernel<<<((u32)rngBufferSize + M - 1) / M, M>>>(device_prngBuffer, seed);
    CudaUtil::SynchronizeDevice();
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