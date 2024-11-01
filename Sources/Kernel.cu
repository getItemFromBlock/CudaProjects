#include "Kernel.cuh"

#include <chrono>

// Reference here: https://developercommunity.microsoft.com/t/Add-support-for-CUDA-extensions-to-Intel/399545#T-N10086632
#ifdef __INTELLISENSE__
#define __CUDACC__
#define CUDA_KERNEL(...)
#define FAKEINIT = {0}
#else
#define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#define FAKEINIT
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace Maths;
using namespace Compute;
using namespace Resources;

#define MAX_ITER 2048
#define M 256

__device__ f32 RandomUniform(curandState* const globalState, const u64 index)
{
    return curand_uniform(globalState + index);
}

__device__ Vec3 RandomDirection(curandState* const globalState, const u64 index)
{
    float z = RandomUniform(globalState, index) * 2 - 1;
    float a = RandomUniform(globalState, index) * CURAND_2PI;
    float r = sqrt(1.0f - z * z);
    float x = r * cos(a);
    float y = r * sin(a);
    return Vec3(x, y, z);
}


__device__ Vec3 Deviate(curandState* const globalState, const u64 index, const Vec3& dir, const float amount)
{
    if (amount == 0) return dir;
    Vec3 tang = dir.GetPerpendicular().Normalize();
    Vec3 cotang = dir.Cross(tang);
    return (tang * (RandomUniform(globalState, index) * amount) + cotang * (RandomUniform(globalState, index) * amount) + dir * (RandomUniform(globalState, index) * (2 - amount))).Normalize();
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

#define INV_SQRT_OF_2PI 0.39894228040143267793994605993439f  // 1.0/SQRT_OF_2PI
#define INV_PI 0.31830988618379067153776752674503f

// https://www.shadertoy.com/view/3dd3Wr
__device__ Vec4 smartDeNoise(const FrameBuffer& tex, Vec2 uv, f32 sigma, f32 kSigma, f32 threshold)
{
    f32 radius = roundf(kSigma * sigma);
    f32 radQ = radius * radius;

    f32 invSigmaQx2 = 0.5f / (sigma * sigma);      // 1.0 / (sigma^2 * 2.0)
    f32 invSigmaQx2PI = INV_PI * invSigmaQx2;    // 1.0 / (sqrt(PI) * sigma)

    f32 invThresholdSqx2 = 0.5f / (threshold * threshold);     // 1.0 / (sigma^2 * 2.0)
    f32 invThresholdSqrt2PI = INV_SQRT_OF_2PI / threshold;   // 1.0 / (sqrt(2*PI) * sigma)

    Vec2 size = Vec2(tex.GetResolution());
    Vec4 centrPx = tex.SampleVec(uv);

    f32 zBuff = 0;
    Vec4 aBuff = Vec4(0);

    for (f32 x = -radius; x <= radius; x++) {
        f32 pt = sqrtf(radQ - x * x);  // pt = yRadius: have circular trend
        for (f32 y = -pt; y <= pt; y++) {
            Vec2 d = Vec2(x, y);

            f32 blurFactor = expf(-d.Dot(d) * invSigmaQx2) * invSigmaQx2PI;

            IVec2 c = IVec2(uv + d);
            c = Util::Clamp(c, IVec2(), tex.GetResolution());
            Vec4 walkPx = tex.SampleVec(c);

            Vec4 dC = walkPx - centrPx;
            f32 deltaFactor = exp(-dC.Dot(dC) * invThresholdSqx2) * invThresholdSqrt2PI * blurFactor;

            zBuff += deltaFactor;
            aBuff += walkPx * deltaFactor;
        }
    }
    return aBuff / zBuff;
}

__device__ Vec3 GetColor(Ray r, curandState* const globalState, const u64 index, const Mesh* meshes, const Material* materials, const Texture* textures, const Cubemap* cbs, const u32 meshCount)
{
    f32 far = 100000.0f;
    const Material* mat = nullptr;
    Vec3 throughput = Vec3(1);
    Vec3 output = Vec3();
    bool inverted = false;
    HitRecord result;
    for (u32 iterator = 0; iterator < 64; iterator++)
    {
        result = RayTrace(r, meshes, materials, meshCount, far, mat, inverted);
        if (result.dist < 0)
        {
            output += cbs[0].Sample(r.dir).GetVector() * throughput * 1.0f;
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
            Vec3 normalT = (tangent * col.x + bitangent * col.y + normal * col.z).Normalize();
            normal = normal * 0.5f + normalT * 0.5f;
        }
        Vec3 diffuse = mat->diffuseTex != ~0 ? textures[mat->diffuseTex].Sample(uv).GetVector() : mat->diffuseColor;

        if (mat->shouldTeleport == 0)
        {
            f32 metallic = mat->metallicTex != ~0 ? textures[mat->metallicTex].Sample(uv).x : mat->metallic;
            f32 roughness = mat->roughnessTex != ~0 ? textures[mat->roughnessTex].Sample(uv).x : mat->roughness;
            roughness *= roughness;
            //roughness = sqrtf(roughness);
            f32 transmit = (mat->transmittanceColor.x + mat->transmittanceColor.y + mat->transmittanceColor.z) / 3;
            transmit = (RandomUniform(globalState, index) < (transmit - metallic / 2)) ? 1.0f : 0.0f;
            f32 doSpecular = (RandomUniform(globalState, index) < metallic) ? 1.0f : 0.0f;
            f32 ior = mat->ior;
            if (!inverted)
            {
                ior = 1 / ior;
            }
            else
            {
                normal = -normal;
                transmit = 1.0f;
                diffuse = Vec3(1);
            }

            Vec3 diffuseRayDir = (normal + RandomDirection(globalState, index)).Normalize();
            Vec3 specularRayDir;
            if (transmit != 0)
            {
                specularRayDir = r.dir.Refract(normal, ior);
                doSpecular = transmit;
                diffuse *= mat->transmittanceColor;
                inverted = !inverted;
            }
            else
            {
                specularRayDir = r.dir.Reflect(normal);
            }
            specularRayDir = Util::Lerp(specularRayDir, diffuseRayDir, roughness).Normalize();

            r.dir = Util::Lerp(diffuseRayDir, specularRayDir, doSpecular);
            //r.dir = specularRayDir;

            r.pos = result.pos;
            if (doSpecular != 0)
            {
                r.pos += normal * 0.00001f;
            }
        }
        else
        {
            r.pos = result.pos + r.dir * 0.00001f;
            RayTracing::ApplyMaterialDisplacement(r, mat);
        }

        // https://www.shadertoy.com/view/WsBBR3

        output += mat->emissionColor * throughput;
        throughput *= diffuse;

        float p = Util::MaxF(throughput.x, Util::MaxF(throughput.y, throughput.z));
        if (RandomUniform(globalState, index) > p)
            break;

        // Add the energy we 'lose' by randomly terminating paths
        throughput *= 1.0f / p;
    }
    //if (output.x == 0 && output.y == 0 && output.z == 0)
    //{
    //    output = cbs[0].Sample(r.dir).GetVector();
    //}
    return output;
}

__global__ void RayTracingKernel(FrameBuffer fb, curandState* const globalState, const Mesh* meshes, const Material* mats, const Texture* texs, const Cubemap* cbs, Vec3 pos, const Vec3 front, const Vec3 up, const f32 fov, const u32 meshCount)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(fb.resolution.x) * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    // Anti aliasing
    Maths::Vec2 pixel2 = Vec2(pixel) + Vec2(RandomUniform(globalState, index), RandomUniform(globalState, index)) - 0.5f;
    Maths::Vec2 coord = (pixel2 * 2 - fb.resolution) / fb.resolution.y;
    Vec3 right = front.Cross(up);
    Ray r = Ray(pos, right * coord.x - up * coord.y + front * fov);
    Vec3 color = GetColor(r, globalState, index, meshes, mats, texs, cbs, meshCount);

    Vec4 lastFrameColor = fb.SampleVec(pixel);

    /*
    lastFrameColor = lastFrameColor * 100.0f;
    lastFrameColor = (Vec4(color, 1.0f) + lastFrameColor) / 101.0f;
    fb.Write(pixel, lastFrameColor);
    */
    f32 blend = lastFrameColor.w == 0.0f ? 1.0f : 1.0f / (1.0f + (1.0f / lastFrameColor.w));
    color = Util::Lerp(lastFrameColor.GetVector(), color, blend);

    fb.Write(pixel, Vec4(color, blend));
}

// WARNING: The two framebuffers MUST have the same resolution
__global__ void CopyKernel(const FrameBuffer source, FrameBuffer destination, const f32 colorMultiplier)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(source.resolution.x) * source.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % source.resolution.x, index / source.resolution.x);
    Vec4 c = source.SampleVec(pixel) * colorMultiplier;
    destination.Write(pixel, c);
}

// WARNING: The two framebuffers MUST have the same resolution
__global__ void CopyKernelDenoise(const FrameBuffer source, FrameBuffer destination, const f32 strength)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(source.resolution.x) * source.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % source.resolution.x, index / source.resolution.x);

    Vec4 c = smartDeNoise(source, Vec2(pixel), 5.0f, 1.0f, strength);
    destination.Write(pixel, c);

    // https://www.shadertoy.com/view/3lcfDM
}

__global__ void ClearKernel(FrameBuffer fb, const Vec4 color)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(fb.resolution.x) * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    fb.Write(pixel, color);
}

__global__ void RayTracingKernelPreview(FrameBuffer fb, const Mesh* meshes, const Material* mats, const Texture* texs, const Cubemap* cbs, Vec3 pos, const Vec3 front, const Vec3 up, const f32 fov, u32 meshCount)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(fb.resolution.x) * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    Maths::Vec2 coord = (Vec2(pixel) * 2 - fb.resolution) / fb.resolution.y;
    Vec3 right = front.Cross(up);
    Ray r = Ray(pos, right * coord.x - up * coord.y + front * fov);
    f32 far = 100000.0f;
    Vec3 color;
    const Material* mat = nullptr;
    HitRecord result = RayTrace(r, meshes, mats, meshCount, far, mat);
    if (result.dist < 0)
    {
        color = cbs[0].Sample(r.dir).GetVector();
        //if (iterator == 1) color *= 0.2f;
    }
    else
    {
        Vec3 normal;
        Vec3 tangent;
        Vec3 bitangent;
        Vec2 uv;
        meshes[result.mesh].FillData(result, normal, tangent, bitangent, uv, false);
        if (mat->normalTex != ~0)
        {
            Vec3 col = texs[mat->normalTex].Sample(uv).GetVector() * 2 - 1;
            normal = (tangent * col.x + bitangent * col.y + normal * col.z).Normalize();
        }
        Vec3 diffuse = mat->diffuseTex != ~0 ? texs[mat->diffuseTex].Sample(uv).GetVector() : mat->diffuseColor;
        const Vec3 lightDir = Vec3(1, -2, 0.5f).Normalize();
        f32 pr = Util::Clamp(-lightDir.Dot(normal));
        pr += 0.2f;
        color = diffuse * pr + mat->emissionColor;
    }
    fb.Write(pixel, color);
}

__global__ void RayTracingKernelDebug(FrameBuffer fb, const Mesh* meshes, const Material* mats, Vec3 pos, const Vec3 front, const Vec3 up, const f32 fov, u32 meshCount)
{
    u64 index = threadIdx.x + static_cast<u64>(blockIdx.x) * blockDim.x;
    if (index >= static_cast<u64>(fb.resolution.x) * fb.resolution.y) return;
    Maths::IVec2 pixel = Maths::IVec2(index % fb.resolution.x, index / fb.resolution.x);
    Maths::Vec2 coord = (Vec2(pixel) * 2 - fb.resolution) / fb.resolution.y;
    Vec3 right = front.Cross(up);
    Ray r = Ray(pos, right * coord.x - up * coord.y + front * fov);
    f32 far = 100000.0f;
    Vec3 output = Vec3(0.5f);
    u32 hitCount = 0;
    for (u32 i = 0; i < meshCount; ++i)
    {
        f32 hit = meshes[i].BoundsCheck(r, Maths::Vec2(0.0f, far));
        if (hit > far) continue;
        u32 a = i >> 1;
        u32 b = a >> 1;
        Vec3 color = Vec3(i & 0x1, a & 0x1, b & 0x1);
        output = (output * hitCount + color) / (hitCount + 1);
        hitCount++;
    }
    fb.Write(pixel, output);
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
    CudaUtil::CreateFrameBuffer(mainFB, resIn, ChannelType::F32);
    CudaUtil::CreateFrameBuffer(surfaceFB, resIn, ChannelType::U8);
#ifdef RAY_TRACING
    u64 newSize = static_cast<u64>(resIn.x) * resIn.y;
    device_prngBuffer = CudaUtil::Allocate<curandState>(newSize);
    rngBufferSize = newSize;
    SeedRNGBuffer();
#endif

}

void Kernel::Resize(IVec2 resIn)
{
    CudaUtil::UnloadFrameBuffer(mainFB);
    CudaUtil::UnloadFrameBuffer(surfaceFB);
    CudaUtil::CreateFrameBuffer(mainFB, resIn, ChannelType::F32);
    CudaUtil::CreateFrameBuffer(surfaceFB, resIn, ChannelType::U8);
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
    CudaUtil::UnloadFrameBuffer(mainFB);
    CudaUtil::UnloadFrameBuffer(surfaceFB);
    CudaUtil::ResetDevice();
}

void Kernel::RunFractalKernels(u32* img, f64 iTime)
{
    u32 count = mainFB.resolution.x * mainFB.resolution.y;
    s32 Mmax = CudaUtil::GetMaxThreads(deviceID);

    FractalKernel CUDA_KERNEL((count + Mmax - 1) / Mmax, Mmax) (surfaceFB, iTime);

    CudaUtil::CheckError(cudaGetLastError(), "FractalKernel launch failed: %s");
    CudaUtil::SynchronizeDevice();

    CudaUtil::CopyFrameBuffer(surfaceFB, img, CudaUtil::CopyType::DToH);
}

void Kernel::UpdateMeshVertices(Mesh* mesh, u32 index, const Maths::Vec3& pos, const Maths::Quat& rot, const Maths::Vec3& scale)
{
    u32 count = mesh->verticeCount + 1;
    s32 Mmax = CudaUtil::GetMaxThreads(deviceID);
    VerticeKernel CUDA_KERNEL((count + Mmax - 1) / Mmax, Mmax) (device_meshes, index, pos, rot, scale);
    CudaUtil::CheckError(cudaGetLastError(), "VerticeKernel launch failed: %s");
}

void Kernel::LaunchRTXKernels(const u32 meshCount, const Vec3& pos, const Vec3& front, const Vec3& up, const f32 fov, const u32 quality, const f32 strength, const LaunchParams params)
{
    const u32 count = mainFB.resolution.x * mainFB.resolution.y;
    if (params & CLEAR) ClearKernel CUDA_KERNEL((count + M - 1) / M, M) (mainFB, Vec4());
    if (params & ADVANCED)
    {
        for (u32 i = 0; i < quality; ++i)
        {
            RayTracingKernel CUDA_KERNEL((count + M - 1) / M, M) (mainFB, device_prngBuffer, device_meshes, device_materials, device_textures, device_cubemaps, pos, front, up, fov, meshCount);
            CudaUtil::SynchronizeDevice();
        }
        if (params & DENOISE)
        {
            CopyKernelDenoise CUDA_KERNEL((count + M - 1) / M, M) (mainFB, surfaceFB, strength);
        }
        else
        {
            CopyKernel CUDA_KERNEL((count + M - 1) / M, M) (mainFB, surfaceFB, 1.0f/(quality));
        }
    }
    else if (params & BOXDEBUG)
    {
        RayTracingKernelDebug CUDA_KERNEL((count + M - 1) / M, M) (surfaceFB, device_meshes, device_materials, pos, front, up, fov, meshCount);
    }
    else
    {
        RayTracingKernelPreview CUDA_KERNEL((count + M - 1) / M, M) (surfaceFB, device_meshes, device_materials, device_textures, device_cubemaps, pos, front, up, fov, meshCount);
    }
}

void Kernel::RenderMeshes(u32* img, const u32 meshCount, const Vec3& pos, const Vec3& front, const Vec3& up, const f32 fov, const u32 quality, const f32 strength, const LaunchParams params)
{
    LaunchRTXKernels(meshCount, pos, front, up, fov, quality, strength, params);
    CudaUtil::CheckError(cudaGetLastError(), "RayTracingKernelDebug launch failed: %s");
    CudaUtil::SynchronizeDevice();
    CudaUtil::CopyFrameBuffer(surfaceFB, img, CudaUtil::CopyType::DToH);
}

void Kernel::SeedRNGBuffer()
{
    s32 Mmax = CudaUtil::GetMaxThreads(deviceID);
    u64 seed = std::chrono::system_clock::now().time_since_epoch().count();
    SeedKernel CUDA_KERNEL(((u32)rngBufferSize + Mmax - 1) / Mmax, Mmax) (device_prngBuffer, seed);
    CudaUtil::SynchronizeDevice();
}

void Kernel::UnloadMeshes(const std::vector<Mesh>& meshes)
{
    if (!meshes.size()) return;
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
    if (!device_materials) return;
    CudaUtil::Free(device_materials);
    device_materials = nullptr;
}

void Kernel::UnloadTextures(const std::vector<Texture>& textures)
{
    if (!textures.size()) return;
    CudaUtil::Free(device_textures);
    device_textures = nullptr;
    for (auto& tex : textures)
    {
        CudaUtil::UnloadTexture(tex);
    }
}

void Kernel::UnloadCubemaps(const std::vector<Cubemap>& cubemaps)
{
    if (!cubemaps.size()) return;
    CudaUtil::Free(device_cubemaps);
    device_cubemaps = nullptr;
    for (auto& tex : cubemaps)
    {
        CudaUtil::UnloadCubemap(tex);
    }
}

void Kernel::Synchronize()
{
    CudaUtil::SynchronizeDevice();
}

void Kernel::LoadMeshes(const std::vector<Mesh> meshes)
{
    if (meshes.size() == 0) return;
    device_meshes = CudaUtil::Allocate<Mesh>(meshes.size());
    CudaUtil::Copy(meshes.data(), device_meshes, sizeof(Mesh) * meshes.size(), CudaUtil::CopyType::HToD);
}

void Kernel::LoadTextures(const std::vector<Texture> textures)
{
    if (textures.size() == 0) return;
    device_textures = CudaUtil::Allocate<Texture>(textures.size());
    CudaUtil::Copy(textures.data(), device_textures, sizeof(Texture) * textures.size(), CudaUtil::CopyType::HToD);
}

void Kernel::LoadCubemaps(const std::vector<Cubemap> cubemaps)
{
    if (cubemaps.size() == 0) return;
    device_cubemaps = CudaUtil::Allocate<Cubemap>(cubemaps.size());
    CudaUtil::Copy(cubemaps.data(), device_cubemaps, sizeof(Cubemap) * cubemaps.size(), CudaUtil::CopyType::HToD);
}

void Kernel::LoadMaterials(const std::vector<Material> materials)
{
    if (materials.size() == 0) return;
    device_materials = CudaUtil::Allocate<Material>(materials.size());
    CudaUtil::Copy(materials.data(), device_materials, sizeof(Material) * materials.size(), CudaUtil::CopyType::HToD);
}

const FrameBuffer& Kernel::GetMainFrameBuffer()
{
    return mainFB;
}