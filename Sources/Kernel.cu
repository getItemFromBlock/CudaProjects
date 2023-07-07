#include "kernel.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace Maths;

#define MAX_ITER 1024

__device__ f64 lerp(f64 a, f64 b, f64 f)
{
    return a * (1.0 - f) + (b * f);
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

__global__ void fractalKernel(u32* a, IVec2 res, f64 iTime)
{
    u64 index = 1llu * threadIdx.x + blockIdx.x * blockDim.x;
    if (index > 1llu * res.x * res.y) return;
    f64 py = ((index / res.x) * 2.0 - res.y) / res.y;
    f64 px = ((index % res.x) * 2.0 - res.x) / res.y;

    f64 tz = 0.5 - 0.5 * cos(0.09 * iTime);
    f64 zoo = pow(0.5, 50.0 * tz);
    f64 x = -0.04801030109002 + px * zoo;
    f64 y = 0.6806 + py * zoo;

    u32 val = mandelbrot(x,y);
    if (val == MAX_ITER)
    {
        a[index] = 0xff000000;
        return;
    }
    f32 norm = val * 1.0f / MAX_ITER;
    Vec3 rgb;
    norm = powf(norm, 0.3f);
    HSVtoRGB(rgb, Vec3(norm*360, 1, 1));
    u32 col = 0xff000000 | static_cast<u32>(rgb.x * 255) << 16 | static_cast<u32>(rgb.y * 255) << 8 | static_cast<u32>(rgb.z * 255);
    a[index] = col;
}

void Kernel::InitKernels(IVec2 resIn)
{
    res = resIn;
    utils.SelectDevice();
    dev_img = utils.Allocate<u32>(1llu*res.x*res.y);
}

void Kernel::Resize(IVec2 resIn)
{
    if (resIn.x * resIn.y > res.x * res.y) // no need to resize the buffer down, it is already large enouth to hold the new image
    {
        utils.Free(dev_img);
        dev_img = utils.Allocate<u32>(1llu * resIn.x * resIn.y);
    }
    res = resIn;
}

void Kernel::ClearKernels()
{
    utils.Free(dev_img);
    utils.ResetDevice();
}

void Kernel::RunKernels(u32* img, f64 iTime)
{
    u64 count = 1llu * res.x * res.y;
    u64 size = sizeof(u32) * count;
    s32 M = utils.GetMaxThreads();
    fractalKernel<<<((u32)count + M - 1) / M, M>>>(dev_img, res, iTime);

    utils.CheckError(cudaGetLastError(), "addKernel launch failed: %s");
    utils.SynchronizeDevice();

    utils.Copy(dev_img, img, size, CudaUtil::CopyType::DToH);
    
    return;
}

void Kernel::DrawText(u32* img, const std::string& text, IVec2 pos, u32 size)
{
    for (auto c : text)
    {

    }
}

void Kernel::SetFont(const u8* imgData, IVec2 res)
{
}
