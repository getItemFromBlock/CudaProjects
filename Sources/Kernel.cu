#include "kernel.hpp"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define MAX_ITER 1024

__device__ f64 lerp(f64 a, f64 b, f64 f)
{
    return a * (1.0 - f) + (b * f);
}

__device__ void HSVtoRGB(float& fR, float& fG, float& fB, float fH, float fS, float fV)
{
    float fC = fV * fS; // Chroma
    float fHPrime = fmodf(fH / 60.0f, 6);
    float fX = fC * (1 - fabsf(fmodf(fHPrime, 2) - 1));
    float fM = fV - fC;

    if (0 <= fHPrime && fHPrime < 1)
    {
        fR = fC;
        fG = fX;
        fB = 0;
    }
    else if (1 <= fHPrime && fHPrime < 2)
    {
        fR = fX;
        fG = fC;
        fB = 0;
    }
    else if (2 <= fHPrime && fHPrime < 3)
    {
        fR = 0;
        fG = fC;
        fB = fX;
    }
    else if (3 <= fHPrime && fHPrime < 4)
    {
        fR = 0;
        fG = fX;
        fB = fC;
    }
    else if (4 <= fHPrime && fHPrime < 5)
    {
        fR = fX;
        fG = 0;
        fB = fC;
    }
    else if (5 <= fHPrime && fHPrime < 6)
    {
        fR = fC;
        fG = 0;
        fB = fX;
    }
    else
    {
        fR = 0;
        fG = 0;
        fB = 0;
    }

    fR += fM;
    fG += fM;
    fB += fM;
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

__global__ void fractalKernel(u32* a, u32 w, u32 h, f64 iTime)
{
    u64 index = 1llu * threadIdx.x + blockIdx.x * blockDim.x;
    if (index > 1llu * w * h) return;
    f64 py = ((index / w) * 2.0 - h) / h;
    f64 px = ((index % w) * 2.0 - w) / h;

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
    f32 r, g, b;
    norm = powf(norm, 0.3f);
    HSVtoRGB(r, g, b, norm*360, 1.0f, 1.0f);
    u32 col = 0xff000000 | static_cast<u32>(r * 255) << 16 | static_cast<u32>(g * 255) << 8 | static_cast<u32>(b * 255);
    a[index] = col;
}

void Kernel::InitKernels(u32 w, u32 h)
{
    width = w;
    height = h;
    utils.SelectDevice();
    dev_img = utils.Allocate<u32>(1llu*width*height);
}

void Kernel::Resize(u32 w, u32 h)
{
    if (w * h > width * height)
    {
        utils.Free(dev_img);
        dev_img = utils.Allocate<u32>(1llu * w * h);
    }
    width = w;
    height = h;
}

void Kernel::ClearKernels()
{
    utils.Free(dev_img);
    utils.ResetDevice();
}

void Kernel::RunKernels(u32* img, u32 w, u32 h, f64 iTime)
{
    u64 count = 1llu * w * h;
    u64 size = sizeof(u32) * count;
    s32 M = utils.GetMaxThreads();
    fractalKernel<<<(count + M - 1) / M, M>>>(dev_img, w, h, iTime);

    utils.CheckError(cudaGetLastError(), "addKernel launch failed: %s");
    utils.SynchronizeDevice();

    utils.Copy(dev_img, img, size, CudaUtil::CopyType::DToH);
    
    return;
}

void Kernel::DrawText(u32* img, u32 w, u32 h, const std::string& text, u32 x, u32 y, u32 size)
{
    for (auto c : text)
    {

    }
}

void Kernel::SetFont(const u8* imgData, u32 w, u32 h)
{
}
