#pragma once

#include "cuda_runtime.h"

#include "Maths/Maths.cuh"

namespace RayTracing
{
    struct Vertice
    {
        Maths::Vec3 pos;
        Maths::Vec3 normal;
        Maths::Vec2 uv;
    };

    struct HitRecord
    {
        Maths::Vec3 normal;
        Maths::Vec3 pos;
        Maths::Vec2 uv;
        f32 dist = -1;
    };

    struct Ray
    {
        Maths::Vec3 pos;
        Maths::Vec3 dir;
    };

    struct Sphere
    {
        Maths::Vec3 pos;
        f32 radius = 0;
    };

    __host__ __device__ bool HitSphere(Ray r, Sphere sp, Maths::Vec2 bounds);

    __host__ __device__ HitRecord HitTriangle(Ray r, Vertice* vertices, Maths::Vec2 bounds);
}
