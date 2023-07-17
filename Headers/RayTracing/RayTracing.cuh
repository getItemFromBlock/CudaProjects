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
        Maths::Vec3 pos;
        Maths::Vec3 normal;
        Maths::Vec2 uv;
        f32 dist = -1;
    };

    struct Ray
    {
        Maths::Vec3 pos;
        Maths::Vec3 dir;

        __host__ __device__ Ray(Maths::Vec3 position, Maths::Vec3 direction) : pos(position), dir(direction.Normalize()) {}
    };

    struct Box
    {
        Maths::Vec3 center;
        Maths::Vec3 radius;
        Maths::Vec3 invRadius;
        Maths::Quat rotation;
    };

    struct Sphere
    {
        Maths::Vec3 pos;
        f32 radius = 0;
    };

    struct BVec2
    {
        bool x;
        bool y;
        __host__ __device__ BVec2() : x(false), y(false) {}
        __host__ __device__ BVec2(bool a, bool b) : x(a), y(b) {}
    };

    struct BVec3
    {
        bool x;
        bool y;
        bool z;
        __host__ __device__ BVec3() : x(false), y(false), z(false) {}
        __host__ __device__ BVec3(bool a, bool b, bool c) : x(a), y(b), z(c) {}
    };

    __host__ __device__ bool HitSphere(const Ray& r, const Sphere& sp, Maths::Vec2 bounds);

    __host__ __device__ HitRecord HitTriangle(const Ray& r, Vertice* vertices, Maths::Vec2 bounds);

    __host__ __device__ bool HitBox(Ray r, const Box& box, Maths::Vec2 bounds);
}
