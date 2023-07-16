#pragma once

#include "cuda_runtime.h"

#include "Maths/Maths.cuh"
#include "RayTracing.cuh"

namespace RayTracing
{
    class Mesh
    {
    public:
        Mesh() {};
        ~Mesh() {};

        __device__ HitRecord Intersect(Ray r, Maths::Vec2 bounds);
        __device__ void ApplyTransform(const Maths::Mat4& transform, u32 index);
        __host__ __device__ u32 GetIndiceCount();
        u32* indices = nullptr;
        Vertice* sourceVertices = nullptr;
        Vertice* transformedVertices = nullptr;
        Sphere sourceSphere;
        Sphere transformedSphere;
        u32 indiceCount = 0;
        u32 verticeCount = 0;
        u32 matIndex = 0;
    private:
    };
}
