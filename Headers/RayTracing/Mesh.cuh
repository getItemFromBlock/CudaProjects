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

        __device__ HitRecord Intersect(Ray r, Maths::Vec2 bounds, bool inverted = false) const;
        __device__ void FillData(const HitRecord& res, Maths::Vec3& normal, Maths::Vec3& tangent, Maths::Vec3& cotangent, Maths::Vec2& uv, const bool inverted) const;
        __host__ __device__ u32 GetIndiceCount() const;
        u32* indices = nullptr;
        Vertice* sourceVertices = nullptr;
        Vertice* transformedVertices = nullptr;
        Box sourceBox;
        Box transformedBox;
        u32 indiceCount = 0;
        u32 verticeCount = 0;
        u32 matIndex = 0;
    private:
    };
}
