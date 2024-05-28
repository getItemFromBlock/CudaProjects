#pragma once

#include "cuda_runtime.h"

#include "Maths/Maths.cuh"
#include "Compute/RayTracing.cuh"

namespace Resources
{
    class Mesh
    {
    public:
        Mesh() {};
        ~Mesh() {};

        __device__ f32 BoundsCheck(Compute::Ray r, Maths::Vec2 bounds) const;
        __device__ Compute::HitRecord Intersect(Compute::Ray r, Maths::Vec2 bounds, bool inverted = false) const;
        __device__ void FillData(const Compute::HitRecord& res, Maths::Vec3& normal, Maths::Vec3& tangent, Maths::Vec3& cotangent, Maths::Vec2& uv, const bool inverted) const;
        __host__ __device__ u32 GetIndiceCount() const;
        u32* indices = nullptr;
        Compute::Vertice* sourceVertices = nullptr;
        Compute::Vertice* transformedVertices = nullptr;
        Compute::Box sourceBox;
        Compute::Box transformedBox;
        u32 indiceCount = 0;
        u32 verticeCount = 0;
        u32 matIndex = 0;
    private:
    };
}
