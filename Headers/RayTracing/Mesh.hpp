#pragma once

#include "Maths/Maths.hpp"
#include "RayTracing.hpp"

namespace RayTracing
{
    class Mesh
    {
    public:
        Mesh() {};
        ~Mesh() {};

        HitRecord Intersect(Ray r, Maths::Vec2 bounds);
        void ApplyTransform(const Maths::Mat4& transform, u32 index);
        u32 GetIndiceCount();
    private:
        u32* indices = nullptr;
        Vertice* sourceVertices = nullptr;
        Vertice* transformedVertices = nullptr;
        Sphere boundingSphere;
        u32 indiceCount = 0;
        u32 matIndex = 0;
    };
}
