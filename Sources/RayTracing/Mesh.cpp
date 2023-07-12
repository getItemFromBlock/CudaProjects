#include "RayTracing/Mesh.hpp"

using namespace Maths;
using namespace RayTracing;

HitRecord Mesh::Intersect(Ray r, Vec2 bounds)
{
    HitRecord result;
    if (!HitSphere(r, boundingSphere, bounds)) return result;
    for (u32 i = 0; i < indiceCount; ++i)
    {
        HitRecord hit = HitTriangle(r, transformedVertices, bounds);
        if (hit.dist < 0) continue;
        result = hit;
        bounds.y = hit.dist;
    }
    return result;
}

void Mesh::ApplyTransform(const Mat4& transform, u32 index)
{
    transformedVertices[index].pos = (transform * Vec4(sourceVertices[index].pos, 1)).GetVector();
    transformedVertices[index].normal = (transform * Vec4(sourceVertices[index].pos, 0)).GetVector();
    transformedVertices[index].uv = sourceVertices[index].uv;
}

u32 Mesh::GetIndiceCount()
{
    return indiceCount;
}