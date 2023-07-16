#include "RayTracing/Mesh.cuh"

using namespace Maths;
using namespace RayTracing;

__device__ HitRecord Mesh::Intersect(Ray r, Vec2 bounds)
{
    HitRecord result;
    if (!HitSphere(r, transformedSphere, bounds)) return result;
    Vertice verts[3];
    for (u32 i = 0; i < indiceCount; i += 3)
    {
        verts[0] = transformedVertices[indices[i]];
        verts[1] = transformedVertices[indices[i+1]];
        verts[2] = transformedVertices[indices[i+2]];
        HitRecord hit = HitTriangle(r, verts, bounds);
        if (hit.dist < 0) continue;
        result = hit;
        bounds.y = hit.dist;
    }
    return result;
}

__device__ void Mesh::ApplyTransform(const Mat4& transform, u32 index)
{
    transformedVertices[index].pos = (transform * Vec4(sourceVertices[index].pos, 1)).GetVector();
    transformedVertices[index].normal = (transform * Vec4(sourceVertices[index].pos, 0)).GetVector();
    transformedVertices[index].uv = sourceVertices[index].uv;
}

__host__ __device__ u32 Mesh::GetIndiceCount()
{
    return indiceCount;
}