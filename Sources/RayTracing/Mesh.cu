#include "RayTracing/Mesh.cuh"

using namespace Maths;
using namespace RayTracing;

__device__ HitRecord Mesh::Intersect(Ray r, Vec2 bounds, bool inverted) const
{
    HitRecord result;
    if (!HitBox(r, transformedBox, bounds)) return result;
    Vec3 verts[3];
    for (u32 i = 0; i < indiceCount; i += 3)
    {
        verts[0] = transformedVertices[indices[i]].pos;
        verts[1] = transformedVertices[indices[i+1+inverted]].pos;
        verts[2] = transformedVertices[indices[i+2-inverted]].pos;
        HitRecord hit = HitTriangle(r, verts, bounds);
        if (hit.dist < 0) continue;
        result = hit;
        result.indice = i;
        bounds.y = hit.dist;
    }
    return result;
}

__device__ const u32 indexes[6] =
{
    2,
    0,
    1,
    2,
    1,
    0
};

__device__ void Mesh::FillData(const HitRecord& res, Vec3& normal, Vec3& tangent, Vec3& cotangent, Vec2& uv, const bool inverted) const
{
    if (res.indice >= indiceCount) return;
    for (u32 i = 0; i < 3; ++i)
    {
        Vertice& vert = transformedVertices[indices[res.indice + i]];
        u32 j = inverted ? indexes[i + 3] : indexes[i];
        normal += vert.normal * res.barycentric[j];
        uv += vert.uv * res.barycentric[j];
        if (i == 0)
        {
            tangent = vert.tangent;
            cotangent = vert.cotangent;
        }
    }
    normal = normal.Normalize();
}

__host__ __device__ u32 Mesh::GetIndiceCount() const
{
    return indiceCount;
}