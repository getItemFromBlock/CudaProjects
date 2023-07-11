#include "Maths/RayTracing.hpp"

bool Maths::HitSphere(Ray r, Sphere sp, Vec2 bounds)
{
    f32 t0, t1;
    Vec3 L = r.pos - sp.pos;
    f32 a = r.dir.Dot();
    f32 b = 2 * r.dir.Dot(L);
    f32 c = L.Dot() - sp.radius;
    f32 discr = b * b - 4 * a * c;
    if (discr < 0)
    {
        return false;
    }
    else if (discr == 0)
    {
        t0 = -0.5f * b / a;
        t1 = t0;
    }
    else
    {
        f32 q = (b > 0) ? -0.5f * (b + sqrtf(discr)) : -0.5f * (b - sqrtf(discr));
        t0 = q / a;
        t1 = c / q;
    }
    if (t0 > t1)
    {
        f32 tmp = t0;
        t0 = t1;
        t1 = tmp;
    }
    if ((t0 < bounds.x && t1 < bounds.x) || (t0 > bounds.y && t1 > bounds.y))
    {
        return false;
    }
    return true;
}

Maths::HitRecord Maths::HitTriangle(Ray r, Vertice* vertices, Maths::Vec2 bounds)
{
    HitRecord result;
    Maths::Vec3 A = vertices[0].pos;
    Maths::Vec3 AB = vertices[1].pos - A;
    Maths::Vec3 AC = vertices[2].pos - A;
    Maths::Vec3 bar;
    Vec3 pvec = r.dir.Cross(AC);
    f32 det = AB.Dot(pvec);
    if (det < 0.00001f) return result;
    Vec3 tvec = r.pos - A;
    bar.x = tvec.Dot(pvec);
    if (bar.x < 0 || bar.x > det) return result;
    Vec3 qvec = tvec.Cross(AB);
    bar.y = r.dir.Dot(qvec);
    if (bar.y < 0 || bar.y + bar.x > det) return result;
    det = 1 / det;
    result.dist = AC.Dot(qvec) * det;
    if (result.dist < bounds.x || result.dist > bounds.y)
    {
        result.dist = -1;
        return result;
    }
    bar *= det;
    bar.z = 1 - (bar.x + bar.y);
    for (u8 i = 0; i < 3; ++i)
    {
        result.uv += vertices[i].uv * bar[i];
        result.normal += vertices[i].normal * bar[i];
    }
    result.pos = r.pos + r.dir * result.dist;
    result.color = bar;
    return result;
}

Maths::HitRecord Maths::Mesh::Intersect(Ray r, Maths::Vec2 bounds)
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

void Maths::Mesh::ApplyTransform(Maths::Vec3 pos, Maths::Quat rot, Maths::Vec3 scale, u32 index)
{
    transformedVertices[index].pos = rot * (sourceVertices[index].pos * scale) + pos;
    transformedVertices[index].normal = rot * sourceVertices[index].normal;
    transformedVertices[index].uv = sourceVertices[index].uv;
}

u32 Maths::Mesh::GetIndiceCount()
{
    return indiceCount;
}
