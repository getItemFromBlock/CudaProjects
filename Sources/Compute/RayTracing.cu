#include "Compute/RayTracing.cuh"
#include "Resources/Material.hpp"

using namespace Maths;
using namespace Compute;

bool RayTracing::HitSphere(const Ray& r, const Sphere& sp, Vec2 bounds)
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

HitRecord RayTracing::HitTriangle(const Ray& r, Vec3 vertices[3], Vec2 bounds)
{
    HitRecord result;
    Vec3 A = vertices[0];
    Vec3 AB = vertices[1] - A;
    Vec3 AC = vertices[2] - A;
    Vec3 pvec = r.dir.Cross(AC);
    f32 det = AB.Dot(pvec);
    if (det < 0.00001f) return result;
    Vec3 tvec = r.pos - A;
    result.barycentric.x = tvec.Dot(pvec);
    if (result.barycentric.x < 0 || result.barycentric.x > det) return result;
    Vec3 qvec = tvec.Cross(AB);
    result.barycentric.y = r.dir.Dot(qvec);
    if (result.barycentric.y < 0 || result.barycentric.y + result.barycentric.x > det) return result;
    det = 1 / det;
    result.dist = AC.Dot(qvec) * det;
    if (result.dist < bounds.x || result.dist > bounds.y)
    {
        result.dist = -1;
        return result;
    }
    result.barycentric *= det;
    result.barycentric.z = 1 - (result.barycentric.x + result.barycentric.y);
    result.pos = r.pos + r.dir * result.dist;
    return result;
}

__host__ __device__ Vec3 sign(Vec3 val)
{
    for (u8 i = 0; i < 3; ++i)
    {
        val[i] = val[i] >= 0 ? 1.0f : -1.0f;
    }
    return val;
}

__host__ __device__ bool all(Compute::BVec3 vec)
{
    return vec.x && vec.y && vec.z;
}

__host__ __device__ bool all(Compute::BVec2 vec)
{
    return vec.x && vec.y;
}

__host__ __device__ Compute::BVec2 lessThan(Vec2 a, Vec2 b)
{
    Compute::BVec2 res;
    res.x = a.x < b.x;
    res.y = a.y < b.y;
    return res;
}

__host__ __device__ Vec2 yz(const Vec3& in)
{
    return Vec2(in.y, in.z);
}

__host__ __device__ Vec2 zx(const Vec3& in)
{
    return Vec2(in.z, in.x);
}

__host__ __device__ Vec2 xy(const Vec3& in)
{
    return Vec2(in.x, in.y);
}

// Adapted this from here:
// https://jcgt.org/published/0007/03/04/paper-lowres.pdf
f32 RayTracing::HitBox(Ray ray, const Box& box, Vec2 bounds)
{
    ray.pos = box.rotation * (ray.pos - box.center);
    ray.dir = box.rotation * ray.dir;

    Vec3 comp = Util::Abs(ray.pos) * box.invRadius;
    if (Util::MaxF(Util::MaxF(comp.x, comp.y), comp.z) <= 1) return true;
    Vec3 sgn = -sign(ray.dir);

    Vec3 distanceToPlane = box.radius * sgn - ray.pos;
    distanceToPlane /= ray.dir;

#   define TEST(U, VW)\
         (distanceToPlane.U >= 0.0) && \
         all(lessThan(Util::Abs(VW(ray.pos) + VW(ray.dir) * distanceToPlane.U), VW(box.radius)))

    BVec3 test = BVec3(TEST(x, yz), TEST(y, zx), TEST(z, xy));

    sgn = test.x ? Vec3(sgn.x, 0, 0) : (test.y ? Vec3(0, sgn.y, 0) : Vec3(0, 0, test.z ? sgn.z : 0));
#   undef TEST

    f32 distance = (sgn.x != 0.0) ? distanceToPlane.x : ((sgn.y != 0.0) ? distanceToPlane.y : (sgn.z != 0 ? distanceToPlane.z : INFINITY));
    if (distance < bounds.x || distance > bounds.y) distance = INFINITY;
    return distance;
}

void Compute::RayTracing::ApplyMaterialDisplacement(Ray& r, const Resources::Material* mat)
{
    r.pos += mat->posDisplacement;
    r.dir = mat->rotDisplacement * r.dir;
}
