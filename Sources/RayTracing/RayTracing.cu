#include "RayTracing/RayTracing.cuh"

using namespace Maths;

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

RayTracing::HitRecord RayTracing::HitTriangle(const Ray& r, Vertice* vertices, Vec2 bounds)
{
    HitRecord result;
    Vec3 A = vertices[0].pos;
    Vec3 AB = vertices[1].pos - A;
    Vec3 AC = vertices[2].pos - A;
    Vec3 bar;
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
    return result;
}

Vec3 sign(Vec3 val)
{
    for (u8 i = 0; i < 3; ++i)
    {
        val[i] = val[i] >= 0 ? 1 : -1;
    }
    return val;
}

bool all(RayTracing::BVec3 vec)
{
    return vec.x && vec.y && vec.z;
}

bool all(RayTracing::BVec2 vec)
{
    return vec.x && vec.y;
}

RayTracing::BVec2 lessThan(Vec2 a, Vec2 b)
{
    RayTracing::BVec2 res;
    res.x = a.x < b.x;
    res.y = a.y < b.y;
    return res;
}

// Took this here:
// https://jcgt.org/published/0007/03/04/paper-lowres.pdf
bool RayTracing::HitBox(Ray ray, const Box& box, Vec2 bounds)
{
    ray.pos = box.rotation * (ray.pos - box.center);
    ray.dir = box.rotation * ray.dir;

    // This "rayCanStartInBox" branch is evaluated at compile time because `const` in GLSL
    // means compile-time constant. The multiplication by 1.0 will likewise be compiled out
    // when rayCanStartInBox = false.
    Vec3 comp = Util::Abs(ray.pos) * box.invRadius;
    float winding = (Util::MaxF(Util::MaxF(comp.x, comp.y), comp.z) < 1) ? -1 : 1;

    // We'll use the negated sign of the ray direction in several places, so precompute it.
    // The sign() instruction is fast...but surprisingly not so fast that storing the result
    // temporarily isn't an advantage.
    Vec3 sgn = -sign(ray.dir);

    // Ray-plane intersection. For each pair of planes, choose the one that is front-facing
    // to the ray and compute the distance to it.
    Vec3 distanceToPlane = box.radius * winding * sgn - ray.pos;
    distanceToPlane /= ray.dir;

    // Perform all three ray-box tests and cast to 0 or 1 on each axis. 
    // Use a macro to eliminate the redundant code (no efficiency boost from doing so, of course!)
    // Could be written with 
#   define TEST(U, VW)\
         /* Is there a hit on this axis in front of the origin? Use multiplication instead of && for a small speedup */\
         (distanceToPlane.U >= 0.0) && \
         /* Is that hit within the face of the box? */\
         all(lessThan(Util::Abs(ray.pos.VW + ray.dir.VW * distanceToPlane.U), box.radius.VW))

    BVec3 test = BVec3(TEST(x, yz()), TEST(y, zx()), TEST(z, xy()));

    // CMOV chain that guarantees exactly one element of sgn is preserved and that the value has the right sign
    sgn = test.x ? Vec3(sgn.x, 0, 0) : (test.y ? Vec3(0, sgn.y, 0) : Vec3(0, 0, test.z ? sgn.z : 0));
#   undef TEST

    // At most one element of sgn is non-zero now. That element carries the negative sign of the 
    // ray direction as well. Notice that we were able to drop storage of the test vector from registers,
    // because it will never be used again.

    // Mask the distance by the non-zero axis
    // Dot product is faster than this CMOV chain, but doesn't work when distanceToPlane contains nans or infs. 
    //
    f32 distance = (sgn.x != 0.0) ? distanceToPlane.x : ((sgn.y != 0.0) ? distanceToPlane.y : distanceToPlane.z);

    return (sgn.x != 0) || (sgn.y != 0) || (sgn.z != 0);
}