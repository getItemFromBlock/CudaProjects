#include "Maths.hpp"

namespace Maths
{
    struct HitRecord
    {
        bool hit = false;
        Vec2 UV;
        Vec3 normal;
        Vec3 point;
        Vec3 color;
        f32 dist = 0;
    };

    struct Ray
    {
        Vec3 pos;
        Vec3 dir;
    };

    struct Sphere
    {
        Vec3 pos;
        f32 radius;
        Vec3 color;
    };

    struct Triangle
    {
        Vec3 A, AB, AC;
        Vec3 normal;
        Vec3 color;
    };

    HitRecord HitSphere(Ray r, Sphere sp, f32 minimum, f32 maximum, Vec3 colorIn, bool reverse)
    {
        HitRecord result;
        f32 t0, t1;
        Vec3 L = r.pos - sp.pos;
        f32 a = r.dir.Dot();
        f32 b = 2 * r.dir.Dot(L);
        f32 c = L.Dot() - sp.radius;
        f32 discr = b * b - 4 * a * c;
        if (discr < 0)
        {
            return result;
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
        if (reverse ? (t1 > t0) : (t0 > t1))
        {
            f32 tmp = t0;
            t0 = t1;
            t1 = tmp;
        }
        if (t0 < minimum || t0 > maximum)
        {
            return result;
        }
        result.hit = true;
        result.dist = t0;
        result.point = r.pos + r.dir * t0;
        result.normal = (result.point - sp.pos).Normalize();
        if (reverse) result.normal = -result.normal;
        result.color = colorIn * sp.color;
        return result;
    }

    HitRecord HitTriangle(Ray r, Triangle tr, f32 minimum, f32 maximum, Vec3 colorIn, bool reverse)
    {
        HitRecord result;
        Vec3 pvec = r.dir.Cross(reverse ? tr.AB : tr.AC);
        f32 det = (reverse ? tr.AC : tr.AB).Dot(pvec);
        if (det < 0.00001f) return result;
        Vec3 tvec = r.pos - tr.A;
        result.UV.x = tvec.Dot(pvec);
        if (result.UV.x < 0 || result.UV.x > det) return result;
        Vec3 qvec = tvec.Cross(reverse ? tr.AC : tr.AB);
        result.UV.y = r.dir.Dot(qvec);
        if (result.UV.y < 0 || result.UV.y + result.UV.x > det) return result;
        det = 1 / det;
        result.dist = (reverse ? tr.AB : tr.AC).Dot(qvec) * det;
        if (result.dist < minimum || result.dist > maximum) return result;
        result.UV *= det;
        result.normal = reverse ? -tr.normal : tr.normal;
        result.point = r.pos + r.dir * result.dist;
        result.color = colorIn * tr.color;
        result.hit = true;
        return result;
    }
}
