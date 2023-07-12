#pragma once

#include "Maths/Maths.hpp"

namespace RayTracing
{
    struct Vertice
    {
        Maths::Vec3 pos;
        Maths::Vec3 normal;
        Maths::Vec2 uv;
    };

    struct HitRecord
    {
        Maths::Vec3 normal;
        Maths::Vec3 pos;
        Maths::Vec2 uv;
        f32 dist = -1;
    };

    struct Ray
    {
        Maths::Vec3 pos;
        Maths::Vec3 dir;
    };

    struct Sphere
    {
        Maths::Vec3 pos;
        f32 radius = 0;
    };

    bool HitSphere(Ray r, Sphere sp, Maths::Vec2 bounds);

    HitRecord HitTriangle(Ray r, Vertice* vertices, Maths::Vec2 bounds);
}
