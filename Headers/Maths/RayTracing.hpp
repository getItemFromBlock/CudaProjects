#include "Maths.hpp"

#include <vector>

namespace Maths
{
    struct Vertice
    {
        Maths::Vec3 pos;
        Maths::Vec3 normal;
        Maths::Vec2 uv;
    };

    struct HitRecord
    {
        Vec3 normal;
        Vec3 color;
        Vec3 pos;
        Vec2 uv;
        f32 dist = -1;
    };

    struct Ray
    {
        Vec3 pos;
        Vec3 dir;
    };

    struct Sphere
    {
        Vec3 pos;
        f32 radius = 0;
    };

    bool HitSphere(Ray r, Sphere sp, Vec2 bounds);

    HitRecord HitTriangle(Ray r, Vertice* vertices, Maths::Vec2 bounds);

    class Mesh
    {
    public:
        Mesh() {};
        ~Mesh() {};

        HitRecord Intersect(Ray r, Maths::Vec2 bounds);
        void ApplyTransform(Maths::Vec3 pos, Maths::Quat rot, Maths::Vec3 scale, u32 index);
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
