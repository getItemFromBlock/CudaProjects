#pragma once

#include <math.h>
#include <vector>
#include <string>

#ifdef JOLT_API
#include <Jolt/Jolt.h>

#include <Jolt/Math/Vec3.h>
#include <Jolt/Math/Float3.h>
#include <Jolt/Math/Float2.h>
#include <Jolt/Math/Mat44.h>
#include <Jolt/Core/Color.h>
#endif

#include "Types.hpp"

#ifdef NAT_EngineDLL
#define NAT_API __declspec(dllexport)
#else
#define NAT_API
#endif

#ifdef __CUDACC__
#define CUDA_FUNCTION __host__ __device__
#else
#define CUDA_FUNCTION
#endif

namespace Maths
{

    static const f32 VEC_COLLINEAR_PRECISION = 0.001f;
    static const f32 VEC_HIGH_VALUE = 1e38f;

    class Vec2;
    class Quat;

    class NAT_API IVec2
    {
    public:
        s32 x, y;
        CUDA_FUNCTION inline IVec2() : x(0), y(0) {}
        CUDA_FUNCTION inline IVec2(const IVec2& in) : x(in.x), y(in.y) {}
        CUDA_FUNCTION inline IVec2(const Vec2 in);
        CUDA_FUNCTION inline IVec2(const s32 a, const s32 b) : x(a), y(b) {}

        void print() const;
        const std::string toString() const;

        CUDA_FUNCTION inline s32 Dot(IVec2 a) const;
        // Return the length squared of this object
        CUDA_FUNCTION inline s32 Dot() const;

        // Return the lenght of the given Vector
        CUDA_FUNCTION inline f32 Length() const;

        CUDA_FUNCTION inline IVec2 operator+(const IVec2 a) const;

        CUDA_FUNCTION inline IVec2 operator+(const s32 a) const;

        CUDA_FUNCTION inline IVec2& operator+=(const IVec2 a);

        CUDA_FUNCTION inline IVec2& operator+=(const s32 a);

        // Return a new vector wich is the substraction of 'a' and 'b'
        CUDA_FUNCTION inline IVec2 operator-(const IVec2 a) const;

        CUDA_FUNCTION inline IVec2 operator-(const s32 a) const;

        CUDA_FUNCTION inline IVec2& operator-=(const IVec2 a);

        CUDA_FUNCTION inline IVec2& operator-=(const s32 a);

        CUDA_FUNCTION inline IVec2 operator-() const;

        // Return the result of the aritmetic multiplication of 'a' and 'b'
        CUDA_FUNCTION inline IVec2 operator*(const IVec2 a) const;

        CUDA_FUNCTION inline IVec2& operator*=(const IVec2 a);

        CUDA_FUNCTION inline IVec2 operator*(const f32 a) const;

        CUDA_FUNCTION inline IVec2& operator*=(const s32 a);

        CUDA_FUNCTION inline IVec2 operator/(const f32 b) const;

        CUDA_FUNCTION inline IVec2& operator/=(const s32 b);

        CUDA_FUNCTION inline bool operator==(const IVec2 b) const;

        CUDA_FUNCTION inline bool operator!=(const IVec2 b) const;
    };

    class NAT_API Vec2
    {
    public:
        f32 x;
        f32 y;

        // Return a new empty Vec2
        CUDA_FUNCTION inline Vec2() : x(0), y(0) {}

        // Return a new Vec2 initialised with 'a' and 'b'
        CUDA_FUNCTION inline Vec2(f32 a, f32 b) : x(a), y(b) {}

        CUDA_FUNCTION inline Vec2(f32 value) : Vec2(value, value) {}

        // Return a new Vec2 initialised with 'in'
        CUDA_FUNCTION inline Vec2(const Vec2& in) : x(in.x), y(in.y) {}
        CUDA_FUNCTION inline Vec2(const IVec2 in) : x((f32)in.x), y((f32)in.y) {}

        void print() const;
        const std::string toString() const;

        // Return a new Vec2 equivalent to Vec(1,0) rotated by 'angle' (in radians)
        CUDA_FUNCTION inline static Vec2 FromAngle(float angle);

        //Return the distance between this object and 'a'
        CUDA_FUNCTION inline f32 GetDistanceFromPoint(Vec2 a) const;

        // Return the lenght of the given Vector
        CUDA_FUNCTION inline f32 Length() const;

        CUDA_FUNCTION inline Vec2 operator+(const Vec2 a) const;
        CUDA_FUNCTION inline Vec2& operator+=(const Vec2 a);
        CUDA_FUNCTION inline Vec2 operator+(const f32 a) const;
        CUDA_FUNCTION inline Vec2& operator+=(const f32 a);

        CUDA_FUNCTION inline Vec2 operator-(const Vec2 a) const;
        CUDA_FUNCTION inline Vec2& operator-=(const Vec2 a);
        CUDA_FUNCTION inline Vec2 operator-(const f32 a) const;
        CUDA_FUNCTION inline Vec2& operator-=(const f32 a);

        CUDA_FUNCTION inline Vec2 operator-() const;

        CUDA_FUNCTION inline Vec2 operator*(const Vec2 a) const;
        CUDA_FUNCTION inline Vec2& operator*=(const Vec2 a);
        CUDA_FUNCTION inline Vec2 operator*(const f32 a) const;
        CUDA_FUNCTION inline Vec2& operator*=(const f32 a);

        CUDA_FUNCTION inline Vec2 operator/(const f32 b) const;
        CUDA_FUNCTION inline Vec2 operator/(const Vec2 b) const;
        CUDA_FUNCTION inline Vec2& operator/=(const f32 b);
        CUDA_FUNCTION inline Vec2& operator/=(const Vec2 b);

        CUDA_FUNCTION inline bool operator==(const Vec2 b) const;
        CUDA_FUNCTION inline bool operator!=(const Vec2 b) const;

        CUDA_FUNCTION inline const f32& operator[](const size_t a) const;

        CUDA_FUNCTION inline f32& operator[](const size_t a);

        // Return true if 'a' and 'b' are collinears (Precision defined by VEC_COLLINEAR_PRECISION)
        CUDA_FUNCTION inline bool IsCollinearWith(Vec2 a) const;

        CUDA_FUNCTION inline f32 Dot(Vec2 a) const;
        // Return the length squared of this object
        CUDA_FUNCTION inline f32 Dot() const;

        // Return the z component of the cross product of 'a' and 'b'
        CUDA_FUNCTION inline f32 Cross(Vec2 a) const;

        // Return a vector with the same direction that 'in', but with length 1
        CUDA_FUNCTION inline Vec2 Normalize() const;

        // Return a vector of length 'in' and with an opposite direction
        CUDA_FUNCTION inline Vec2 Negate() const;

        // Return the normal vector of 'in'
        CUDA_FUNCTION inline Vec2 GetNormal() const;

        // return true if 'a' converted to s32 is equivalent to 'in' converted to s32
        CUDA_FUNCTION inline bool IsIntEquivalent(Vec2 a) const;

        // Get the angle defined by this vector, in radians
        CUDA_FUNCTION inline f32 GetAngle() const;

        CUDA_FUNCTION inline bool IsNearlyEqual(Vec2 a, f32 prec = 1e-5f);

#ifdef IMGUI_API
        inline Vec2(const ImVec2& in) : x(in.x), y(in.y) {}

        inline operator ImVec2() const { return ImVec2(x, y); }
#endif

#ifdef JOLT_API
        inline Vec2(const JPH::Float2& in) : x(in.x), y(in.y) {}
#endif

#ifdef ASSIMP_API
        inline Vec2(const aiVector2D& in) : x(in.x), y(in.y) {}

        inline operator aiVector2D() const { return aiVector2D(x, y); }
#endif

    };

    class Vec3;

    class NAT_API IVec3
    {
    public:
        s32 x, y, z;
        CUDA_FUNCTION inline IVec3() : x(0), y(0), z(0) {}
        CUDA_FUNCTION inline IVec3(const IVec3& in) : x(in.x), y(in.y), z(in.z) {}
        CUDA_FUNCTION inline IVec3(const Vec3& in);
        CUDA_FUNCTION inline IVec3(const s32& a, const s32& b, const s32& c) : x(a), y(b), z(c) {}

        void print() const;
        const std::string toString() const;

        CUDA_FUNCTION inline s32 Dot(IVec3 a) const;
        // Return the length squared of this object
        CUDA_FUNCTION inline s32 Dot() const;

        // Return the lenght of the given Vector
        CUDA_FUNCTION inline f32 Length() const;

        CUDA_FUNCTION inline IVec3 operator+(const IVec3& a) const;
        CUDA_FUNCTION inline IVec3 operator+(const s32 a) const;
        CUDA_FUNCTION inline IVec3& operator+=(const IVec3& a);
        CUDA_FUNCTION inline IVec3& operator+=(const s32 a);

        CUDA_FUNCTION inline IVec3 operator-(const IVec3& a) const;
        CUDA_FUNCTION inline IVec3 operator-(const s32 a) const;
        CUDA_FUNCTION inline IVec3& operator-=(const IVec3& a);
        CUDA_FUNCTION inline IVec3& operator-=(const s32 a);

        CUDA_FUNCTION inline IVec3 operator*(const IVec3& a) const;
        CUDA_FUNCTION inline IVec3 operator*(const f32 a) const;
        CUDA_FUNCTION inline IVec3& operator*=(const IVec3& a);
        CUDA_FUNCTION inline IVec3& operator*=(const s32 a);

        CUDA_FUNCTION inline IVec3 operator/(const IVec3& a) const;
        CUDA_FUNCTION inline IVec3 operator/(const f32 b) const;
        CUDA_FUNCTION inline IVec3& operator/=(const IVec3& a);
        CUDA_FUNCTION inline IVec3& operator/=(const s32 a);

        CUDA_FUNCTION inline bool operator==(const IVec3& b) const;
        CUDA_FUNCTION inline bool operator!=(const IVec3& b) const;

        CUDA_FUNCTION inline const s32& operator[](const size_t a) const;

        CUDA_FUNCTION inline s32& operator[](const size_t a);
    };

    class NAT_API Vec3
    {
    public:
        f32 x;
        f32 y;
        f32 z;

        CUDA_FUNCTION inline Vec3() : x(0), y(0), z(0) {}

        CUDA_FUNCTION inline Vec3(f32 content) : x(content), y(content), z(content) {}

        CUDA_FUNCTION inline Vec3(f32 a, f32 b, f32 c) : x(a), y(b), z(c) {}

        // Return a new Vec3 initialised with 'in'
        CUDA_FUNCTION inline Vec3(const Vec3& in) : x(in.x), y(in.y), z(in.z) {}

        CUDA_FUNCTION inline Vec3(const IVec3& in) : x((f32)in.x), y((f32)in.y), z((f32)in.z) {}

        void Print() const;
        const std::string ToString() const;

        CUDA_FUNCTION inline f32 Dot() const;

        CUDA_FUNCTION inline f32 Length() const;

        CUDA_FUNCTION inline Vec3 operator+(const Vec3& a) const;
        CUDA_FUNCTION inline Vec3 operator+(const f32 a) const;
        CUDA_FUNCTION inline Vec3& operator+=(const Vec3& a);
        CUDA_FUNCTION inline Vec3& operator+=(const f32 a);

        CUDA_FUNCTION inline Vec3 operator-(const Vec3& a) const;
        CUDA_FUNCTION inline Vec3 operator-(const f32 a) const;
        CUDA_FUNCTION inline Vec3& operator-=(const Vec3& a);
        CUDA_FUNCTION inline Vec3& operator-=(const f32 a);

        CUDA_FUNCTION inline Vec3 operator-() const;

        CUDA_FUNCTION inline Vec3 operator*(const Vec3& a) const;
        CUDA_FUNCTION inline Vec3 operator*(const f32 a) const;
        CUDA_FUNCTION inline Vec3& operator*=(const Vec3& a);
        CUDA_FUNCTION inline Vec3& operator*=(const f32 a);

        CUDA_FUNCTION inline Vec3 operator/(const Vec3& b) const;
        CUDA_FUNCTION inline Vec3 operator/(const f32 a) const;
        CUDA_FUNCTION inline Vec3& operator/=(const Vec3& a);
        CUDA_FUNCTION inline Vec3& operator/=(const f32 a);

        CUDA_FUNCTION inline bool operator==(const Vec3& b) const;
        CUDA_FUNCTION inline bool operator!=(const Vec3& b) const;

        CUDA_FUNCTION inline const f32& operator[](const size_t a) const;

        CUDA_FUNCTION inline f32& operator[](const size_t a);

        CUDA_FUNCTION inline Vec3 Reflect(const Vec3& normal);

        CUDA_FUNCTION inline Vec3 Refract(const Vec3& normal, f32 ior);

        // Return tue if 'a' and 'b' are collinears (Precision defined by VEC_COLLINEAR_PRECISION)
        CUDA_FUNCTION inline bool IsCollinearWith(Vec3 a) const;

        // Return the dot product of 'a' and 'b'
        CUDA_FUNCTION inline f32 Dot(Vec3 a) const;

        // Return the z component of the cross product of 'a' and 'b'
        CUDA_FUNCTION inline Vec3 Cross(Vec3 a) const;

        // Return a vector with the same direction that 'in', but with length 1
        CUDA_FUNCTION inline Vec3 Normalize() const;

        // Return a vector of length 'in' and with an opposite direction
        CUDA_FUNCTION inline Vec3 Negate() const;

        // Found this here: https://math.stackexchange.com/q/4112622
        CUDA_FUNCTION inline Vec3 GetPerpendicular() const;

        // return true if 'a' converted to s32 is equivalent to 'in' converted to s32
        CUDA_FUNCTION inline bool IsIntEquivalent(Vec3 a) const;

        CUDA_FUNCTION inline bool IsNearlyEqual(Vec3 a, f32 prec = 1e-5f);
#ifdef JOLT_API
        inline Vec3(const JPH::Vec3Arg& in) : x(in.GetX()), y(in.GetY()), z(in.GetZ()) {}

        inline operator JPH::Vec3Arg() const { return JPH::Vec3Arg(x, y, z); }

        inline operator JPH::Float3() const { return JPH::Float3(x, y, z); }

        inline Vec3(const JPH::Float3& in) : x(in.x), y(in.y), z(in.z) {}

        inline Vec3(const JPH::ColorArg& in) : x(in.r), y(in.g), z(in.b) {}
#endif

#ifdef ASSIMP_API
        inline Vec3(const aiVector3D& in) : x(in.x), y(in.y), z(in.z) {}

        inline operator aiVector3D() const { return aiVector3D(x, y, z); }
#endif
    };

    class Vec4;

    class NAT_API Color4
    {
    public:
        u8 r;
        u8 g;
        u8 b;
        u8 a;

        CUDA_FUNCTION inline Color4() : r(0), g(0), b(0), a(0) {}
        CUDA_FUNCTION inline Color4(const f32* in);
        CUDA_FUNCTION inline Color4(const Vec4& in);
        CUDA_FUNCTION inline Color4(u8 red, u8 green, u8 blue, u8 alpha = 0xff) : r(red), g(green), b(blue), a(alpha) {}
        CUDA_FUNCTION inline Color4(u32 rgba) : r((rgba & 0xff000000) >> 24), g((rgba & 0x00ff0000) >> 16), b((rgba & 0x0000ff00) >> 8), a(rgba & 0x000000ff) {}

        CUDA_FUNCTION inline Color4 operator*(const f32 a) const;
        CUDA_FUNCTION inline Color4 operator+(const Color4& a) const;
    };

    class NAT_API Vec4
    {
    public:
        f32 x;
        f32 y;
        f32 z;
        f32 w;

        // Return a new empty Vec4
        CUDA_FUNCTION inline Vec4() : x(0), y(0), z(0), w(0) {}

        // Return a new Vec4 initialised with 'a', 'b', 'c' and 'd'
        CUDA_FUNCTION inline Vec4(f32 a, f32 b, f32 c, f32 d = 1) : x(a), y(b), z(c), w(d) {}

        // Return a new Vec4 initialised with 'in'
        CUDA_FUNCTION inline Vec4(const Vec3& in, f32 wIn = 1.0f) : x(in.x), y(in.y), z(in.z), w(wIn) {}

        // Return a new Vec4 initialised with 'in'
        CUDA_FUNCTION inline Vec4(const Vec4& in) : x(in.x), y(in.y), z(in.z), w(in.w) {}

        CUDA_FUNCTION inline Vec4(const Color4& in) : x(in.r / 255.0f), y(in.g / 255.0f), z(in.b / 255.0f), w(in.a / 255.0f) {}


        // Print the Vec4
        void print() const;
        const std::string toString() const;

        // Return the Vec3 of Vec4
        CUDA_FUNCTION inline Vec3 GetVector() const;

        // Return the length squared
        CUDA_FUNCTION inline f32 Dot() const;

        // Return the length
        CUDA_FUNCTION inline f32 Length() const;

        // Divide each components by w, or set to VEC_HIGH_VALUE if w equals 0
        CUDA_FUNCTION inline Vec4 Homogenize() const;

        CUDA_FUNCTION inline Vec4 operator+(const Vec4& a) const;
        CUDA_FUNCTION inline Vec4 operator+(const f32 a) const;
        CUDA_FUNCTION inline Vec4& operator+=(const Vec4& a);
        CUDA_FUNCTION inline Vec4& operator+=(const f32 a);

        CUDA_FUNCTION inline Vec4 operator-(const Vec4& a) const;
        CUDA_FUNCTION inline Vec4 operator-(const f32 a) const;
        CUDA_FUNCTION inline Vec4& operator-=(const Vec4& a);
        CUDA_FUNCTION inline Vec4& operator-=(const f32 a);

        CUDA_FUNCTION inline Vec4 operator-() const;

        CUDA_FUNCTION inline Vec4 operator*(const Vec4& a) const;
        CUDA_FUNCTION inline Vec4 operator*(const f32 a) const;
        CUDA_FUNCTION inline Vec4& operator*=(const Vec4& a);
        CUDA_FUNCTION inline Vec4& operator*=(const f32 a);

        CUDA_FUNCTION inline Vec4 operator/(const Vec4& b) const;
        CUDA_FUNCTION inline Vec4 operator/(const f32 a) const;
        CUDA_FUNCTION inline Vec4& operator/=(const Vec4& a);
        CUDA_FUNCTION inline Vec4& operator/=(const f32 a);

        CUDA_FUNCTION inline bool operator==(const Vec4& b) const;
        CUDA_FUNCTION inline bool operator!=(const Vec4& b) const;

        CUDA_FUNCTION inline f32& operator[](const size_t a);
        CUDA_FUNCTION inline const f32& operator[](const size_t a) const;

        // Return tue if 'a' and 'b' are collinears (Precision defined by VEC_COLLINEAR_PRECISION)
        CUDA_FUNCTION inline bool IsCollinearWith(Vec4 a) const;

        CUDA_FUNCTION inline f32 Dot(Vec4 a) const;

        // Return the z component of the cross product of 'a' and 'b'
        CUDA_FUNCTION inline Vec4 Cross(Vec4 a) const;

        // Return a vector with the same direction that 'in', but with length 1
        CUDA_FUNCTION inline Vec4 Normalize() const;

        // Return a vector of length 'in' and with an opposite direction
        CUDA_FUNCTION inline Vec4 Negate() const;

        CUDA_FUNCTION inline Vec4 Clip(const Vec4& other);

        // return true if 'a' converted to s32 is equivalent to 'in' converted to s32
        CUDA_FUNCTION inline bool IsIntEquivalent(Vec4 a) const;

        CUDA_FUNCTION inline bool IsNearlyEqual(Vec4 a, f32 prec = 1e-5f);

        CUDA_FUNCTION inline f32 GetSignedDistanceToPlane(const Vec3& point) const;

#ifdef IMGUI_API
        inline Vec4(const ImVec4& in) : x(in.x), y(in.y), z(in.z), w(in.w) {}

        inline operator ImVec4() const { return ImVec4(x, y, z, w); }
#endif

#ifdef JOLT_API
        inline Vec4(const JPH::Vec4Arg& in) : x(in.GetX()), y(in.GetY()), z(in.GetZ()), w(in.GetW()) {}

        inline operator JPH::Vec4Arg() const { return JPH::Vec4Arg(x, y, z, w); }
#endif
    };


    class Mat3;

    class NAT_API Mat4
    {
    public:
        /* data of the matrix : content[y][x]
         * Matrix is indexed with:
         *
         * 00 | 04 | 08 | 12
         * 01 | 05 | 09 | 13
         * 02 | 06 | 10 | 14
         * 03 | 07 | 11 | 15
         *
        */
        f32 content[16] = { 0 };

        CUDA_FUNCTION Mat4() {}

        CUDA_FUNCTION Mat4(f32 diagonal);

        CUDA_FUNCTION Mat4(const Mat4& in);

        CUDA_FUNCTION Mat4(const Mat3& in);

        CUDA_FUNCTION Mat4(const f32* data);

        CUDA_FUNCTION Mat4 operator*(const Mat4& a) const;

        CUDA_FUNCTION Vec4 operator*(const Vec4& a) const;

        CUDA_FUNCTION  static Mat4 Identity();

        CUDA_FUNCTION static Mat4 CreateTransformMatrix(const Vec3& position, const Vec3& rotation, const Vec3& scale);

        CUDA_FUNCTION static Mat4 CreateTransformMatrix(const Vec3& position, const Vec3& rotation);

        CUDA_FUNCTION static Mat4 CreateTransformMatrix(const Vec3& position, const Quat& rotation, const Vec3& scale);

        CUDA_FUNCTION static Mat4 CreateTransformMatrix(const Vec3& position, const Quat& rotation);

        CUDA_FUNCTION static Mat4 CreateTranslationMatrix(const Vec3& translation);

        CUDA_FUNCTION static Mat4 CreateScaleMatrix(const Vec3& scale);

        CUDA_FUNCTION static Mat4 CreateRotationMatrix(const Quat& rot);

        CUDA_FUNCTION static Mat4 CreateRotationMatrix(Vec3 angles);

        CUDA_FUNCTION static Mat4 CreateXRotationMatrix(f32 angle);

        CUDA_FUNCTION static Mat4 CreateYRotationMatrix(f32 angle);

        CUDA_FUNCTION static Mat4 CreateZRotationMatrix(f32 angle);

        // aspect ratio is width / height
        CUDA_FUNCTION static Mat4 CreatePerspectiveProjectionMatrix(f32 near, f32 far, f32 fov, f32 aspect);

        CUDA_FUNCTION static Mat4 CreateOrthoProjectionMatrix(f32 near, f32 far, f32 fov, f32 aspect);

        CUDA_FUNCTION static Mat4 CreateViewMatrix(const Vec3& position, const Vec3& focus, const Vec3& up);

        CUDA_FUNCTION static Mat4 CreateObliqueProjectionMatrix(const Mat4& projMatrix, const Vec4& nearPlane);

        CUDA_FUNCTION Mat4 InverseDepth() const;

        CUDA_FUNCTION Vec3 GetPositionFromTranslation() const;

        CUDA_FUNCTION Vec3 GetRotationFromTranslation(const Vec3& scale) const;

        CUDA_FUNCTION Vec3 GetRotationFromTranslation() const;

        CUDA_FUNCTION Vec3 GetScaleFromTranslation() const;

        CUDA_FUNCTION Mat4 TransposeMatrix();

        CUDA_FUNCTION inline f32& operator[](const size_t a);

        CUDA_FUNCTION inline const f32& operator[](const size_t a) const;

        CUDA_FUNCTION inline f32& at(const u8 x, const u8 y);
        CUDA_FUNCTION inline const f32& at(const u8 x, const u8 y) const;

        void PrintMatrix(bool raw = false);
        const std::string toString() const;

        CUDA_FUNCTION Mat4 CreateInverseMatrix() const;

        CUDA_FUNCTION Mat4 CreateAdjMatrix() const;

        CUDA_FUNCTION Mat4 GetCofactor(s32 p, s32 q, s32 n) const;

        // Recursive function for finding determinant of matrix. n is current dimension of 'in'.
        CUDA_FUNCTION f32 GetDeterminant(f32 n) const;

#ifdef JOLT_API
        CUDA_FUNCTION inline Mat4(const JPH::Mat44Arg& pMat) { pMat.StoreFloat(content); }
#endif // JOLT_API

    };

    class NAT_API Mat3
    {
    public:
        /* data of the matrix : content[y][x]
         * Matrix is indexed with:
         *
         * 00 | 03 | 06
         * 01 | 04 | 07
         * 02 | 05 | 08
         *
        */
        f32 content[9] = { 0 };

        CUDA_FUNCTION Mat3() {}

        CUDA_FUNCTION Mat3(f32 diagonal);

        CUDA_FUNCTION Mat3(const Mat3& in);

        CUDA_FUNCTION Mat3(const Mat4& in);

        CUDA_FUNCTION Mat3(const f32* data);

        CUDA_FUNCTION Mat3 operator*(const Mat3& a);

        CUDA_FUNCTION Vec3 operator*(const Vec3& a);

        CUDA_FUNCTION static Mat3 Identity();

        CUDA_FUNCTION static Mat3 CreateScaleMatrix(const Vec3& scale);

        //Angle is in degrees
        CUDA_FUNCTION static Mat3 CreateXRotationMatrix(f32 angle);

        //Angle is in degrees
        CUDA_FUNCTION static Mat3 CreateYRotationMatrix(f32 angle);

        //Angle is in degrees
        CUDA_FUNCTION static Mat3 CreateZRotationMatrix(f32 angle);

        //Angles are in degrees
        CUDA_FUNCTION static Mat3 CreateRotationMatrix(Vec3 angles);

        CUDA_FUNCTION Vec3 GetRotationFromTranslation(const Vec3& scale) const;

        CUDA_FUNCTION Vec3 GetRotationFromTranslation() const;

        CUDA_FUNCTION Mat3 TransposeMatrix();

        CUDA_FUNCTION inline f32& operator[](const size_t a);

        CUDA_FUNCTION inline const f32& operator[](const size_t a) const;

        CUDA_FUNCTION inline f32& at(const u8 x, const u8 y);

        void PrintMatrix(bool raw = false);
        const std::string toString() const;

        CUDA_FUNCTION Mat3 CreateInverseMatrix();

        CUDA_FUNCTION Mat3 CreateAdjMatrix();

        CUDA_FUNCTION Mat3 GetCofactor(s32 p, s32 q, s32 n);

        // Recursive function for finding determinant of matrix. n is current dimension of 'in'.
        CUDA_FUNCTION f32 GetDeterminant(f32 n);
    };

    class NAT_API Quat
    {
    public:
        Vec3 v;
        f32 a;

        CUDA_FUNCTION inline Quat() : v(), a(1) {}

        CUDA_FUNCTION inline Quat(Vec3 vector, f32 real) : v(vector), a(real) {}

        CUDA_FUNCTION inline Quat(const Mat3& in);

        CUDA_FUNCTION inline Quat(const Mat4& in);

        // Return the length squared
        CUDA_FUNCTION inline f32 Dot() const;

        // Return the length
        CUDA_FUNCTION inline f32 Length() const;

        CUDA_FUNCTION inline Quat Conjugate() const;

        CUDA_FUNCTION inline Quat Inverse() const;

        CUDA_FUNCTION inline Quat Normalize() const;

        CUDA_FUNCTION inline Quat NormalizeAxis() const;

        // Makes a quaternion representing a rotation in 3d space. Angle is in radians.
        CUDA_FUNCTION static Quat AxisAngle(Vec3 axis, f32 angle);

        // Makes a quaternion from Euler angles (angle order is YXZ)
        CUDA_FUNCTION static Quat FromEuler(Vec3 euler);

        CUDA_FUNCTION inline f32 GetAngle();

        CUDA_FUNCTION inline Vec3 GetAxis();

        CUDA_FUNCTION inline Mat3 GetRotationMatrix3() const;

        CUDA_FUNCTION inline Mat4 GetRotationMatrix4() const;

        CUDA_FUNCTION inline Quat operator+(const Quat& other) const;

        CUDA_FUNCTION inline Quat operator-(const Quat& other) const;

        CUDA_FUNCTION inline Quat operator-() const;

        CUDA_FUNCTION inline Quat operator*(const Quat& other) const;

        CUDA_FUNCTION inline Vec3 operator*(const Vec3& other) const;

        CUDA_FUNCTION inline Quat operator*(const f32 scalar) const;

        CUDA_FUNCTION inline Quat operator/(const Quat& other) const;

        CUDA_FUNCTION inline Quat operator/(const f32 scalar) const;

        CUDA_FUNCTION inline Vec3 GetRight() const;

        CUDA_FUNCTION inline Vec3 GetUp() const;

        CUDA_FUNCTION inline Vec3 GetFront() const;

        CUDA_FUNCTION inline static Quat Slerp(const Quat& a, Quat b, f32 alpha);
#ifdef JOLT_API
        inline Quat(const JPH::QuatArg& input) : v(input.GetX(), input.GetY(), input.GetZ()), a(input.GetW()) {}

        inline operator JPH::QuatArg() const { return JPH::QuatArg(v.x, v.y, v.z, a); }
#endif
    };

    class NAT_API Frustum
    {
    public:
        CUDA_FUNCTION Frustum() {}
        CUDA_FUNCTION ~Frustum() {}

        Vec4 top;
        Vec4 bottom;
        Vec4 right;
        Vec4 left;
        Vec4 front;
        Vec4 back;
    };

    class NAT_API AABB
    {
    public:
        CUDA_FUNCTION AABB() {}
        CUDA_FUNCTION AABB(Vec3 position, Vec3 extent) : center(position), size(extent) {}
        CUDA_FUNCTION ~AABB() {}

        Vec3 center;
        Vec3 size;

        CUDA_FUNCTION bool IsOnFrustum(const Frustum& camFrustum, const Maths::Mat4& transform) const;
        CUDA_FUNCTION bool IsOnOrForwardPlane(const Vec4& plane) const;
    };

    namespace Util
    {
        // Return the given angular value in degrees converted to radians
        CUDA_FUNCTION inline NAT_API f32 ToRadians(f32 in);

        // Return the given angular value in radians converted to degrees
        CUDA_FUNCTION inline NAT_API f32 ToDegrees(f32 in);

        CUDA_FUNCTION inline NAT_API f32 Clamp(f32 in, f32 min = 0.0f, f32 max = 1.0f);

        CUDA_FUNCTION inline NAT_API Vec2 Clamp(Vec2 in, f32 min = 0.0f, f32 max = 1.0f);

        CUDA_FUNCTION inline NAT_API Vec3 Clamp(Vec3 in, f32 min = 0.0f, f32 max = 1.0f);

        CUDA_FUNCTION inline NAT_API Vec4 Clamp(Vec4 in, f32 min = 0.0f, f32 max = 1.0f);

        CUDA_FUNCTION inline NAT_API f32 Abs(f32 in);

        CUDA_FUNCTION inline NAT_API Vec2 Abs(Vec2 in);

        CUDA_FUNCTION inline NAT_API Vec3 Abs(Vec3 in);

        CUDA_FUNCTION inline NAT_API Vec4 Abs(Vec4 in);

        CUDA_FUNCTION inline NAT_API s32 IClamp(s32 in, s32 min, s32 max);

        CUDA_FUNCTION inline NAT_API u32 UClamp(u32 in, u32 min, u32 max);

        CUDA_FUNCTION inline NAT_API f32 Lerp(f32 a, f32 b, f32 delta);

        CUDA_FUNCTION inline NAT_API Vec3 Lerp(Vec3 a, Vec3 b, f32 delta);

        CUDA_FUNCTION inline NAT_API f32 Mod(f32 in, f32 value);

        CUDA_FUNCTION inline NAT_API Vec2 Mod(Vec2 in, f32 value);

        CUDA_FUNCTION inline NAT_API Vec3 Mod(Vec3 in, f32 value);

        CUDA_FUNCTION inline NAT_API s32 IMod(s32 in, s32 value);

        CUDA_FUNCTION inline NAT_API f32 MinF(f32 a, f32 b);

        CUDA_FUNCTION inline NAT_API f32 MaxF(f32 a, f32 b);

        CUDA_FUNCTION inline NAT_API Vec3 MinV(Vec3 a, Vec3 b);

        CUDA_FUNCTION inline NAT_API Vec3 MaxV(Vec3 a, Vec3 b);

        CUDA_FUNCTION inline NAT_API s32 MinI(s32 a, s32 b);

        CUDA_FUNCTION inline NAT_API s32 MaxI(s32 a, s32 b);

        // Smooth min function
        CUDA_FUNCTION inline NAT_API f32 SMin(f32 a, f32 b, f32 delta);

        CUDA_FUNCTION inline NAT_API bool IsNear(f32 a, f32 b, f32 prec = 0.0001f);

        // Returns a string with the hex representation of number
        // TODO Test parity with big/little endian
        inline NAT_API std::string GetHex(u64 number);

        // Fills the given buffer with the hex representation of number
        // WARNING: buffer must be at least 16 char long
        // TODO Test parity with big/little endian
        inline NAT_API void GetHex(char* buffer, u64 number);

        inline NAT_API u64 ReadHex(const std::string& input);

        // Set of functions used to generate some shapes
        // TODO is this still relevant ?

        void NAT_API GenerateSphere(s32 x, s32 y, std::vector<Vec3>* PosOut, std::vector<Vec3>* NormOut, std::vector<Vec2>* UVOut);

        void NAT_API GenerateCube(std::vector<Vec3>* PosOut, std::vector<Vec3>* NormOut, std::vector<Vec2>* UVOut);

        void NAT_API GenerateDome(s32 x, s32 y, bool reversed, std::vector<Vec3>* PosOut, std::vector<Vec3>* NormOut, std::vector<Vec2>* UVOut);

        void NAT_API GenerateCylinder(s32 x, s32 y, std::vector<Vec3>* PosOut, std::vector<Vec3>* NormOut, std::vector<Vec2>* UVOut);

        void NAT_API GeneratePlane(std::vector<Vec3>* PosOut, std::vector<Vec3>* NormOut, std::vector<Vec2>* UVOut);

        void NAT_API GenerateSkyPlane(std::vector<Vec3>* PosOut, std::vector<Vec3>* NormOut, std::vector<Vec2>* UVOut);

        Vec3 NAT_API GetSphericalCoord(f32 longitude, f32 latitude);
    };
}

#include "Maths.inl"