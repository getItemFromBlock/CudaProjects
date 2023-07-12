#include "RayTracing/Texture.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

using namespace Maths;
using namespace RayTracing;

void RayTracing::Texture::Load(const char* path)
{
	s32 comp;
	data = stbi_load(path, &resolution.x, &resolution.y, &comp, 4);
	if (!data)
	{
		std::cerr << "Unable to load texture: " << path << " : " << stbi_failure_reason() << std::endl;
	}
}

void RayTracing::Texture::Unload()
{
	if (data)
	{
		stbi_image_free(data);
		data = nullptr; // in case of Unload being called multiple times
	}
}

Vec4 Texture::Sample(Vec2 uv, bool linear)
{
	if (!data) return Vec4(1, 1, 1, 1);
	if (linear)
	{
		Vec2 texPos = Vec2(Util::Mod(uv.x * resolution.x - 0.5f, resolution.x), Util::Mod(uv.y * resolution.y - 0.5f, resolution.y));
		Vec2 delta = Vec2(texPos.x - static_cast<int>(texPos.x), texPos.y - static_cast<int>(texPos.y));
		IVec2 px1 = IVec2(texPos);
		IVec2 px2 = IVec2(Util::Mod(uv.x * resolution.x + 0.5f, resolution.x), Util::Mod(uv.y * resolution.y + 0.5f, resolution.y));
		Vec4 color11 = TexelFetch(IVec2(px1.x, px1.y));
		Vec4 color12 = TexelFetch(IVec2(px1.x, px2.y));
		Vec4 color21 = TexelFetch(IVec2(px2.x, px1.y));
		Vec4 color22 = TexelFetch(IVec2(px2.x, px2.y));
		Vec4 deltaX = color21 - color11;
		Vec4 deltaY = color12 - color11;
		Vec4 deltaXY = color11 + color22 - color21 - color12;
		return deltaX * delta.x + deltaY * delta.y + deltaXY * delta.x * delta.y + color11;
	}
	else
	{
		uv = Util::Mod(uv, 1.0f);
		IVec2 px = uv * resolution;
		return TexelFetch(px);
	}
}

Vec4 Texture::TexelFetch(IVec2 pos, u8 lod)
{
	assert(pos.x >= 0 && pos.y >= 0);
	if (lod > mipmap) lod = mipmap;
	if (lod)
	{
		pos = Maths::IVec2(pos.x >> lod, pos.y >> lod);
	}
	u64 index = (pos.y * (resolution.x >> lod) + pos.x) * 4;
	return Vec4(Color4(data[index], data[index + 1], data[index + 2], data[index + 3]));
}