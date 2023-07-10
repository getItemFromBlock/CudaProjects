#pragma once

#include <vector>
#include <string>
#include "CudaUtil.hpp"
#include "Maths/Maths.hpp"

class Kernel
{
public:
	Kernel() {}
	~Kernel() {}
	void InitKernels(Maths::IVec2 res, s32 deviceID);
	void Resize(Maths::IVec2 newRes);
	void ClearKernels();
	void RunKernels(u32* img, f64 iTime);
	void DrawText(u32* img, const std::string& text, Maths::IVec2 pos, u32 size);
	void SetFont(const u8* imgData, Maths::IVec2 res);
private:
	u32* dev_img = nullptr;
	Maths::IVec2 res;
	s32 deviceID = 0;
};