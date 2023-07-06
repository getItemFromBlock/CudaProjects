#pragma once

#include <vector>
#include <string>
#include "CudaUtil.hpp"
#include "Types.hpp"

class Kernel
{
public:
	Kernel() {}
	~Kernel() {}
	void InitKernels(u32 w, u32 h);
	void Resize(u32 w, u32 h);
	void ClearKernels();
	void RunKernels(u32* img, u32 w, u32 h, f64 iTime);
	void DrawText(u32* img, u32 w, u32 h, const std::string& text, u32 x, u32 y, u32 size);
	void SetFont(const u8* imgData, u32 w, u32 h);
private:
	CudaUtil utils;
	u32* dev_img = nullptr;
	u32 width = 0;
	u32 height = 0;
};