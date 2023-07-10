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
private:
	u32* dev_img = nullptr;
	Maths::IVec2 res;
	s32 deviceID = 0;
};