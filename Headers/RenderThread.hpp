#pragma once

#include <Windows.h>
#include <thread>
#include <vector>
#include <chrono>

#include "Types.hpp"
#include "Signal.hpp"
#include "Kernel.hpp"
#include "Maths/Maths.hpp"

class RenderThread
{
public:
	RenderThread() {};
	~RenderThread() {};

	void Init(HWND hwnd, Maths::IVec2 res, bool isRealTime);
	void Resize(Maths::IVec2 newRes);
	void Quit();
private:
	std::thread thread;
	std::chrono::system_clock::duration start = std::chrono::system_clock::duration();
	Kernel kernels;
	HWND hwnd = {};
	Core::Signal resize;
	Core::Signal exit;
	std::vector<u32> colorBuffer;
	Maths::IVec2 res;
	Maths::IVec2 storedRes;

	void ThreadFuncRealTime();
	void ThreadFuncFrames();
	void CopyToScreen();
	void RunKernels();
	void HandleResize();
};