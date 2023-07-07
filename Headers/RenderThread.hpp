#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#define HWND void*
#endif

#include <thread>
#include <vector>
#include <chrono>

#include "Types.hpp"
#include "Signal.hpp"
#include "Kernel.cuh"
#include "Maths/Maths.hpp"

#define LENGTH 34.90658503988659
/*
// 4K
#define WIDTH 3840
#define HEIGHT 2160
*/

struct Parameters
{
	Maths::IVec2 targetResolution = Maths::IVec2(1920, 1080);
	s32 targetFPS = 30;
};

class RenderThread
{
public:
	RenderThread() {};
	~RenderThread() {};

	void Init(const Parameters& params);
	void Init(HWND hwnd, Maths::IVec2 res);
	void Resize(Maths::IVec2 newRes);
	bool HasFinished() const;
	std::vector<std::vector<u32>> GetFrames();
	void Quit();
private:
	std::thread thread;
	std::chrono::system_clock::duration start = std::chrono::system_clock::duration();
	Kernel kernels;
	HWND hwnd = {};
	Core::Signal resize;
	Core::Signal exit;
	Core::Signal queueLock;
	std::vector<u32> colorBuffer;
	std::vector<std::vector<u32>> queuedFrames;
	std::vector<std::vector<u32>> bufferedFrames;
	Maths::IVec2 res;
	Maths::IVec2 storedRes;
	Parameters params;

	void ThreadFuncRealTime();
	void ThreadFuncFrames();
	void CopyToScreen();
	void RunKernels();
	void HandleResize();
	void InitThread();
};