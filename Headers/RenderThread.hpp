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
	s32 startFrame = 0;
};

struct FrameHolder
{
	u64 frameID = 0;
	std::vector<u32> frameData;
};

class RenderThread
{
public:
	RenderThread() {};
	~RenderThread() {};

	void Init(const Parameters& params, s32 id);
	void Init(HWND hwnd, Maths::IVec2 res);
	void Resize(Maths::IVec2 newRes);
	bool HasFinished() const;
	std::vector<FrameHolder> GetFrames();
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
	std::vector<FrameHolder> queuedFrames;
	std::vector<FrameHolder> bufferedFrames;
	Maths::IVec2 res;
	Maths::IVec2 storedRes;
	Parameters params;
	s32 threadID = -1;

	void ThreadFuncRealTime();
	void ThreadFuncFrames();
	void CopyToScreen();
	void RunKernels();
	void HandleResize();
	void InitThread();
};