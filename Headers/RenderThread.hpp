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

#define FPS 30
#define LENGTH 34.90658503988659
/*
// 4K
#define WIDTH 3840
#define HEIGHT 2160
*/
#define WIDTH 1920
#define HEIGHT 1080

class RenderThread
{
public:
	RenderThread() {};
	~RenderThread() {};

	void Init();
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

	void ThreadFuncRealTime();
	void ThreadFuncFrames();
	void CopyToScreen();
	void RunKernels();
	void HandleResize();
	void InitThread();
};