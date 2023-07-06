#pragma once

#include <Windows.h>
#include <thread>
#include <vector>
#include <chrono>

#include "Types.hpp"
#include "Signal.hpp"
#include "Kernel.hpp"

class RenderThread
{
public:
	RenderThread() {};
	~RenderThread() {};

	void Init(HWND hwnd, u32 width, u32 height);
	void Resize(u32 newWidth, u32 newHeight);
	void Quit();
private:
	std::thread thread;
	std::chrono::system_clock::duration start = std::chrono::system_clock::duration();
	Kernel kernels;
	HWND hwnd = {};
	Core::Signal resize;
	Core::Signal exit;
	std::vector<u32> colorBuffer;
	u32 width = 0;
	u32 height = 0;
	u32 storedWidth = 0;
	u32 storedHeight = 0;

	void ThreadFunc();
	void CopyToScreen();
	void RunKernels();
	void HandleResize();
};