#include "RenderThread.hpp"

#include <assert.h>

using namespace Maths;

void RenderThread::Init(HWND hwnIn, IVec2 resIn)
{
	hwnd = hwnIn;
	res = resIn;
	thread = std::thread(&RenderThread::ThreadFuncRealTime, this);
}

void RenderThread::Init()
{
	res = Vec2(WIDTH, HEIGHT);
	thread = std::thread(&RenderThread::ThreadFuncFrames, this);
}

void RenderThread::Resize(IVec2 newRes)
{
	while (resize.Load())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	storedRes = newRes;
	resize.Store(true);
}

bool RenderThread::HasFinished() const
{
	return exit.Load();
}

std::vector<std::vector<u32>> RenderThread::GetFrames()
{
	std::vector<std::vector<u32>> result;
	if (queueLock.Load()) return result;
	result = queuedFrames;
	queueLock.Store(true);
	return result;
}

void RenderThread::Quit()
{
	exit.Store(true);
	thread.join();
}

void RenderThread::CopyToScreen()
{
	HDC hdc = GetDC(hwnd);
	BITMAPINFO info = { 0 };
	info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	info.bmiHeader.biWidth = res.x;
	info.bmiHeader.biHeight = -res.y; // top-down image 
	info.bmiHeader.biPlanes = 1;
	info.bmiHeader.biBitCount = 32;
	info.bmiHeader.biCompression = BI_RGB;
	info.bmiHeader.biSizeImage = res.x * res.y * sizeof(u32);
	int t = SetDIBitsToDevice(hdc, 0, 0, res.x, res.y, 0, 0, 0, res.y, colorBuffer.data(), &info, DIB_RGB_COLORS);
	ReleaseDC(hwnd, hdc);
}

void RenderThread::RunKernels()
{
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch() - start;
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	f64 iTime = micros / 1000000.0;
	kernels.RunKernels(colorBuffer.data(), iTime);
}

void RenderThread::HandleResize()
{
	if (resize.Load())
	{
		if (colorBuffer.size() < 1llu * storedRes.x * storedRes.y)
		{
			colorBuffer.resize(storedRes.x * storedRes.y);
		}
		res = storedRes;
		kernels.Resize(res);
		resize.Store(false);
	}
}

void RenderThread::InitThread()
{
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	start = now.time_since_epoch();
	kernels.InitKernels(res);
	colorBuffer.resize(res.x * res.y);
}

void RenderThread::ThreadFuncRealTime()
{
	InitThread();
	while (!exit.Load())
	{
		HandleResize();
		RunKernels();
		CopyToScreen();
		//std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	kernels.ClearKernels();
}

void RenderThread::ThreadFuncFrames()
{
	InitThread();
	f64 iTime = 0;
	u64 frame = 0;
	while (iTime < LENGTH && !exit.Load())
	{
		kernels.RunKernels(colorBuffer.data(), iTime);
		bufferedFrames.push_back(colorBuffer);
		if (queueLock.Load())
		{
			queuedFrames = bufferedFrames;
			bufferedFrames.clear();
			queueLock.Store(false);
		}
		frame++;
		iTime += 1.0 / FPS;
	}
	kernels.ClearKernels();
	exit.Store(true);
}