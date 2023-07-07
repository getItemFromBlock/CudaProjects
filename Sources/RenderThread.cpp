#include "RenderThread.hpp"

#include <assert.h>

using namespace Maths;

void RenderThread::Init(HWND hwnIn, IVec2 resIn, bool isRealTime)
{
	res = resIn;
	hwnd = hwnIn;
	if (isRealTime)
	{
		thread = std::thread(&RenderThread::ThreadFuncRealTime, this);
	}
	else
	{

	}
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	start = now.time_since_epoch();
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

void RenderThread::ThreadFuncRealTime()
{
	kernels.InitKernels(res);
	colorBuffer.resize(res.x * res.y);
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

}