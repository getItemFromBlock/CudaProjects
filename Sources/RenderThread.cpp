#include "RenderThread.hpp"

#include <assert.h>

#include "Kernel.hpp"

void RenderThread::Init(HWND hwnIn, u32 w, u32 h)
{
	width = w;
	height = h;
	hwnd = hwnIn;
	kernels.InitKernels(w, h);
	thread = std::thread(&RenderThread::ThreadFunc, this);
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	start = now.time_since_epoch();
}

void RenderThread::Resize(u32 newWidth, u32 newHeight)
{
	while (resize.Load())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	storedWidth = newWidth;
	storedHeight = newHeight;
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
	info.bmiHeader.biWidth = width;
	info.bmiHeader.biHeight = -(s32)(height); // top-down image 
	info.bmiHeader.biPlanes = 1;
	info.bmiHeader.biBitCount = 32;
	info.bmiHeader.biCompression = BI_RGB;
	info.bmiHeader.biSizeImage = width * height * sizeof(u32);
	int t = SetDIBitsToDevice(hdc, 0, 0, width, height, 0, 0, 0, height, colorBuffer.data(), &info, DIB_RGB_COLORS);
	ReleaseDC(hwnd, hdc);
}

void RenderThread::RunKernels()
{
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch() - start;
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	f64 iTime = micros / 1000000.0;
	kernels.RunKernels(colorBuffer.data(), storedWidth, storedHeight, iTime);
}

void RenderThread::HandleResize()
{
	if (resize.Load())
	{
		if (colorBuffer.size() < 1llu * storedHeight * storedWidth)
		{
			colorBuffer.resize(storedHeight * storedWidth);
		}
		width = storedWidth;
		height = storedHeight;
		kernels.Resize(width, height);
		resize.Store(false);
	}
}

void RenderThread::ThreadFunc()
{
	colorBuffer.resize(width*height);
	while (!exit.Load())
	{
		HandleResize();
		RunKernels();
		CopyToScreen();
		//std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	kernels.ClearKernels();
}
