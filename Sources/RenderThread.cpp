#include "RenderThread.hpp"

#include <filesystem>
#include <time.h>
#include <assert.h>

#include "Resources/Texture.cuh"
#include "Resources/Mesh.cuh"
#include "Resources/Material.hpp"
#include "Resources/ModelLoader.hpp"

using namespace Maths;
using namespace Resources;
using namespace Compute;


std::string GetTime()
{
	time_t timeObj;
	time(&timeObj);
	tm* pTime = gmtime(&timeObj);
	char buffer[256];
	sprintf(buffer, "%d-%d-%d_%d-%d-%d", pTime->tm_year+1900, pTime->tm_mon+1, pTime->tm_mday, pTime->tm_hour, pTime->tm_min, pTime->tm_sec);
	return std::string(buffer);
}

void RenderThread::Init(HWND hwnIn, IVec2 resIn)
{
	hwnd = hwnIn;
	res = resIn;
#ifdef RAY_TRACING
	thread = std::thread(&RenderThread::RayTracingRealTime, this);
#else
	thread = std::thread(&RenderThread::MandelbrotRealTime, this);
#endif
}

void RenderThread::Init(const Parameters& paramsIn, s32 id)
{
	params = paramsIn;
	threadID = id;
	res = IVec2(params.targetResolution.x, params.targetResolution.y);
#ifdef RAY_TRACING
	thread = std::thread(&RenderThread::RayTracingFrames, this);
#else
	thread = std::thread(&RenderThread::MandelbrotFrames, this);
#endif
}

void RenderThread::Resize(IVec2 newRes)
{
	while (resize.Load())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
	storedRes = IVec2(Util::MaxI(newRes.x, 32), Util::MaxI(newRes.y, 32));
	resize.Store(true);
}

bool RenderThread::HasFinished() const
{
	return exit.Load();
}

std::vector<FrameHolder> RenderThread::GetFrames()
{
	std::vector<FrameHolder> result;
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

f32 RenderThread::GetElapsedTime()
{
	return elapsedTime;
}

void RenderThread::MoveMouse(Vec2 delta)
{
	mouseLock.lock();
	storedDelta -= delta;
	mouseLock.unlock();
}

void RenderThread::SetKeyState(u8 key, bool state)
{
	keyLock.lock();
	keys.set(key, state);
	keyLock.unlock();
}

void RenderThread::ToggleKeyState(u8 key)
{
	keyLock.lock();
	keys.flip(key);
	keyLock.unlock();
}

void RenderThread::CopyToScreen()
{
#ifdef _WIN32
	HDC hdc = GetDC(hwnd);
	BITMAPINFO info = { 0 };
	info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	info.bmiHeader.biWidth = res.x;
	info.bmiHeader.biHeight = -res.y; // top-down image 
	info.bmiHeader.biPlanes = 1;
	info.bmiHeader.biBitCount = 32;
	info.bmiHeader.biCompression = BI_RGB;
	info.bmiHeader.biSizeImage = sizeof(u32) * res.x * res.y;
	int t = SetDIBitsToDevice(hdc, 0, 0, res.x, res.y, 0, 0, 0, res.y, colorBuffer.data(), &info, DIB_RGB_COLORS);
	ReleaseDC(hwnd, hdc);
#endif
}

void RenderThread::RunKernels()
{
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch() - start;
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	f64 iTime = micros / 1000000.0;
	kernels.RunFractalKernels(colorBuffer.data(), iTime);
}

void RenderThread::HandleResize()
{
	if (resize.Load())
	{
		if (colorBuffer.size() < static_cast<u64>(storedRes.x) * storedRes.y)
		{
			colorBuffer.resize(static_cast<u64>(storedRes.x) * storedRes.y);
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
	kernels.InitKernels(res, threadID);
	colorBuffer.resize(static_cast<u64>(res.x) * res.y);
}

void RenderThread::MandelbrotRealTime()
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

void RenderThread::MandelbrotFrames()
{
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	auto start = now.time_since_epoch();
	InitThread();
	u64 frame = static_cast<u64>(params.startFrame) + threadID;
	f64 iTime = 1.0 / params.targetFPS * frame;
	const s32 count = CudaUtil::GetDevicesCount();
	while (iTime < LENGTH && !exit.Load())
	{
		kernels.RunFractalKernels(colorBuffer.data(), iTime);
		FrameHolder fr;
		fr.frameData = colorBuffer;
		fr.frameID = frame;
		bufferedFrames.push_back(fr);
		if (queueLock.Load())
		{
			queuedFrames = bufferedFrames;
			bufferedFrames.clear();
			queueLock.Store(false);
		}
		frame += count;
		iTime += 1.0 * count / params.targetFPS;
	}
	if (!bufferedFrames.empty())
	{
		while (!queueLock.Load())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		queuedFrames = bufferedFrames;
		bufferedFrames.clear();
		queueLock.Store(false);
	}
	kernels.ClearKernels();
	now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch() - start;
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	elapsedTime = micros / 1000000.0f;
	exit.Store(true);
}

void RenderThread::RayTracingRealTime()
{
	InitThread();
	LoadAssets();

	f64 last = 0;
	while (!exit.Load())
	{
		std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
		auto duration = now.time_since_epoch() - start;
		auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
		f64 iTime = micros / 1000000.0;
		f32 deltaTime = static_cast<f32>(iTime - last);
		last = iTime;

		mouseLock.lock();
		Vec2 delta = storedDelta;
		storedDelta = Vec2();
		mouseLock.unlock();
		delta *= 0.005f;
		rotation.x = Util::Clamp(rotation.x - delta.y, static_cast<f32>(-M_PI_2), static_cast<f32>(M_PI_2));
		rotation.y = Util::Mod(rotation.y + delta.x, static_cast<f32>(2 * M_PI));
		Maths::Vec3 dir;
		keyLock.lock();
		for (u8 i = 0; i < 6; ++i)
		{
			dir[i % 3] += (i > 2) ? -static_cast<f32>(keys.test(i)) : static_cast<f32>(keys.test(i));
		}
		f32 fovDir = static_cast<f32>(keys.test(6)) - static_cast<f32>(keys.test(7));
		LaunchParams params = keys.test(9) ? BOXDEBUG : (keys.test(8) ? ADVANCED : NONE);
		if (keys.test(10)) params = (LaunchParams)(params | DENOISE);
		if (keys.test(12))
		{
			keys.reset(12);
			denoiseStrength += 0.05f;
		}
		if (keys.test(13))
		{
			keys.reset(13);
			denoiseStrength -= 0.05f;
		}
		keyLock.unlock();
		fov = Util::Clamp(fov + fovDir * deltaTime * fov, 0.5f, 100.0f);
		Quat q = Quat::FromEuler(Vec3(rotation.x, rotation.y, 0.0f));
		if (dir.Dot())
		{
			dir = dir.Normalize() * deltaTime * 10;
			position += q * dir;
		}
		HandleResize();
		kernels.RenderMeshes(colorBuffer.data(), static_cast<u32>(meshes.size()), position, q * Vec3(0,0,1), q * Vec3(0,1,0), fov, 1, denoiseStrength, params);
		CopyToScreen();
		if (keys.test(11))
		{
			if (!std::filesystem::exists("Screenshots"))
			{
				std::filesystem::create_directory("Screenshots");
			}
			std::string name = "Screenshots/";
			name += GetTime();
			CudaUtil::SaveFrameBuffer(kernels.GetMainFrameBuffer(), name);
			keys.reset(11);
		}
	}
	UnloadAssets();
}

void RenderThread::RayTracingFrames()
{
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	auto start = now.time_since_epoch();
	InitThread();
	
	LoadAssets();

	u64 frame = static_cast<u64>(params.startFrame) + threadID;
	f64 iTime = 1.0 / params.targetFPS * frame;
	const s32 count = CudaUtil::GetDevicesCount();
	while (iTime < LENGTH2 && !exit.Load())
	{
		Quat q = Quat::FromEuler(Vec3(rotation.x, rotation.y, 0.0f));
		kernels.RenderMeshes(colorBuffer.data(), static_cast<u32>(meshes.size()), position, q * Vec3(0, 0, 1), q * Vec3(0, 1, 0), fov, params.quality, 0.2f, static_cast<LaunchParams>(ADVANCED | INVERTED_RB));

		FrameHolder fr;
		fr.frameData = colorBuffer;
		fr.frameID = frame;
		bufferedFrames.push_back(fr);
		if (queueLock.Load())
		{
			queuedFrames = bufferedFrames;
			bufferedFrames.clear();
			queueLock.Store(false);
		}
		frame += count;
		iTime += 1.0 * count / params.targetFPS;
	}
	if (!bufferedFrames.empty())
	{
		while (!queueLock.Load())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
		queuedFrames = bufferedFrames;
		bufferedFrames.clear();
		queueLock.Store(false);
	}

	UnloadAssets();

	now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch() - start;
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	elapsedTime = micros / 1000000.0f;
	exit.Store(true);
}

void RenderThread::LoadAssets()
{
	ModelLoader::LoadModel(meshes, materials, textures, "Assets/Scenes/scene2.obj");
	ModelLoader::LoadCubemap(cubemaps, "Assets/Cubemaps/sky_space.cbm");

	kernels.LoadTextures(textures);
	kernels.LoadCubemaps(cubemaps);
	kernels.LoadMaterials(materials);
	kernels.LoadMeshes(meshes);

	for (u32 i = 0; i < meshes.size(); ++i)
	{
		kernels.UpdateMeshVertices(&meshes[i], i, Vec3(0, 0, 0), Quat(), Vec3(1));
	}
	kernels.Synchronize();
}

void RenderThread::UnloadAssets()
{
	kernels.UnloadTextures(textures);
	kernels.UnloadCubemaps(cubemaps);
	kernels.UnloadMaterials();
	kernels.UnloadMeshes(meshes);
	kernels.ClearKernels();
}