#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#define HWND void*
#endif

#include <thread>
#include <vector>
#include <chrono>
#include <mutex>
#include <bitset>

#include "Types.hpp"
#include "Signal.hpp"
#include "Kernel.cuh"
#include "Maths/Maths.cuh"

#define LENGTH 34.90658503988659
#define LENGTH2 0.0001
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
	s32 quality = 16;
};

struct FrameHolder
{
	u64 frameID = 0;
	std::vector<u32> frameData;
};

namespace Resources
{
	class Texture;
	class Cubemap;
	class Material;
	class Mesh;
}

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
	f32 GetElapsedTime();
	void MoveMouse(Maths::Vec2 delta);
	void SetKeyState(u8 key, bool state);
	void ToggleKeyState(u8 key);
private:
	std::thread thread;
	std::chrono::system_clock::duration start = std::chrono::system_clock::duration();
	Kernel kernels;
	HWND hwnd = {};
	Core::Signal resize;
	Core::Signal exit;
	Core::Signal queueLock;
	std::mutex mouseLock;
	std::mutex keyLock;
	std::vector<Resources::Texture> textures;
	std::vector<Resources::Cubemap> cubemaps;
	std::vector<Resources::Material> materials;
	std::vector<Resources::Mesh> meshes;
	std::vector<u32> colorBuffer;
	std::vector<FrameHolder> queuedFrames;
	std::vector<FrameHolder> bufferedFrames;
	std::bitset<14> keys = 0;
	Maths::IVec2 res;
	Maths::IVec2 storedRes;
	Maths::Vec2 storedDelta;
	Maths::Vec3 position = Maths::Vec3(-5.30251f, 6.38824f, -7.8891f);
	Maths::Vec2 rotation = Maths::Vec2(static_cast<f32>(M_PI_2) - 1.059891f, 0.584459f);
	f32 fov = 3.55f;
	Parameters params;
	s32 threadID = -1;
	f32 elapsedTime = 0;
	f32 denoiseStrength = 0.2f;

	void MandelbrotRealTime();
	void MandelbrotFrames();
	void RayTracingRealTime();
	void RayTracingFrames();
	void LoadAssets();
	void UnloadAssets();
	void CopyToScreen();
	void RunKernels();
	void HandleResize();
	void InitThread();
};