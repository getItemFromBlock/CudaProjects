#include <iostream>
#include <thread>
#include <filesystem>
#include <sstream>
#include <chrono>
#include <string.h>

#include "RenderThread.hpp"
#include "EncoderThread.hpp"
#include "Maths/Maths.cuh"

struct Resolution
{
	const char* const name = nullptr;
	const Maths::IVec2 value;
};

const Resolution resolutions[] =
{
	{"hd",  Maths::IVec2(1280, 720)},
	{"fhd", Maths::IVec2(1920,1080)},
	{"2k",  Maths::IVec2(2560,1440)},
	{"4k",  Maths::IVec2(3840,2160)},
	{"8k",  Maths::IVec2(7680,4320)}
};

std::filesystem::path output = std::filesystem::current_path().append("output");

bool ReadInteger(s32& i, char const* s, bool extended)
{
	char c;
	std::stringstream ss(s);
	ss >> i;
	if (ss.fail() || (ss.get(c) && extended && c != 'x' && c != 'X') || i <= 0)
	{
		return false;
	}
	return true;
}

bool ReadResolution(Maths::IVec2& i, char const* s)
{
	if (!s || !*s)
	{
		std::cerr << "Invalid argument" << std::endl;
		return false;
	}
	for (auto& res : resolutions)
	{
		if (!strcmp(s, res.name))
		{
			i = res.value;
			return true;
		}
	}
	if (!ReadInteger(i.x, s, true))
	{
		std::cerr << "Invalid argument " << s << std::endl;
		return false;
	}
	for (auto c = s; *c != 0; ++c)
	{
		if (*c == 'x' || *c == 'X')
		{
			++c;
			if (!ReadInteger(i.y, c, false))
			{
				std::cerr << "Invalid argument " << s << std::endl;
				return false;
			}
			return true;
		}
	}
	std::cerr << "Invalid argument " << s << std::endl;
	return false;
}

void ParseArgs(int argc, char* argv[], Parameters& params)
{
	for (s32 i = 1; i < argc-1; ++i)
	{
		if (argv[i][0] != '-' && argv[i][0] != '/') continue;
		switch (argv[i][1])
		{
		case 's':
			if (!ReadInteger(params.startFrame, argv[i + 1], false))
			{
				params.startFrame = 0;
				if (strcmp(argv[i + 1], "0"))
				{
					std::cerr << "Invalid number " << argv[i + 1] << std::endl;
				}
			}
			else
			{
				std::cout << "Starting at frame " << params.startFrame << std::endl;
			}
			++i;
			break;
		case 'r':
			if (!ReadResolution(params.targetResolution, argv[i+1]))
			{
				params.targetResolution = Maths::IVec2(1920,1080);
			}
			++i;
			break;
		case 'f':
			if (!ReadInteger(params.targetFPS, argv[i + 1], false))
			{
				params.targetFPS = 30;
				std::cerr << "Invalid number " << argv[i + 1] << std::endl;
			}
			++i;
			break;
		default:
			break;
		}
	}
}

bool HasFinished(RenderThread* threads, u32 count)
{
	for (u32 i = 0; i < count; ++i)
	{
		if (!threads[i].HasFinished()) return false;
	}
	return true;
}

int main(int argc, char* argv[])
{
	Parameters params;
	ParseArgs(argc, argv, params);
	if (!std::filesystem::exists(output))
	{
		std::filesystem::create_directory(output);
	}
	else if (params.startFrame == 0)
	{
		for (const auto& entry : std::filesystem::directory_iterator(output))
		{
			std::filesystem::remove_all(entry.path());
		}
	}
	const u32 threadCount = std::thread::hardware_concurrency() > 1 ? std::thread::hardware_concurrency() - 1 : std::thread::hardware_concurrency();
	std::cout << "Using " << threadCount << " threads for encoding" << std::endl;
	EncoderThread* threadPool = new EncoderThread[threadCount];
	for (u32 i = 0; i < threadCount; ++i)
	{
		threadPool[i].Init(params);
	}
	std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now();
	auto start = now.time_since_epoch();
	const s32 deviceCount = CudaUtil::GetDevicesCount();
	CudaUtil::PrintDevicesName();
	RenderThread* deviceThreadPool = new RenderThread[deviceCount];
	for (s32 i = 0; i < deviceCount; ++i)
	{
		deviceThreadPool[i].Init(params, i, false);
	}
	std::cout << std::endl;
	u64 max_frames = static_cast<u64>(LENGTH * params.targetFPS);
	while (true)
	{
		std::vector<FrameHolder> frames;
		bool finished = HasFinished(deviceThreadPool, deviceCount);
		for (s32 d = 0; d < deviceCount; ++d)
		{
			auto data = deviceThreadPool[d].GetFrames();
			if (data.empty()) continue;
			frames.insert(frames.end(), data.begin(), data.end());
		}
		if (frames.empty() && finished)
		{
			break;
		}
		for (auto& frame : frames)
		{
			bool assigned = false;
			while (!assigned)
			{
				u32 selected = 0;
				u32 busy = 0;
				for (u32 i = 0; i < threadCount; ++i)
				{
					if (threadPool[i].IsAvailable())
					{
						if (assigned) continue;
						selected = i;
						assigned = true;
					}
					else
					{
						++busy;
					}
				}
				if (assigned)
				{
					threadPool[selected].AssignFrame(frame.frameID, frame.frameData);
					std::cout << "Encoding frame " << frame.frameID << " out of " << max_frames << " - ThreadPool usage : " << busy + 1 << "/" << threadCount << "               " << '\r';
					break;
				}
				std::cout << "No more thread available !" << "                                          " << '\r';
				std::this_thread::sleep_for(std::chrono::milliseconds(10));
			}
		}
	}
	for (s32 d = 0; d < deviceCount; ++d)
	{
		deviceThreadPool[d].Quit();
		std::cout << "GPU " << d << " time: " << deviceThreadPool[d].GetElapsedTime() << "                                                         " << std::endl;
	}
	delete[] deviceThreadPool;
	for (u32 d = 0; d < threadCount; ++d)
	{
		threadPool[d].Quit();
	}
	delete[] threadPool;
	now = std::chrono::system_clock::now();
	auto duration = now.time_since_epoch() - start;
	auto micros = std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
	f32 iTime = micros / 1000000.0f;
	std::cout << "Done in " << iTime << " seconds; Frames are located here : " << output << "               " << std::endl;
}