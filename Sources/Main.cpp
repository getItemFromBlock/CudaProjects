#include <iostream>
#include <thread>
#include <filesystem>
#include <sstream>

#include "RenderThread.hpp"
#include "EncoderThread.hpp"
#include "Maths/Maths.hpp"

bool str2int(s32& i, char const* s)
{
	char c;
	std::stringstream ss(s);
	ss >> i;
	if (ss.fail() || ss.get(c) || i <= 0)
	{
		return false;
	}
	return true;
}

void ParseArgs(int argc, char* argv[], Parameters& params)
{
	for (s32 i = 1; i < argc-1; ++i)
	{
		if (argv[i][0] != '-' && argv[i][0] != '/') continue;
		switch (argv[i][1])
		{
		case 'w':
			if (!str2int(params.targetResolution.x, argv[i+1]))
			{
				params.targetResolution.x = 1920;
				std::cerr << "Invalid number " << argv[i+1] << std::endl;
			}
			++i;
			break;
		case 'h':
			if (!str2int(params.targetResolution.y, argv[i+1]))
			{
				params.targetResolution.x = 1080;
				std::cerr << "Invalid number " << argv[i + 1] << std::endl;
			}
			++i;
			break;
		case 'r':
			if (!str2int(params.targetFPS, argv[i + 1]))
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

int main(int argc, char* argv[])
{
	Parameters params;
	ParseArgs(argc, argv, params);
	std::filesystem::path p = std::filesystem::current_path().append("output");
	if (!std::filesystem::exists(p))
	{
		std::filesystem::create_directory(p);
	}
	else
	{
		for (const auto& entry : std::filesystem::directory_iterator(p))
		{
			std::filesystem::remove_all(entry.path());
		}
	}
	u32 threadCount = std::thread::hardware_concurrency();
	std::cout << "Using " << threadCount << " threads for encoding" << std::endl;
	RenderThread th;
	EncoderThread* threadPool = new EncoderThread[threadCount];
	for (u32 i = 0; i < threadCount; ++i)
	{
		threadPool[i].Init(params);
	}
	th.Init(params);
	u64 current_frame = 0;
	u64 max_frames = static_cast<u64>(LENGTH * params.targetFPS);
	while (!th.HasFinished())
	{
		auto frames = th.GetFrames();
		for (auto& frame : frames)
		{
			bool assigned = false;
			while (!assigned)
			{
				for (u32 i = 0; i < threadCount; ++i)
				{
					if (threadPool[i].IsAvailable())
					{
						threadPool[i].AssignFrame(current_frame, frame);
						std::cout << "Encoding frame " << current_frame << " out of " << max_frames << std::endl;
						assigned = true;
						break;
					}
				}
				if (assigned) break;
				std::cout << "No more thread available !" << std::endl;
				std::this_thread::sleep_for(std::chrono::milliseconds(1000));
			}
			current_frame++;
		}
	}
	th.Quit();
	delete[] threadPool;
	std::cout << "All done, frames are located here: " << p << std::endl;
}