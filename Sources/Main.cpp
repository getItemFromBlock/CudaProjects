#include <iostream>
#include <thread>
#include <filesystem>

#include "RenderThread.hpp"
#include "EncoderThread.hpp"

int main(int argc, char* argv[])
{
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
	th.Init();
	u64 current_frame = 0;
	u64 max_frames = static_cast<u64>(LENGTH * FPS);
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