#include "EncoderThread.hpp"

#include <chrono>
#include <assert.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "RenderThread.hpp"

EncoderThread::EncoderThread()
{
	Init();
}

EncoderThread::~EncoderThread()
{
	Quit();
}

void EncoderThread::Init()
{
	thread = std::thread(&EncoderThread::ThreadFunc, this);
}

void EncoderThread::AssignFrame(u64 frameID, const std::vector<u32>& frameData)
{
	assert(!running.Load() && !exit.Load());
	id = frameID;
	data = frameData;
	running.Store(true);
}

u64 EncoderThread::GetFrameID() const
{
	return id;
}

bool EncoderThread::IsAvailable() const
{
	return !running.Load();
}

void EncoderThread::Quit()
{
	if (exit.Load()) return;
	exit.Store(true);
	thread.join();
}

void EncoderThread::ThreadFunc()
{
	char fileName[] = "output/frame_0000000.jpg";
	while (!exit.Load())
	{
		if (!running.Load())
		{
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
			continue;
		}
		s64 tmp = id;
		u8 c = 0;
		while (tmp && c < 7)
		{
			auto res = std::div(tmp, 10llu);
			fileName[19-c] = '0' + static_cast<char>(res.rem);
			tmp = res.quot;
			c++;
		}
		stbi_write_jpg(fileName, WIDTH, HEIGHT, 4, data.data(), 100);
		running.Store(false);
	}
}
