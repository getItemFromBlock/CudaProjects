#pragma once

#include <thread>
#include <vector>

#include "Signal.hpp"
#include "Types.hpp"
#include "RenderThread.hpp"

class EncoderThread
{
public:
	EncoderThread();

	~EncoderThread();

	void Init(const Parameters& params);
	void AssignFrame(u64 frameID, const std::vector<u32>& frameData);
	u64 GetFrameID() const;
	bool IsAvailable() const;
	void Quit();

private:
	Parameters params;
	std::thread thread;
	Core::Signal running;
	Core::Signal exit;
	std::vector<u32> data;
	u64 id = 0;

	void ThreadFunc();
};
