#pragma once

#include <thread>
#include <vector>

#include "Signal.hpp"
#include "Types.hpp"

class EncoderThread
{
public:
	EncoderThread();

	~EncoderThread();

	void Init();
	void AssignFrame(u64 frameID, const std::vector<u32>& frameData);
	u64 GetFrameID() const;
	bool IsAvailable() const;
	void Quit();

private:
	std::thread thread;
	Core::Signal running;
	Core::Signal exit;
	std::vector<u32> data;
	u64 id = 0;

	void ThreadFunc();
};
