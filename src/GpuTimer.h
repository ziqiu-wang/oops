#pragma once

#include <cuda_runtime.h>
#include <string>

// instead of using CPU timer (need to synchronize GPU events to measure)
// use cuda events

class GpuTimer {
public: 
	GpuTimer() {
		check(cudaEventCreate(&start), "cudaEventCreate");
		check(cudaEventCreate(&stop), "cudaEventCreate");
	}
	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void begin(cudaStream_t stream) {
		check(cudaEventRecord(start, stream), "cudaEventRecord");
	}

	void end(cudaStream_t stream) {
		check(cudaEventRecord(stop, stream), "cudaEventRecord");
	}
	
	float elapsedTime() {
		check(cudaEventSynchronize(stop), "cudaEventSynchronize");
		float ms = 0.f;
		check(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
		return ms;
	}


private: 
	static void check(cudaError_t result, const char* msg) {
		if (result != cudaSuccess) {
			throw std::runtime_error(std::string(msg) + " " + cudaGetErrorString(result));
		}
	}

	cudaEvent_t start, stop;
};
