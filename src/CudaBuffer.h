#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace oops {

inline void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

template<class T> class DeviceBuffer {
public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) { resize(count); }
    ~DeviceBuffer() { cudaFree(pointer_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    void resize(size_t count) {
        cudaCheck(cudaFree(pointer_));
        pointer_ = nullptr;
        size_ = count;
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
    }
    void upload(const std::vector<T>& values) {
        if (size_ != values.size()) resize(values.size());
        if (size_) cudaCheck(cudaMemcpy(pointer_, values.data(), size_ * sizeof(T), cudaMemcpyHostToDevice));
    }
    std::vector<T> download(size_t count) const {
        std::vector<T> values(count);
        if (count) cudaCheck(cudaMemcpy(values.data(), pointer_, count * sizeof(T), cudaMemcpyDeviceToHost));
        return values;
    }
    std::vector<T> download() const { return download(size_); }
    void clear() { if (size_) cudaCheck(cudaMemset(pointer_, 0, size_ * sizeof(T))); }
    T* data() const { return pointer_; }
    size_t size() const { return size_; }

private:
    T* pointer_ = nullptr;
    size_t size_ = 0;
};

} // namespace oops
