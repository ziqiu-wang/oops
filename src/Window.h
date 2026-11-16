#pragma once
#include <cstdint>

struct GLFWwindow;
struct cudaGraphicsResource;

namespace oops {

class Window {
public:
    Window(int width, int height, const char* title);
    ~Window();
    // ban copying
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const;
    uint32_t* map();
    void unmap();
    void present();

private:
    int width_, height_;
    GLFWwindow* window_ = nullptr;
    unsigned pbo_ = 0, texture_ = 0, vao_ = 0, program_ = 0;
    cudaGraphicsResource* resource_ = nullptr;
};

}