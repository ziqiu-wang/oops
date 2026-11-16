#include "Window.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <exception>

int main() {
    try {
        oops::Window window(800, 600, "window test");
        while (!window.shouldClose()) {
            uint32_t* pixels = window.map();
            cudaMemset(pixels, 0x80, 800 * 600 * 4);
            cudaMemset(pixels, 0xFF, 800 * 4);
            window.unmap();
            window.present();
        }
    } catch (const std::exception& e) {
        printf("error: %s\n", e.what());
        return 1;
    }
}
