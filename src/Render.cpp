#include "Simulation.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb/stb_image_write.h>

namespace oops {

void writeImage(const std::string& path, int width, int height,
                const std::vector<uint32_t>& pixels) {
    if (!stbi_write_png(path.c_str(), width, height, 4, pixels.data(), width * 4))
        throw std::runtime_error("Could not write image: " + path);
}

} // namespace oops
