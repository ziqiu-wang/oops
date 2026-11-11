#pragma once

#include "Types.h"
#include <string>
#include <vector>

namespace oops {

std::vector<Particle> makeScene(int count, const std::string& kind, unsigned seed);
std::vector<Particle> loadScene(const std::string& path);
void saveScene(const std::string& path, const std::vector<Particle>& particles);
void mortonSort(std::vector<Particle>& particles);
void saveSnapshot(const std::string& path, const std::vector<Particle>& particles,
                  const std::vector<Vec3>& forces, const std::vector<Contact>& contacts);

} // namespace oops
