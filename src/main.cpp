#include "Simulation.h"
#include "Reference.h"
#include "Scene.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace oops;

namespace {
void help() {
    std::cout <<
        "Mochi sphere simulation (CUDA + OptiX)\n"
        "  --input FILE                x y z radius mass [vx vy vz]\n"
        "  --particles N --scene mixed|uniform|clustered --seed N\n"
        "  --save-input FILE           Export the generated/input scene\n"
        "  --steps N --warmup N        Defaults: 1000 steps, no warmup\n"
        "  --response contacts|fused   Default: contacts\n"
        "  --rebuild-every N           0 = refit only, 1 = always rebuild\n"
        "  --dt F --stiffness F --restitution F --box-width F\n"
        "  --gravity X Y Z            Default: 0 -9.81 0\n"
        "  --strict-upstream          Original strict boundary predicates\n"
        "  --morton                   Original initial particle sorting\n"
        "  --validate                 Compare static pairs/forces with CPU and exit\n"
        "  --snapshot FILE            CSV state and FILE.pairs\n"
        "  --image FILE.png           Render the final state\n"
        "  --render-every N --render-dir DIR   Export an image sequence\n"
        "  --state-every N --state-dir DIR     Export matching particle states\n"
        "  --timings FILE.csv         Phase and total wall times, excluding setup/warmup\n";
}

void parentDirectory(const std::string& path) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent);
}
}

int main(int argc, char** argv) {
    try {
        Settings settings;
        Response response = Response::Contacts;
        std::string input, saveInput, snapshot, image, timingFile;
        std::string scene = "mixed", renderDir = "build/frames";
        std::string stateDir = "build/states";
        int count = 4096, steps = 1000, warmup = 0, rebuildEvery = 0, renderEvery = 0;
        int stateEvery = 0;
        unsigned seed = 1;
        bool validate = false, morton = false;
        for (int i = 1; i < argc; ++i) {
            std::string option = argv[i];
            auto value = [&]() -> std::string {
                if (++i >= argc) throw std::runtime_error("Missing value for " + option);
                return argv[i];
            };
            if (option == "--help" || option == "-h") { help(); return 0; }
            else if (option == "--input") input = value();
            else if (option == "--particles") count = std::stoi(value());
            else if (option == "--scene") scene = value();
            else if (option == "--seed") seed = std::stoul(value());
            else if (option == "--steps") steps = std::stoi(value());
            else if (option == "--warmup") warmup = std::stoi(value());
            else if (option == "--rebuild-every") rebuildEvery = std::stoi(value());
            else if (option == "--dt") settings.dt = std::stof(value());
            else if (option == "--stiffness") settings.stiffness = std::stof(value());
            else if (option == "--restitution") settings.restitution = std::stof(value());
            else if (option == "--box-width") settings.boxWidth = std::stof(value());
            else if (option == "--gravity") {
                settings.gravity.x = std::stof(value());
                settings.gravity.y = std::stof(value());
                settings.gravity.z = std::stof(value());
            } else if (option == "--response") {
                auto mode = value();
                if (mode != "contacts" && mode != "fused") throw std::runtime_error("Unknown response: " + mode);
                response = mode == "contacts" ? Response::Contacts : Response::Fused;
            } else if (option == "--save-input") saveInput = value();
            else if (option == "--snapshot") snapshot = value();
            else if (option == "--image") image = value();
            else if (option == "--timings") timingFile = value();
            else if (option == "--render-every") renderEvery = std::stoi(value());
            else if (option == "--render-dir") renderDir = value();
            else if (option == "--state-every") stateEvery = std::stoi(value());
            else if (option == "--state-dir") stateDir = value();
            else if (option == "--validate") validate = true;
            else if (option == "--morton") morton = true;
            else if (option == "--strict-upstream") settings.strictUpstream = true;
            else throw std::runtime_error("Unknown option: " + option);
        }
        if (count < 1 || steps < 0 || warmup < 0 || rebuildEvery < 0 || renderEvery < 0 || stateEvery < 0
            || !(settings.dt > 0.f) || !(settings.stiffness > 0.f)
            || !(settings.restitution >= 0.f && settings.restitution <= 1.f))
            throw std::runtime_error("Invalid particle count, timestep, or simulation parameter.");
        if (scene != "mixed" && scene != "uniform" && scene != "clustered")
            throw std::runtime_error("Scene must be mixed, uniform, or clustered.");

        auto initial = input.empty() ? makeScene(count, scene, seed) : loadScene(input);
        if (!saveInput.empty()) { parentDirectory(saveInput); saveScene(saveInput, initial); }
        if (morton) mortonSort(initial);
        Simulation simulation(initial, settings, response);

        if (validate) {
            simulation.computeForces();
            auto actual = simulation.contacts();
            auto expected = referenceContacts(initial, settings.strictUpstream);
            auto forces = simulation.forces();
            auto reference = referenceForces(initial, settings);
            float maximumError = 0.f;
            bool correct = actual == expected;
            for (size_t i = 0; i < forces.size(); ++i) {
                for (int axis = 0; axis < 3; ++axis) {
                    float error = std::abs(forces[i][axis] - reference[i][axis]);
                    maximumError = std::max(maximumError, error);
                    correct &= error <= 0.001f + 0.00005f * std::abs(reference[i][axis]);
                }
            }
            std::cout << "CPU pairs: " << expected.size() << ", GPU pairs: " << actual.size()
                      << ", maximum force error: " << maximumError << '\n'
                      << (correct ? "PASS\n" : "FAIL\n");
            if (!correct) return 1;
        } else {
            for (int i = 0; i < warmup; ++i) simulation.step(rebuildEvery);
            simulation.timings = {};
            auto start = std::chrono::steady_clock::now();
            for (int i = 0; i < steps; ++i) {
                simulation.step(rebuildEvery);
                if (stateEvery && (i + 1) % stateEvery == 0) {
                    std::filesystem::create_directories(stateDir);
                    saveScene(stateDir + "/state_" + std::to_string(i + 1) + ".txt",
                              simulation.particles());
                }
                if (renderEvery && (i + 1) % renderEvery == 0) {
                    std::filesystem::create_directories(renderDir);
                    writeImage(renderDir + "/frame_" + std::to_string(i + 1) + ".png",
                               800, 600, simulation.render());
                }
            }
            double total = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            if (steps == 0 && warmup == 0) simulation.computeForces();
            const auto& t = simulation.timings;
            std::cout << initial.size() << " particles, " << steps << " measured steps\n"
                      << "accel=" << t.accel << " query=" << t.query << " response=" << t.response
                      << " integrate=" << t.integrate << " wall=" << total << " ms\n";
            if (!timingFile.empty()) {
                parentDirectory(timingFile);
                std::ofstream csv(timingFile);
                if (!csv) throw std::runtime_error("Could not write timings: " + timingFile);
                csv << "particles,steps,accel_ms,query_ms,response_ms,integrate_ms,wall_ms\n"
                    << initial.size() << ',' << steps << ',' << t.accel << ',' << t.query << ','
                    << t.response << ',' << t.integrate << ',' << total << '\n';
            }
        }
        if (!snapshot.empty()) {
            parentDirectory(snapshot);
            saveSnapshot(snapshot, simulation.particles(), simulation.forces(), simulation.contacts());
        }
        if (!image.empty()) {
            parentDirectory(image);
            writeImage(image, 800, 600, simulation.render());
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mochi: " << error.what() << '\n';
        return 1;
    }
}
