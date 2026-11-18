#include "Simulation.h"
#include "Reference.h"
#include "Scene.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cstdio>
#include "Window.h"
#include "Panel.h" // for custom imgui

using namespace oops;

namespace
{
    void help()
    {
        std::cout << "Mochi sphere simulation (CUDA + OptiX)\n"
                     "  --live [--display-every N]  Show the simulation in a window (default N = 50)\n"
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

    void parentDirectory(const std::string &path)
    {
        auto parent = std::filesystem::path(path).parent_path();
        if (!parent.empty())
            std::filesystem::create_directories(parent);
    }
}

int main(int argc, char **argv)
{
    using Clock = std::chrono::steady_clock;
    auto ms = [](Clock::time_point a, Clock::time_point b)
    {
        return std::chrono::duration<float, std::milli>(b - a).count();
    };
    auto smooth = [](float &value, float sample)
    { value = 0.9f * value + 0.1f * sample; };
    try
    {
        Settings settings;
        Response response = Response::Contacts;
        std::string input, saveInput, snapshot, image, timingFile;
        std::string scene = "mixed", renderDir = "build/frames";
        std::string stateDir = "build/states";
        // for live output
        bool live = false;
        int displayEvery = 50;
        // original vars
        int count = 4096, steps = 1000, warmup = 0, rebuildEvery = 0, renderEvery = 0;
        int stateEvery = 0;
        unsigned seed = 1;
        bool validate = false, morton = false;
        for (int i = 1; i < argc; ++i)
        {
            std::string option = argv[i];
            auto value = [&]() -> std::string
            {
                if (++i >= argc)
                    throw std::runtime_error("Missing value for " + option);
                return argv[i];
            };
            if (option == "--help" || option == "-h")
            {
                help();
                return 0;
            }
            else if (option == "--input")
                input = value();
            else if (option == "--particles")
                count = std::stoi(value());
            else if (option == "--scene")
                scene = value();
            else if (option == "--seed")
                seed = std::stoul(value());
            else if (option == "--steps")
                steps = std::stoi(value());
            else if (option == "--warmup")
                warmup = std::stoi(value());
            else if (option == "--rebuild-every")
                rebuildEvery = std::stoi(value());
            else if (option == "--dt")
                settings.dt = std::stof(value());
            else if (option == "--stiffness")
                settings.stiffness = std::stof(value());
            else if (option == "--restitution")
                settings.restitution = std::stof(value());
            else if (option == "--box-width")
                settings.boxWidth = std::stof(value());
            else if (option == "--gravity")
            {
                settings.gravity.x = std::stof(value());
                settings.gravity.y = std::stof(value());
                settings.gravity.z = std::stof(value());
            }
            else if (option == "--response")
            {
                auto mode = value();
                if (mode != "contacts" && mode != "fused")
                    throw std::runtime_error("Unknown response: " + mode);
                response = mode == "contacts" ? Response::Contacts : Response::Fused;
            }
            else if (option == "--save-input")
                saveInput = value();
            else if (option == "--snapshot")
                snapshot = value();
            else if (option == "--image")
                image = value();
            else if (option == "--timings")
                timingFile = value();
            else if (option == "--render-every")
                renderEvery = std::stoi(value());
            else if (option == "--render-dir")
                renderDir = value();
            else if (option == "--state-every")
                stateEvery = std::stoi(value());
            else if (option == "--state-dir")
                stateDir = value();
            else if (option == "--validate")
                validate = true;
            else if (option == "--morton")
                morton = true;
            else if (option == "--strict-upstream")
                settings.strictUpstream = true;
            // for live
            else if (option == "--live")
                live = true;
            else if (option == "--display-every")
                displayEvery = std::stoi(value());
            else
                throw std::runtime_error("Unknown option: " + option);
        }
        if (count < 1 || steps < 0 || warmup < 0 || rebuildEvery < 0 || renderEvery < 0 || stateEvery < 0 || displayEvery < 1 || !(settings.dt > 0.f) || !(settings.stiffness > 0.f) || !(settings.restitution >= 0.f && settings.restitution <= 1.f))
            throw std::runtime_error("Invalid particle count, timestep, or simulation parameter.");
        if (scene != "mixed" && scene != "uniform" && scene != "clustered")
            throw std::runtime_error("Scene must be mixed, uniform, or clustered.");

        auto initial = input.empty() ? makeScene(count, scene, seed) : loadScene(input);
        if (!saveInput.empty())
        {
            parentDirectory(saveInput);
            saveScene(saveInput, initial);
        }
        if (morton)
            mortonSort(initial);
        Simulation simulation(initial, settings, response);

        if (validate)
        {
            simulation.computeForces();
            auto actual = simulation.contacts();
            auto expected = referenceContacts(initial, settings.strictUpstream);
            auto forces = simulation.forces();
            auto reference = referenceForces(initial, settings);
            float maximumError = 0.f;
            bool correct = actual == expected;
            for (size_t i = 0; i < forces.size(); ++i)
            {
                for (int axis = 0; axis < 3; ++axis)
                {
                    float error = std::abs(forces[i][axis] - reference[i][axis]);
                    maximumError = std::max(maximumError, error);
                    correct &= error <= 0.001f + 0.00005f * std::abs(reference[i][axis]);
                }
            }
            std::cout << "CPU pairs: " << expected.size() << ", GPU pairs: " << actual.size()
                      << ", maximum force error: " << maximumError << '\n'
                      << (correct ? "PASS\n" : "FAIL\n");
            if (!correct)
                return 1;
        }
        else
        {
            for (int i = 0; i < warmup; ++i)
                simulation.step(rebuildEvery);
            simulation.timings = {};
            auto start = std::chrono::steady_clock::now();
            int stepCount = 0;

            auto afterStep = [&]()
            {
                ++stepCount;
                if (stateEvery && stepCount % stateEvery == 0)
                {
                    std::filesystem::create_directories(stateDir);
                    saveScene(stateDir + "/state_" + std::to_string(stepCount) + ".txt",
                              simulation.particles());
                }
                if (renderEvery && stepCount % renderEvery == 0)
                {
                    std::filesystem::create_directories(renderDir);
                    writeImage(renderDir + "/frame_" + std::to_string(stepCount) + ".png",
                               800, 600, simulation.render());
                }
            };

            if (live)
            {
                Window window(800, 600, "mochi");
                OrbitCamera orbit = simulation.defaultCamera();
                PanelState panel;
                panel.displayEvery = displayEvery;
                // ----- recording vars
                std::string recordDir;
                bool wasRecording = false;
                std::vector<uint32_t> frame;
                // -----
                while (!window.shouldClose())
                {
                    window.beginUi();
                    drawPanel(simulation, panel);
                    window.setLeftInset(panelInset(panel));
                    simulation.profile = panel.profilePhases;
                    if (panel.resetRequested)
                    {
                        simulation.upload(initial);
                        stepCount = 0;
                        panel.resetRequested = false;
                    }
                    if (panel.applySceneRequested)
                    {
                        panel.applySceneRequested = false;
                        auto generated = makeScene(panel.scene);
                        if (generated.empty())
                        {
                            panel.sceneStatus = "No grains fit: check that the blocks are inside the box.";
                        }
                        else
                        {
                            initial = std::move(generated);
                            simulation.settings.boxWidth = panel.scene.boxWidth;
                            simulation.settings.stiffness = panel.scene.stiffness;
                            simulation.settings.dt = std::min(0.0004f, stableDt(initial, panel.scene.stiffness));
                            simulation.resize(initial);
                            stepCount = 0;
                            orbit = simulation.defaultCamera();
                            char status[96];
                            std::snprintf(status, sizeof(status), "%d grains, dt = %.1e s",
                                          int(initial.size()), simulation.settings.dt);
                            panel.sceneStatus = status;
                        }
                    }
                    if (panel.recording && !wasRecording)
                    {
                        int take = 1;
                        while (std::filesystem::exists("out/recordings/take_" + std::to_string(take)))
                            ++take;
                        recordDir = "out/recordings/take_" + std::to_string(take);
                        std::filesystem::create_directories(recordDir);
                        panel.recordedFrames = 0;
                    }
                    wasRecording = panel.recording;

                    Timings before = simulation.timings;
                    auto t0 = Clock::now(); // start of physics time
                    // int stepsThisFrame = panel.paused ? (panel.stepOnce ? 1 : 0) : panel.displayEvery;
                    int stepsThisFrame = panel.paused ? (panel.stepOnce ? panel.displayEvery : 0) : panel.displayEvery; // one frame instead of 1 physics step
                    panel.stepOnce = false;

                    for (int i = 0; i < stepsThisFrame; ++i)
                    {
                        simulation.step(rebuildEvery);
                        afterStep();
                    }
                    if (panel.advancedStats)
                    {
                        cudaDeviceSynchronize();
                    }
                    panel.stepsThisFrame = stepsThisFrame;
                    auto t1 = Clock::now(); // end of physics time
                    orbit.update(window.input());
                    if (uint32_t *pixels = window.map())
                    {
                        simulation.renderTo(pixels, orbit.camera(), window.width(), window.height());
                        if (panel.recording)
                        {
                            frame.resize(size_t(window.width()) * window.height());
                            cudaMemcpy(frame.data(), pixels, frame.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost);
                            char name[32];
                            std::snprintf(name, sizeof(name), "/frame_%05d.png", panel.recordedFrames++);
                            writeImage(recordDir + name, window.width(), window.height(), frame);
                        }
                        window.unmap();
                    }
                    auto t2 = Clock::now();
                    window.present();
                    auto t3 = Clock::now();
                    float frameSeconds = ms(t0, t3) / 1000.f;
                    if (frameSeconds > 0.f)
                    {
                        smooth(panel.stepsPerSecond, stepsThisFrame / frameSeconds);
                        panel.realtimeFactor = panel.stepsPerSecond * simulation.settings.dt;
                    }
                    if (panel.advancedStats)
                    {
                        smooth(panel.physicsMs, ms(t0, t1));
                        smooth(panel.renderMs, ms(t1, t2));
                        smooth(panel.presentMs, ms(t2, t3));
                        panel.frameHistory[panel.historyIndex] = ms(t0, t3);
                        panel.historyIndex = (panel.historyIndex + 1) % 120;
                        if (stepsThisFrame > 0)
                        {
                            const Timings &now = simulation.timings;
                            float n = float(stepsThisFrame);
                            smooth(panel.accelMs, float(now.accel - before.accel) / n);
                            smooth(panel.queryMs, float(now.query - before.query) / n);
                            smooth(panel.responseMs, float(now.response - before.response) / n);
                            smooth(panel.integrateMs, float(now.integrate - before.integrate) / n);
                        }
                    }
                }
            }
            else
            {
                for (int i = 0; i < steps; ++i)
                {
                    simulation.step(rebuildEvery);
                    afterStep();
                }
            }

            double total = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count();
            if (steps == 0 && warmup == 0)
                simulation.computeForces();
            const auto &t = simulation.timings;
            std::cout << initial.size() << " particles, " << stepCount << " measured steps\n"
                      << "accel=" << t.accel << " query=" << t.query << " response=" << t.response
                      << " integrate=" << t.integrate << " wall=" << total << " ms\n";
            if (!timingFile.empty())
            {
                parentDirectory(timingFile);
                std::ofstream csv(timingFile);
                if (!csv)
                    throw std::runtime_error("Could not write timings: " + timingFile);
                csv << "particles,steps,accel_ms,query_ms,response_ms,integrate_ms,wall_ms\n"
                    << initial.size() << ',' << stepCount << ',' << t.accel << ',' << t.query << ','
                    << t.response << ',' << t.integrate << ',' << total << '\n';
            }
        }
        if (!snapshot.empty())
        {
            parentDirectory(snapshot);
            saveSnapshot(snapshot, simulation.particles(), simulation.forces(), simulation.contacts());
        }
        if (!image.empty())
        {
            parentDirectory(image);
            writeImage(image, 800, 600, simulation.render());
        }
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "mochi: " << error.what() << '\n';
        return 1;
    }
}
