#include "Panel.h"
#include <imgui.h>
#include <algorithm>
#include <cstdio>

namespace oops
{

    namespace
    {
        void drawFpsOverlay()
        {
            const ImGuiViewport *vp = ImGui::GetMainViewport();
            const float pad = 10.f;
            ImVec2 pos(vp->WorkPos.x + vp->WorkSize.x - pad, vp->WorkPos.y + pad);
            ImGui::SetNextWindowPos(pos, ImGuiCond_Always, ImVec2(1.f, 0.f));
            ImGui::SetNextWindowBgAlpha(0.35f);
            ImGui::Begin("##fps", nullptr,
                         ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                             ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                             ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
            ImGui::Text("%.1f fps", ImGui::GetIO().Framerate);
            ImGui::End();
        }

        const char *shapeNames[] = {"Box", "Sphere", "Cylinder"};

        void editBlock(Block &block)
        {
            int shape = int(block.shape);
            if (ImGui::Combo("Shape", &shape, shapeNames, 3))
                block.shape = Shape(shape);
            ImGui::DragFloat3("Center", &block.center.x, 0.005f, 0.f, 3.f, "%.3f");
            ImGui::DragFloat3("Size", &block.size.x, 0.005f, 0.01f, 3.f, "%.3f");
            ImGui::SliderFloat("Grain radius", &block.radius, 0.004f, 0.06f, "%.4f",
                               ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
            ImGui::SliderFloat("Size spread", &block.spread, 0.f, 0.8f, "%.2f", ImGuiSliderFlags_AlwaysClamp);
            ImGui::DragFloat3("Velocity", &block.velocity.x, 0.05f, -10.f, 10.f, "%.2f");
        }

        void drawSceneSection(PanelState &state)
        {
            ImGui::SeparatorText("Scene");
            ImGui::PushItemWidth(-100.f);

            static const std::vector<Preset> presets = scenePresets();
            const char *current = state.presetIndex >= 0 ? presets[state.presetIndex].name : "Custom";
            if (ImGui::BeginCombo("Preset", current))
            {
                for (int i = 0; i < int(presets.size()); ++i)
                {
                    if (ImGui::Selectable(presets[i].name, i == state.presetIndex))
                    {
                        state.presetIndex = i;
                        state.scene = presets[i].scene;
                        state.selectedBlock = 0;
                        state.applySceneRequested = true;
                    }
                }
                ImGui::EndCombo();
            }

            auto &blocks = state.scene.blocks;
            int total = 0;
            for (int i = 0; i < int(blocks.size()); ++i)
            {
                int estimate = estimateCount(blocks[i]);
                total += estimate;
                char label[64];
                std::snprintf(label, sizeof(label), "%d: %s (~%d grains)", i + 1, shapeNames[int(blocks[i].shape)], estimate);
                if (ImGui::Selectable(label, i == state.selectedBlock))
                    state.selectedBlock = i;
            }

            if (ImGui::Button("Add block"))
            {
                Block copy = blocks[state.selectedBlock];
                blocks.push_back(copy);
                state.selectedBlock = int(blocks.size()) - 1;
                state.presetIndex = -1;
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(blocks.size() <= 1);
            if (ImGui::Button("Remove block"))
            {
                blocks.erase(blocks.begin() + state.selectedBlock);
                state.selectedBlock = std::min(state.selectedBlock, int(blocks.size()) - 1);
                state.presetIndex = -1;
            }
            ImGui::EndDisabled();

            ImGui::PushID(state.selectedBlock);
            editBlock(blocks[state.selectedBlock]);
            ImGui::PopID();

            ImGui::SliderFloat("Container size", &state.scene.boxWidth, 0.3f, 3.f, "%.2f m", ImGuiSliderFlags_AlwaysClamp);
            ImGui::TextDisabled("About %d grains in total", total);
            if (ImGui::Button("Apply scene"))
                state.applySceneRequested = true;
            if (!state.sceneStatus.empty())
                ImGui::TextWrapped("%s", state.sceneStatus.c_str());

            ImGui::PopItemWidth();
        }

    } // namespace

    void drawPanel(Simulation &simulation, PanelState &state)
    {
        // pausing from space regardless of open
        if (!ImGui::GetIO().WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_Space, false))
        {
            state.paused = !state.paused;
        }
        // fps at top right
        if (state.showFps)
        {
            drawFpsOverlay();
        }
        const ImGuiViewport *vp = ImGui::GetMainViewport();
        const ImGuiWindowFlags fixed = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;

        if (!state.open)
        {
            ImGui::SetNextWindowPos(vp->WorkPos, ImGuiCond_Always);
            ImGui::Begin("##expand", nullptr, fixed | ImGuiWindowFlags_AlwaysAutoResize);
            if (ImGui::Button(">>"))
                state.open = true;
            ImGui::End();
            return;
        }

        ImGui::SetNextWindowPos(vp->WorkPos, ImGuiCond_Always);
        ImGui::SetNextWindowSize(ImVec2(state.width, vp->WorkSize.y), ImGuiCond_Always);
        ImGui::Begin("Settings", nullptr, fixed);
        if (ImGui::Button("<<"))
            state.open = false;
        ImGui::SameLine();
        ImGui::TextUnformatted("Settings");
        ImGui::Separator();

        // playback buttons
        ImGui::SeparatorText("Playback");
        if (ImGui::Button(state.paused ? "Play" : "Pause", ImVec2(60, 0)))
            state.paused = !state.paused;
        ImGui::SameLine();
        ImGui::BeginDisabled(!state.paused);
        if (ImGui::Button("Next frame"))
            state.stepOnce = true;
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
            state.resetRequested = true;

        ImGui::SliderInt("Steps/frame", &state.displayEvery, 1, 500, "%d", ImGuiSliderFlags_AlwaysClamp);
        ImGui::Text("Sim time: %.3f s", simulation.steps * simulation.settings.dt);

        ImGui::SeparatorText("Record");
        if (ImGui::Button(state.recording ? "Stop" : "Record", ImVec2(60, 0)))
            state.recording = !state.recording;
        if (state.recording)
        {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.f, 0.3f, 0.3f, 1.f), "REC  %d frames", state.recordedFrames);
        }

        drawSceneSection(state);

        // stat buttons
        ImGui::Checkbox("Show FPS", &state.showFps);
        ImGui::Checkbox("Advanced stats", &state.advancedStats);
        if (state.advancedStats)
        {
            ImGui::Indent();
            ImGui::Text("Physics: %.2f ms", state.physicsMs);
            ImGui::Text("Render:  %.2f ms", state.renderMs);
            ImGui::Text("UI+present: %.2f ms", state.presentMs);
            // ImGui::Text("Steps this frame: %d", state.stepsThisFrame);
            // ImGui::Text("Steps/s: %.0f", state.stepsPerSecond);
            ImGui::Text("Real-time: %.2fx", state.realtimeFactor);
            ImGui::PlotLines("Frame ms", state.frameHistory, 120, state.historyIndex,
                             nullptr, 0.f, 50.f, ImVec2(0, 60));
            ImGui::Checkbox("Profile phases (slower)", &state.profilePhases);
            if (state.profilePhases)
            {
                ImGui::Text("Per step: accel %.3f", state.accelMs);
                ImGui::Text("          query %.3f", state.queryMs);
                ImGui::Text("          response %.3f", state.responseMs);
                ImGui::Text("          integrate %.3f", state.integrateMs);
            }
            else
            {
                ImGui::TextDisabled("(enable to measure per-step phases)");
            }
            ImGui::Unindent();
        }

        ImGui::End();
    }

}
