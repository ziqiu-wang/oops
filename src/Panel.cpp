#include "Panel.h"
#include <imgui.h>

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
    }

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
        ImGui::SameLine();
        if (ImGui::Button("1 step")) state.stepSubstep = true;
        ImGui::EndDisabled();
        ImGui::SameLine();
        if (ImGui::Button("Reset"))
            state.resetRequested = true;

        ImGui::SliderInt("Steps/frame", &state.displayEvery, 1, 500, "%d", ImGuiSliderFlags_AlwaysClamp);
        ImGui::Text("Sim time: %.3f s", simulation.steps * simulation.settings.dt);

        ImGui::SeparatorText("Material (apply with Reset)");
        int physics = int(state.pending.physics);
        ImGui::Combo("Physics", &physics, "Mochi baseline\0Normal only\0Velocity friction\0History friction\0");
        state.pending.physics = Physics(physics);
        ImGui::SliderFloat("Friction", &state.pending.friction, 0.f, 1.5f);
        ImGui::SliderFloat("k_t / k_n", &state.pending.tangentialStiffness, 0.05f, 1.f);
        ImGui::SliderFloat("Tangential damping", &state.pending.tangentialDamping, 0.f, 3.f);
        ImGui::SliderFloat("Restitution", &state.pending.restitution, 0.f, 1.f);
        int walls = state.pending.walls == 63 ? 0 : state.pending.walls == 4 ? 1 : 2;
        ImGui::Combo("Walls", &walls, "Box\0Floor only\0None\0");
        state.pending.walls = walls == 0 ? 63 : walls == 1 ? 4 : 0;
        ImGui::SeparatorText("Rendering");
        ImGui::Combo("View", &state.view, "Path traced\0Stable preview\0Particle speed\0Contact state\0");
        ImGui::SliderInt("Samples", &state.samples, 1, 64);
        if (state.view == 3) ImGui::TextWrapped("Blue: loaded contact; orange: sliding contact; gray: no loaded contact. Any sliding contact wins per grain.");

        ImGui::SeparatorText("Record");
        if (ImGui::Button(state.recording ? "Stop" : "Record", ImVec2(60, 0)))
            state.recording = !state.recording;
        if (state.recording)
        {
            ImGui::SameLine();
            ImGui::TextColored(ImVec4(1.f, 0.3f, 0.3f, 1.f), "REC  %d frames", state.recordedFrames);
        }

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
            ImGui::Text("Contacts: %u (%u sliding)", state.diagnostics.contacts, state.diagnostics.sliding);
            ImGui::Text("Max speed: %.5f", state.diagnostics.maximumSpeed);
            ImGui::Text("Kinetic energy: %.6f", state.diagnostics.kineticEnergy);
            ImGui::Text("Contact storage: %.2f MiB", state.diagnostics.historyBytes / (1024. * 1024.));
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
