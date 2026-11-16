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

        ImGui::Checkbox("Show FPS", &state.showFps);
        ImGui::Checkbox("Advanced stats", &state.advancedStats);
        if (state.advancedStats)
        {
            ImGui::Indent();
            ImGui::Text("Physics: %.2f ms", state.physicsMs);
            ImGui::Text("Render:  %.2f ms", state.renderMs);
            ImGui::Text("UI+present: %.2f ms", state.presentMs);
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
