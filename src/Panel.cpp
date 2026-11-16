#include "Panel.h"
#include <imgui.h>

namespace oops {

void drawPanel(Simulation& simulation, PanelState& state) {
    const ImGuiViewport* vp = ImGui::GetMainViewport();
    const ImGuiWindowFlags fixed = ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoResize
                                 | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar;

    if (!state.open) {
        ImGui::SetNextWindowPos(vp->WorkPos, ImGuiCond_Always);
        ImGui::Begin("##expand", nullptr, fixed | ImGuiWindowFlags_AlwaysAutoResize);
        if (ImGui::Button(">>")) state.open = true;
        ImGui::End();
        return;
    }

    ImGui::SetNextWindowPos(vp->WorkPos, ImGuiCond_Always);
    ImGui::SetNextWindowSize(ImVec2(state.width, vp->WorkSize.y), ImGuiCond_Always);
    ImGui::Begin("Settings", nullptr, fixed);
    if (ImGui::Button("<<")) state.open = false;
    ImGui::SameLine();
    ImGui::TextUnformatted("Settings");
    ImGui::Separator();

    ImGui::End();
}

}
