#pragma once
#include "Simulation.h"
#include "Scene.h"

namespace oops
{

    struct PanelState
    {
        float width = 320.f;
        int displayEvery = 50;

        bool open = true;
        bool showFps = true; // iterations per second where one iteration includes [N physics steps] + [OptiX render] + [ImGui] + [present]

        bool profilePhases = false;
        bool advancedStats = false;
        float physicsMs = 0.f, renderMs = 0.f, presentMs = 0.f;
        float accelMs = 0.f, queryMs = 0.f, responseMs = 0.f, integrateMs = 0.f;
        int stepsThisFrame = 0;
        float frameHistory[120] = {};
        int historyIndex = 0;

        // more stats
        float stepsPerSecond = 0.f;
        float realtimeFactor = 0.f;

        // playback tools
        bool paused = false;
        bool stepOnce = false;
        bool resetRequested = false;
        bool recording = false;
        int recordedFrames = 0;

        // scene builder
        SceneDesc scene;
        int selectedBlock = 0;
        int presetIndex = -1;
        bool applySceneRequested = false;
        std::string sceneStatus;
    };

    void drawPanel(Simulation &simulation, PanelState &state);

    inline int panelInset(const PanelState &s)
    {
        return s.open ? int(s.width) : 0;
    }

}
