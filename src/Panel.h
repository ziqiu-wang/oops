#pragma once
#include "Simulation.h"

namespace oops
{

    struct PanelState
    {
        bool open = true;
        float width = 320.f;

        bool paused = false;
        bool resetRequested = false;
        int displayEvery = 50;
    };

    void drawPanel(Simulation &simulation, PanelState &state);

    inline int panelInset(const PanelState& s) {
        return s.open ? int(s.width) : 0;
    }

}
