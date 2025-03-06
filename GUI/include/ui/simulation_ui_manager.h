#ifndef SIMULATION_UI_MANAGER_H
#define SIMULATION_UI_MANAGER_H

// OpenGL and GLFW headers
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// Standard library headers
#include <vector>
#include <string>

// Project headers
#include "../common/types.cuh"
#include "../common/constants.cuh"
#include "simulation_state.h"

class SimulationUIManager
{
public:
    SimulationUIManager(SimulationState &state);

    // Render the simulation UI
    void renderUI(GLFWwindow *window);

private:
    SimulationState &simulationState_;

    // UI rendering methods
    void renderSimulationControls();
    void renderPerformanceInfo();
    void renderBodyCountSelector();
    void renderSimulationOptionsPanel();
};

#endif // SIMULATION_UI_MANAGER_H