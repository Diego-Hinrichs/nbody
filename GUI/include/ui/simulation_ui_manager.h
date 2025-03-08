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
#include "opengl_renderer.h" // Incluir el header del renderer

class SimulationUIManager
{
public:
    // SimulationUIManager(SimulationState &state);
    SimulationUIManager(SimulationState& simulationState, OpenGLRenderer& rendererRef)
    : simulationState_(simulationState), renderer(rendererRef) {}
    // Render the simulation UI
    void renderUI(GLFWwindow *window);

private:
    SimulationState &simulationState_;
    OpenGLRenderer& renderer; // Agregar referencia al renderer
    // UI rendering methods
    void renderSimulationControls();
    void renderPerformanceInfo();
    void renderBodyCountSelector();
    void renderDistributionOptions(); // New method for distribution options
    void renderSimulationOptionsPanel();
};

#endif // SIMULATION_UI_MANAGER_H