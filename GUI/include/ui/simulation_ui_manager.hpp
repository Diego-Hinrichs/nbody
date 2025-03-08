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
#include "simulation_state.hpp"
#include "opengl_renderer.hpp" // Incluir el header del renderer

class SimulationUIManager
{
public:
    // SimulationUIManager(SimulationState &state);
    SimulationUIManager(SimulationState& simulationState, OpenGLRenderer& rendererRef)
    : simulationState_(simulationState), renderer(rendererRef) {}

    void renderUI(GLFWwindow *window);

private:
    SimulationState &simulationState_;
    OpenGLRenderer& renderer;

    // void renderSimulationControls();
    void renderPerformanceInfo();
    // void renderBodyCountSelector();
    // void renderDistributionOptions();
    // void renderSimulationOptionsPanel();
    void renderSimulationMethodSelector();

    void renderBasicControls();
    void renderVisualizationOptions();
    void renderBodyGenerationOptions();
    void renderAdvancedOptions();
};

#endif // SIMULATION_UI_MANAGER_H