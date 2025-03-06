#include "../../include/ui/simulation_ui_manager.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>

SimulationUIManager::SimulationUIManager(SimulationState &state)
    : simulationState_(state)
{
}

void SimulationUIManager::renderUI(GLFWwindow *window)
{
    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Render simulation UI
    ImGui::Begin("Simulation Controls", nullptr,
                 ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoCollapse);

    renderPerformanceInfo();
    renderSimulationControls();
    renderBodyCountSelector();
    renderSimulationOptionsPanel();

    ImGui::End();

    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void SimulationUIManager::renderPerformanceInfo()
{
    // Performance metrics
    ImGui::Text("Bodies: %d", simulationState_.numBodies.load());
    ImGui::Text("FPS: %.1f", simulationState_.fps);
    ImGui::Text("Last Iteration Time: %.2f ms",
                simulationState_.lastIterationTime);
}

void SimulationUIManager::renderSimulationControls()
{
    ImGui::Separator();
    ImGui::Text("Simulation Controls");

    // Pause/Resume button
    if (ImGui::Button("Pause/Resume"))
    {
        simulationState_.isPaused.store(!simulationState_.isPaused.load());
    }

    ImGui::SameLine();

    // Restart button
    if (ImGui::Button("Restart"))
    {
        simulationState_.restart.store(true);
    }

    // Zoom slider
    float zoomFactor = simulationState_.zoomFactor.load();
    if (ImGui::SliderFloat("Zoom", &zoomFactor, 0.1f, 10.0f))
    {
        simulationState_.zoomFactor.store(zoomFactor);
    }
}

void SimulationUIManager::renderBodyCountSelector()
{
    ImGui::Separator();
    ImGui::Text("Simulation Setup");

    // Body count selection
    const char *bodyCounts[] = {"1000", "5000", "10000", "20000", "50000"};
    static int currentBodyCount = 0;

    if (ImGui::Combo("Number of Bodies", &currentBodyCount, bodyCounts, IM_ARRAYSIZE(bodyCounts)))
    {
        simulationState_.numBodies.store(std::stoi(bodyCounts[currentBodyCount]));
        simulationState_.restart.store(true);
    }
}

void SimulationUIManager::renderSimulationOptionsPanel()
{
    ImGui::Separator();
    ImGui::Text("Simulation Options");

    // SFC Toggle
    bool sfcEnabled = simulationState_.useSFC.load();
    if (ImGui::Checkbox("Space-Filling Curve", &sfcEnabled))
    {
        simulationState_.useSFC.store(sfcEnabled);
        simulationState_.restart.store(true);
    }
}