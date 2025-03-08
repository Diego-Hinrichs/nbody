#include "../../include/ui/simulation_ui_manager.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <string>

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
    ImGui::Separator();
    renderSimulationControls();

    ImGui::Separator();
    ImGui::Text("Simulation Setup");
    renderDistributionOptions();
    ImGui::Separator();
    renderBodyCountSelector();
    ImGui::Separator();

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
    int numBodies = simulationState_.numBodies.load();
    if (ImGui::SliderInt("Number of Bodies", &numBodies, 1024, 1024000, "%d"))
    {
        simulationState_.numBodies.store(numBodies);
        simulationState_.restart.store(true);
    }

    // Particle size slider
    static float particleSize = renderer.getParticleSize(); // Get initial value from renderer
    if (ImGui::SliderFloat("Tamaño de Partículas", &particleSize, 5.0f, 15.0f, "%.1f"))
    {
        // Update renderer when value changes
        renderer.setParticleSize(particleSize);
    }

    renderSimulationOptionsPanel();
}

void SimulationUIManager::renderDistributionOptions()
{
    // Ensure buffer is always null-terminated
    simulationState_.seedInputBuffer[sizeof(simulationState_.seedInputBuffer) - 1] = '\0';

    if (ImGui::InputText("##SeedInput", simulationState_.seedInputBuffer,
                         sizeof(simulationState_.seedInputBuffer),
                         ImGuiInputTextFlags_CharsDecimal))
    {
        // Convert input to unsigned int
        unsigned int newSeed = 0;
        try
        {
            newSeed = static_cast<unsigned int>(std::stoul(simulationState_.seedInputBuffer));
            simulationState_.randomSeed.store(newSeed);
            simulationState_.seedWasChanged = true;
        }
        catch (...)
        {
            // Invalid input, keep old value
            snprintf(simulationState_.seedInputBuffer, sizeof(simulationState_.seedInputBuffer),
                     "%u", simulationState_.randomSeed.load());
        }
    }
    ImGui::SameLine();
    ImGui::Text("Random Seed");

    if (ImGui::Button("New Seed"))
    {
        unsigned int newSeed = static_cast<unsigned int>(time(nullptr));
        simulationState_.randomSeed.store(newSeed);
        simulationState_.seedWasChanged = true;
        snprintf(simulationState_.seedInputBuffer, sizeof(simulationState_.seedInputBuffer),
                 "%u", newSeed);
        simulationState_.restart.store(true);
    }
    ImGui::SameLine();
    if (ImGui::Button("Apply Seed"))
    {
        simulationState_.restart.store(true);
    }

    static const char *distributionTypes[] = {
        "Solar System",
        "Galaxy",
        "Binary System",
        "Uniform Sphere",
        "Random Clusters"};

    int currentDist = static_cast<int>(simulationState_.bodyDistribution.load());
    if (ImGui::Combo("Distribution Type", &currentDist, distributionTypes, IM_ARRAYSIZE(distributionTypes)))
    {
        simulationState_.bodyDistribution.store(static_cast<BodyDistribution>(currentDist));
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

    // Only show SFC options if SFC is enabled
    if (sfcEnabled)
    {
        // SFC Ordering Mode Selection
        static const char *orderingModes[] = {"Particle Ordering", "Octant Ordering"};
        int currentMode = static_cast<int>(simulationState_.sfcOrderingMode.load());

        if (ImGui::Combo("Ordering Mode", &currentMode, orderingModes, IM_ARRAYSIZE(orderingModes)))
        {
            simulationState_.sfcOrderingMode.store(static_cast<SFCOrderingMode>(currentMode));
            simulationState_.restart.store(true);
        }

        // Reorder Frequency Slider
        int reorderFreq = simulationState_.reorderFrequency.load();
        if (ImGui::SliderInt("Reorder Frequency", &reorderFreq, 1, 100, "Every %d iterations"))
        {
            simulationState_.reorderFrequency.store(reorderFreq);
        }
    }

    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Press ESC key to exit the simulation");
}