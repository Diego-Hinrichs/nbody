#include "../../include/ui/simulation_ui_manager.h"
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <string>
#include <omp.h>

void SimulationUIManager::renderUI(GLFWwindow *window)
{
    // Start ImGui frame
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    // Main window with simulation controls
    ImGui::Begin("N-Body Simulation", nullptr,
                 ImGuiWindowFlags_AlwaysAutoResize |
                     ImGuiWindowFlags_NoCollapse);

    // Performance metrics and control buttons at the top
    renderPerformanceInfo();
    ImGui::Separator();
    renderBasicControls();
    ImGui::Separator();

    // Create a tabbed interface for better organization
    if (ImGui::BeginTabBar("SimulationTabBar"))
    {
        // Tab 1: Simulation methods
        if (ImGui::BeginTabItem("Simulation Method"))
        {
            renderSimulationMethodSelector();
            ImGui::EndTabItem();
        }

        // Tab 2: Visualization settings
        if (ImGui::BeginTabItem("Visualization"))
        {
            renderVisualizationOptions();
            ImGui::EndTabItem();
        }

        // Tab 3: Body generation settings
        if (ImGui::BeginTabItem("Body Generation"))
        {
            renderBodyGenerationOptions();
            ImGui::EndTabItem();
        }

        // Tab 4: Advanced options (SFC)
        if (ImGui::BeginTabItem("Advanced Options"))
        {
            renderAdvancedOptions();
            ImGui::EndTabItem();
        }

        ImGui::EndTabBar();
    }

    // Help text at the bottom
    ImGui::Separator();
    ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.0f, 1.0f), "Press ESC key to exit the simulation");

    ImGui::End();

    // Render ImGui
    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void SimulationUIManager::renderPerformanceInfo()
{
    // Performance metrics
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Performance Metrics");
    ImGui::Text("Bodies: %d", simulationState_.numBodies.load());
    ImGui::Text("FPS: %.1f", simulationState_.fps);
    ImGui::Text("Iteration Time: %.2f ms", simulationState_.lastIterationTime);
}

void SimulationUIManager::renderBasicControls()
{
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Basic Controls");

    // Simulation control buttons in the same row
    if (ImGui::Button("Pause/Resume"))
    {
        simulationState_.isPaused.store(!simulationState_.isPaused.load());
    }

    ImGui::SameLine();

    if (ImGui::Button("Restart Simulation"))
    {
        simulationState_.restart.store(true);
    }

    // Zoom control
    float zoomFactor = simulationState_.zoomFactor.load();
    if (ImGui::SliderFloat("Zoom Level", &zoomFactor, 0.1f, 10.0f))
    {
        simulationState_.zoomFactor.store(zoomFactor);
    }
}

void SimulationUIManager::renderSimulationMethodSelector()
{
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Select Simulation Algorithm");

    // Method selection
    static const char *methodTypes[] = {
        "CPU Direct Sum",
        "GPU Direct Sum",
        "CPU Barnes-Hut",
        "GPU Barnes-Hut"};

    int currentMethod = static_cast<int>(simulationState_.simulationMethod.load());
    if (ImGui::Combo("Algorithm", &currentMethod, methodTypes, IM_ARRAYSIZE(methodTypes)))
    {
        simulationState_.simulationMethod.store(static_cast<SimulationMethod>(currentMethod));

        // Reset options based on new method selected
        switch (static_cast<SimulationMethod>(currentMethod))
        {
        case SimulationMethod::CPU_DIRECT_SUM:
            // Default options for CPU Direct Sum
            simulationState_.useOpenMP.store(true);
            simulationState_.openMPThreads.store(omp_get_max_threads());
            break;

        case SimulationMethod::GPU_DIRECT_SUM:
            // No specific options for GPU Direct Sum
            break;

        case SimulationMethod::CPU_BARNES_HUT:
            // Default options for CPU Barnes-Hut
            simulationState_.useOpenMP.store(true);
            simulationState_.openMPThreads.store(omp_get_max_threads());
            break;

        case SimulationMethod::BARNES_HUT:
            // By default, disable SFC for regular Barnes-Hut
            simulationState_.useSFC.store(false);
            break;

        case SimulationMethod::SFC_BARNES_HUT:
            // SFC is always enabled for SFC Barnes-Hut
            simulationState_.useSFC.store(true);
            simulationState_.sfcOrderingMode.store(SFCOrderingMode::PARTICLES);
            simulationState_.reorderFrequency.store(10);
            break;
        }

        // Trigger restart
        simulationState_.restart.store(true);
    }

    // Algorithm description
    ImGui::TextWrapped("Description:");
    switch (static_cast<SimulationMethod>(currentMethod))
    {
    case SimulationMethod::CPU_DIRECT_SUM:
        ImGui::TextWrapped("O(n²) all-pairs calculation on CPU. Suitable for small simulations.");
        break;
    case SimulationMethod::GPU_DIRECT_SUM:
        ImGui::TextWrapped("O(n²) all-pairs calculation on GPU. Higher performance than CPU for medium-sized simulations.");
        break;
    case SimulationMethod::CPU_BARNES_HUT:
        ImGui::TextWrapped("O(n log n) Barnes-Hut tree algorithm on CPU. Good for medium to large simulations.");
        break;
    case SimulationMethod::BARNES_HUT:
        ImGui::TextWrapped("O(n log n) Barnes-Hut tree algorithm on GPU. High performance for large simulations.");
        break;
    case SimulationMethod::SFC_BARNES_HUT:
        ImGui::TextWrapped("Space-Filling Curve enhanced Barnes-Hut algorithm. Optimized memory access for very large simulations.");
        break;
    }
}

void SimulationUIManager::renderVisualizationOptions()
{
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Visualization Settings");

    // Particle size slider
    static float particleSize = renderer.getParticleSize();
    if (ImGui::SliderFloat("Particle Size", &particleSize, 1.0f, 20.0f, "%.1f"))
    {
        renderer.setParticleSize(particleSize);
    }

    // Add more visualization options as needed
    // (camera position, colors, etc.)
}

void SimulationUIManager::renderBodyGenerationOptions()
{
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Body Generation");

    // Body count slider
    int numBodies = simulationState_.numBodies.load();
    if (ImGui::SliderInt("Number of Bodies", &numBodies, 1024, 64000, "%d"))
    {
        simulationState_.numBodies.store(numBodies);
        simulationState_.restart.store(true);
    }

    ImGui::Separator();

    // Distribution type selection
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Distribution Pattern");

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

    ImGui::Separator();

    // Random seed controls
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Random Seed");

    // Ensure buffer is always null-terminated
    simulationState_.seedInputBuffer[sizeof(simulationState_.seedInputBuffer) - 1] = '\0';

    ImGui::InputText("Seed Value", simulationState_.seedInputBuffer,
                     sizeof(simulationState_.seedInputBuffer),
                     ImGuiInputTextFlags_CharsDecimal);

    if (ImGui::Button("Apply Seed"))
    {
        // Convert input to unsigned int
        try
        {
            unsigned int newSeed = static_cast<unsigned int>(std::stoul(simulationState_.seedInputBuffer));
            simulationState_.randomSeed.store(newSeed);
            simulationState_.seedWasChanged = true;
            simulationState_.restart.store(true);
        }
        catch (...)
        {
            // Invalid input, keep old value
            snprintf(simulationState_.seedInputBuffer, sizeof(simulationState_.seedInputBuffer),
                     "%u", simulationState_.randomSeed.load());
        }
    }

    ImGui::SameLine();

    if (ImGui::Button("Random Seed"))
    {
        unsigned int newSeed = static_cast<unsigned int>(time(nullptr));
        simulationState_.randomSeed.store(newSeed);
        simulationState_.seedWasChanged = true;
        snprintf(simulationState_.seedInputBuffer, sizeof(simulationState_.seedInputBuffer),
                 "%u", newSeed);
        simulationState_.restart.store(true);
    }
}

void SimulationUIManager::renderAdvancedOptions()
{
    SimulationMethod currentMethod = simulationState_.simulationMethod.load();

    // CPU Parallelization Options
    if (currentMethod == SimulationMethod::CPU_DIRECT_SUM ||
        currentMethod == SimulationMethod::CPU_BARNES_HUT)
    {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "CPU Parallelization Options");

        bool useOpenMP = simulationState_.useOpenMP.load();
        if (ImGui::Checkbox("Enable OpenMP Multithreading", &useOpenMP))
        {
            simulationState_.useOpenMP.store(useOpenMP);
            simulationState_.restart.store(true);
        }

        if (useOpenMP)
        {
            int maxThreads = omp_get_max_threads();
            int threads = simulationState_.openMPThreads.load();

            // Ensure thread count is valid
            if (threads <= 0 || threads > maxThreads)
            {
                threads = maxThreads;
                simulationState_.openMPThreads.store(threads);
            }

            if (ImGui::SliderInt("Thread Count", &threads, 1, maxThreads))
            {
                simulationState_.openMPThreads.store(threads);
                simulationState_.restart.store(true);
            }

            ImGui::TextWrapped("Available CPU threads: %d", maxThreads);
        }

        ImGui::Separator();
    }

    // Space-Filling Curve Options
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Space-Filling Curve Options");

    // Determine if SFC options should be shown based on the method
    bool showSFCOptions = (currentMethod == SimulationMethod::BARNES_HUT ||
                           currentMethod == SimulationMethod::SFC_BARNES_HUT);

    if (!showSFCOptions)
    {
        ImGui::TextWrapped("Space-Filling Curve options are only applicable to Barnes-Hut methods.");
        return;
    }

    // SFC Toggle for regular Barnes-Hut
    if (currentMethod == SimulationMethod::BARNES_HUT)
    {
        bool sfcEnabled = simulationState_.useSFC.load();
        if (ImGui::Checkbox("Enable Space-Filling Curve", &sfcEnabled))
        {
            simulationState_.useSFC.store(sfcEnabled);
            simulationState_.restart.store(true);
        }

        if (!sfcEnabled)
        {
            ImGui::TextWrapped("Enable Space-Filling Curve to access ordering options.");
            return;
        }
    }

    // SFC is either enabled by user or always on for SFC Barnes-Hut

    // SFC Ordering Mode
    static const char *orderingModes[] = {"Particle Ordering", "Octant Ordering"};
    int currentMode = static_cast<int>(simulationState_.sfcOrderingMode.load());

    if (ImGui::Combo("Ordering Mode", &currentMode, orderingModes, IM_ARRAYSIZE(orderingModes)))
    {
        simulationState_.sfcOrderingMode.store(static_cast<SFCOrderingMode>(currentMode));
        simulationState_.restart.store(true);
    }

    // Ordering mode descriptions
    if (currentMode == 0)
    { // Particle ordering
        ImGui::TextWrapped("Particle Ordering: Bodies are sorted according to their position along a Space-Filling Curve to improve memory locality.");
    }
    else
    { // Octant ordering
        ImGui::TextWrapped("Octant Ordering: Tree nodes are arranged according to a Space-Filling Curve to improve traversal efficiency.");
    }

    // Reorder Frequency Slider
    int reorderFreq = simulationState_.reorderFrequency.load();
    if (ImGui::SliderInt("Reorder Frequency", &reorderFreq, 1, 100, "Every %d iterations"))
    {
        simulationState_.reorderFrequency.store(reorderFreq);
    }
    ImGui::TextWrapped("How often to recalculate the space-filling curve. Lower values improve accuracy but reduce performance.");
}