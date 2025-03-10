#include "../../include/ui/simulation_ui_manager.hpp"
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

        // Tab 4: Advanced options (SFC, OpenMP)
        if (ImGui::BeginTabItem("Advanced Options"))
        {
            renderAdvancedOptions();
            ImGui::EndTabItem();
        }

        // Tab 5: SFC-specific options (optionally, if you want a dedicated tab)
        if (ImGui::BeginTabItem("SFC Options"))
        {
            renderSFCOptions();
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

void SimulationUIManager::renderVisualizationOptions()
{
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Visualization Settings");

    // Particle size slider
    static float particleSize = renderer.getParticleSize();
    if (ImGui::SliderFloat("Particle Size", &particleSize, 1.0f, 20.0f, "%.1f"))
    {
        renderer.setParticleSize(particleSize);
    }
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

void SimulationUIManager::renderSimulationMethodSelector()
{
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Select Simulation Algorithm");

    // Method selection with the complete list of algorithms including SFC variants
    static const char *methodTypes[] = {
        "CPU Direct Sum",
        "CPU SFC Direct Sum",
        "GPU Direct Sum",
        "GPU SFC Direct Sum",
        "CPU Barnes-Hut",
        "CPU SFC Barnes-Hut",
        "GPU Barnes-Hut",
        "GPU SFC Barnes-Hut"};

    // The enum values now directly correspond to array indices
    int currentMethod = static_cast<int>(simulationState_.simulationMethod.load());

    if (ImGui::Combo("Algorithm", &currentMethod, methodTypes, IM_ARRAYSIZE(methodTypes)))
    {
        // Set the selected simulation method
        simulationState_.simulationMethod.store(static_cast<SimulationMethod>(currentMethod));

        // Reset options based on new method selected
        switch (static_cast<SimulationMethod>(currentMethod))
        {
        case SimulationMethod::CPU_DIRECT_SUM:
        case SimulationMethod::CPU_BARNES_HUT:
            // Default options for CPU methods
            simulationState_.useOpenMP.store(true);
            simulationState_.openMPThreads.store(omp_get_max_threads());
            simulationState_.useSFC.store(false);
            break;

        case SimulationMethod::CPU_SFC_DIRECT_SUM:
        case SimulationMethod::CPU_SFC_BARNES_HUT:
            // Default options for CPU SFC methods
            simulationState_.useOpenMP.store(true);
            simulationState_.openMPThreads.store(omp_get_max_threads());
            simulationState_.useSFC.store(true);
            break;

        case SimulationMethod::GPU_DIRECT_SUM:
        case SimulationMethod::GPU_BARNES_HUT:
            // Default options for basic GPU methods
            simulationState_.useSFC.store(false);
            break;

        case SimulationMethod::GPU_SFC_DIRECT_SUM:
        case SimulationMethod::GPU_SFC_BARNES_HUT:
            // Default options for GPU SFC methods
            simulationState_.useSFC.store(true);
            break;
        }

        simulationState_.restart.store(true);
    }

    // Add method descriptions
    ImGui::Spacing();
    ImGui::TextWrapped("Method Description:");

    switch (static_cast<SimulationMethod>(currentMethod))
    {
    case SimulationMethod::CPU_DIRECT_SUM:
        ImGui::TextWrapped("CPU Direct Sum: Computes all body-to-body interactions directly (O(n²) complexity). Good for smaller simulations.");
        break;

    case SimulationMethod::CPU_SFC_DIRECT_SUM:
        ImGui::TextWrapped("CPU SFC Direct Sum: Enhances Direct Sum with Space-Filling Curve for better cache utilization and memory access patterns.");
        break;

    case SimulationMethod::GPU_DIRECT_SUM:
        ImGui::TextWrapped("GPU Direct Sum: Leverages GPU parallelism for direct body-to-body calculations. Good for medium-sized simulations.");
        break;

    case SimulationMethod::GPU_SFC_DIRECT_SUM:
        ImGui::TextWrapped("GPU SFC Direct Sum: Combines GPU parallelism with Space-Filling Curve for improved memory access coherence.");
        break;

    case SimulationMethod::CPU_BARNES_HUT:
        ImGui::TextWrapped("CPU Barnes-Hut: Uses an octree to approximate distant interactions, reducing complexity to O(n log n). Good for larger simulations.");
        break;

    case SimulationMethod::CPU_SFC_BARNES_HUT:
        ImGui::TextWrapped("CPU SFC Barnes-Hut: Enhances Barnes-Hut with Space-Filling Curve for better cache performance and memory access patterns.");
        break;

    case SimulationMethod::GPU_BARNES_HUT:
        ImGui::TextWrapped("GPU Barnes-Hut: Implements Barnes-Hut algorithm on GPU for high-performance large-scale simulations.");
        break;

    case SimulationMethod::GPU_SFC_BARNES_HUT:
        ImGui::TextWrapped("GPU SFC Barnes-Hut: The most advanced algorithm, combining GPU Barnes-Hut with Space-Filling Curve optimization for maximum performance with very large simulations.");
        break;
    }
}

// Fix for renderAdvancedOptions()
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

    // Show SFC options for Barnes-Hut methods only
    bool isBarnesHut = (currentMethod == SimulationMethod::CPU_BARNES_HUT ||
                        currentMethod == SimulationMethod::GPU_BARNES_HUT);

    if (!isBarnesHut)
    {
        ImGui::TextWrapped("Space-Filling Curve options are only applicable to Barnes-Hut methods.");
        return;
    }

    // SFC Toggle
    bool sfcEnabled = simulationState_.useSFC.load();
    if (ImGui::Checkbox("Enable Space-Filling Curve", &sfcEnabled))
    {
        simulationState_.useSFC.store(sfcEnabled);
        simulationState_.restart.store(true);
    }

    if (sfcEnabled)
    {
        ImGui::TextWrapped("SFC improves memory access patterns by organizing data along a space-filling curve.");

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
    else
    {
        ImGui::TextWrapped("Enable Space-Filling Curve to access ordering options.");
    }

    renderSFCOptions();
}

void SimulationUIManager::renderSFCOptions()
{
    SimulationMethod currentMethod = simulationState_.simulationMethod.load();

    // Only show SFC options for methods that support it
    bool supportsParticleOrdering = (currentMethod == SimulationMethod::CPU_SFC_DIRECT_SUM ||
                                     currentMethod == SimulationMethod::GPU_SFC_DIRECT_SUM ||
                                     currentMethod == SimulationMethod::CPU_SFC_BARNES_HUT ||
                                     currentMethod == SimulationMethod::GPU_SFC_BARNES_HUT ||
                                     currentMethod == SimulationMethod::CPU_BARNES_HUT ||
                                     currentMethod == SimulationMethod::GPU_BARNES_HUT);

    bool supportsOctantOrdering = (currentMethod == SimulationMethod::CPU_SFC_BARNES_HUT ||
                                   currentMethod == SimulationMethod::GPU_SFC_BARNES_HUT ||
                                   currentMethod == SimulationMethod::CPU_BARNES_HUT ||
                                   currentMethod == SimulationMethod::GPU_BARNES_HUT);

    if (!supportsParticleOrdering && !supportsOctantOrdering)
    {
        ImGui::TextWrapped("Space-Filling Curve options are not applicable to the current simulation method.");
        return;
    }

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Space-Filling Curve Options");

    // SFC Toggle
    bool sfcEnabled = simulationState_.useSFC.load();
    if (ImGui::Checkbox("Enable Space-Filling Curve", &sfcEnabled))
    {
        simulationState_.useSFC.store(sfcEnabled);
        simulationState_.restart.store(true);
    }

    if (sfcEnabled)
    {
        ImGui::TextWrapped("SFC improves memory access patterns by organizing data along a space-filling curve.");

        // SFC Type Selection
        static const char *curveTypes[] = {"Morton (Z-order)", "Hilbert"};
        static int curveTypeIndex = 0; // Default to Morton

        if (ImGui::Combo("Curve Type", &curveTypeIndex, curveTypes, IM_ARRAYSIZE(curveTypes)))
        {
            // Store in simulation state - this will need to be added to SimulationState
            // Alternatively, add this in the GUI code without storage
            simulationState_.restart.store(true);
        }

        // Show appropriate description based on selection
        if (curveTypeIndex == 0)
        {
            ImGui::TextWrapped("Morton/Z-order: Simple bit-interleaving curve. Faster to compute but less spatial coherence.");
        }
        else
        {
            ImGui::TextWrapped("Hilbert curve: Better spatial coherence (neighboring points stay closer), slightly more expensive to compute.");
        }

        // Only show ordering mode for Barnes-Hut methods
        if (supportsOctantOrdering)
        {
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
                ImGui::TextWrapped("Particle Ordering: Bodies are sorted according to their position along a Space-Filling Curve.");
            }
            else
            { // Octant ordering
                ImGui::TextWrapped("Octant Ordering: Tree nodes are arranged according to a Space-Filling Curve.");
            }
        }

        // Reorder Frequency Slider
        int reorderFreq = simulationState_.reorderFrequency.load();
        if (ImGui::SliderInt("Reorder Frequency", &reorderFreq, 1, 100, "Every %d iterations"))
        {
            simulationState_.reorderFrequency.store(reorderFreq);
        }
        ImGui::TextWrapped("How often to recalculate the space-filling curve ordering.");

        // Performance metrics specific to SFC
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "SFC Performance Impact");

        // In a real implementation, you'd track these metrics in simulation_state
        float memorySavings = 0.0f;      // Percent memory bandwidth saved
        float cacheMissReduction = 0.0f; // Percent cache miss reduction

        ImGui::Text("Memory bandwidth saved: %.1f%%", memorySavings);
        ImGui::Text("Cache miss reduction: %.1f%%", cacheMissReduction);

        ImGui::TextWrapped("Note: These metrics would need to be measured in the simulation for accurate reporting.");
    }
    else
    {
        ImGui::TextWrapped("Enable Space-Filling Curve to access ordering options.");
    }
}