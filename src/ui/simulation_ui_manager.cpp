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

        if (ImGui::BeginTabItem("Octree View"))
        {
            renderOctreeVisualizationOptions();
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
        "Random Clusters",
        "Random Bodies"};

    int currentDist = static_cast<int>(simulationState_.bodyDistribution.load());
    if (ImGui::Combo("Distribution Type", &currentDist, distributionTypes, IM_ARRAYSIZE(distributionTypes)))
    {
        simulationState_.bodyDistribution.store(static_cast<BodyDistribution>(currentDist));
        simulationState_.restart.store(true);
    }

    // Mass distribution selection
    static const char *massDistributionTypes[] = {
        "Uniform (All Equal)",
        "Normal Distribution"};

    int currentMassDist = static_cast<int>(simulationState_.massDistribution.load());
    if (ImGui::Combo("Mass Distribution", &currentMassDist, massDistributionTypes, IM_ARRAYSIZE(massDistributionTypes)))
    {
        simulationState_.massDistribution.store(static_cast<MassDistribution>(currentMassDist));
        simulationState_.restart.store(true);
    }

    // Show description based on selected mass distribution
    switch (static_cast<MassDistribution>(currentMassDist)) {
        case MassDistribution::UNIFORM:
            ImGui::TextWrapped("Uniform: All bodies have exactly the same mass. Good for studying pure positional effects.");
            break;
        case MassDistribution::NORMAL:
            ImGui::TextWrapped("Normal: Masses follow a normal (Gaussian) distribution. Provides moderate variation.");
            break;
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

    // Simplified method selection with just 4 main options
    static const char *methodTypes[] = {
        "Direct Sum (CPU)",
        "Direct Sum (GPU)",
        "Barnes-Hut (CPU)",
        "Barnes-Hut (GPU)"};

    // Convert enum value to simplified index
    int currentMethod = static_cast<int>(simulationState_.simulationMethod.load());
    int simplifiedMethod;

    // Map the original enum values to our simplified selection
    switch (currentMethod)
    {
    case 0:                   // CPU_DIRECT_SUM
    case 1:                   // CPU_SFC_DIRECT_SUM
        simplifiedMethod = 0; // Direct Sum (CPU)
        break;
    case 2:                   // GPU_DIRECT_SUM
    case 3:                   // GPU_SFC_DIRECT_SUM
        simplifiedMethod = 1; // Direct Sum (GPU)
        break;
    case 4:                   // CPU_BARNES_HUT
    case 5:                   // CPU_SFC_BARNES_HUT
        simplifiedMethod = 2; // Barnes-Hut (CPU)
        break;
    case 6: // GPU_BARNES_HUT
    case 7: // GPU_SFC_BARNES_HUT
    default:
        simplifiedMethod = 3; // Barnes-Hut (GPU)
        break;
    }

    if (ImGui::Combo("Algorithm", &simplifiedMethod, methodTypes, IM_ARRAYSIZE(methodTypes)))
    {
        // Set the selected simulation method, preserving SFC settings
        bool useSFC = simulationState_.useSFC.load();

        // Map simplified selection back to the appropriate enum value
        SimulationMethod newMethod;

        switch (simplifiedMethod)
        {
        case 0: // Direct Sum (CPU)
            newMethod = useSFC ? SimulationMethod::CPU_SFC_DIRECT_SUM : SimulationMethod::CPU_DIRECT_SUM;
            break;
        case 1: // Direct Sum (GPU)
            newMethod = useSFC ? SimulationMethod::GPU_SFC_DIRECT_SUM : SimulationMethod::GPU_DIRECT_SUM;
            break;
        case 2: // Barnes-Hut (CPU)
            newMethod = useSFC ? SimulationMethod::CPU_SFC_BARNES_HUT : SimulationMethod::CPU_BARNES_HUT;
            break;
        case 3: // Barnes-Hut (GPU)
        default:
            newMethod = SimulationMethod::GPU_BARNES_HUT;
            break;
        }

        simulationState_.simulationMethod.store(newMethod);
        simulationState_.restart.store(true);
    }

    // Add method descriptions
    ImGui::Spacing();
    ImGui::TextWrapped("Method Description:");

    switch (simplifiedMethod)
    {
    case 0: // Direct Sum (CPU)
        ImGui::TextWrapped("CPU Direct Sum: Computes all body-to-body interactions directly (O(n²) complexity). Good for smaller simulations.");
        break;

    case 1: // Direct Sum (GPU)
        ImGui::TextWrapped("GPU Direct Sum: Leverages GPU parallelism for direct body-to-body calculations. Good for medium-sized simulations.");
        break;

    case 2: // Barnes-Hut (CPU)
        ImGui::TextWrapped("CPU Barnes-Hut: Uses an octree to approximate distant interactions, reducing complexity to O(n log n). Good for larger simulations.");
        break;

    case 3: // Barnes-Hut (GPU)
        ImGui::TextWrapped("GPU Barnes-Hut: Implements Barnes-Hut algorithm on GPU for high-performance large-scale simulations.");
        break;
    }
}

// Fix for renderAdvancedOptions()
void SimulationUIManager::renderAdvancedOptions()
{
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Advanced Options");

    // CPU Parallelization Options - show for any CPU method
    SimulationMethod currentMethod = simulationState_.simulationMethod.load();
    bool isCPUMethod = (currentMethod == SimulationMethod::CPU_DIRECT_SUM ||
                        currentMethod == SimulationMethod::CPU_SFC_DIRECT_SUM ||
                        currentMethod == SimulationMethod::CPU_BARNES_HUT ||
                        currentMethod == SimulationMethod::CPU_SFC_BARNES_HUT);

    if (isCPUMethod)
    {
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "CPU Parallelization");

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

    // Space-Filling Curve Options - show for all methods
    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Space-Filling Curve Options");

    // SFC Toggle
    bool sfcEnabled = simulationState_.useSFC.load();
    if (ImGui::Checkbox("Enable Space-Filling Curve", &sfcEnabled))
    {
        // Update the useSFC flag
        simulationState_.useSFC.store(sfcEnabled);

        // Update the method to match the appropriate SFC/non-SFC variant
        SimulationMethod baseMethod = currentMethod;
        SimulationMethod newMethod;

        // Convert to the SFC or non-SFC version of the current method
        switch (baseMethod)
        {
        case SimulationMethod::CPU_DIRECT_SUM:
        case SimulationMethod::CPU_SFC_DIRECT_SUM:
            newMethod = sfcEnabled ? SimulationMethod::CPU_SFC_DIRECT_SUM : SimulationMethod::CPU_DIRECT_SUM;
            break;

        case SimulationMethod::GPU_DIRECT_SUM:
        case SimulationMethod::GPU_SFC_DIRECT_SUM:
            newMethod = sfcEnabled ? SimulationMethod::GPU_SFC_DIRECT_SUM : SimulationMethod::GPU_DIRECT_SUM;
            break;

        case SimulationMethod::CPU_BARNES_HUT:
        case SimulationMethod::CPU_SFC_BARNES_HUT:
            newMethod = sfcEnabled ? SimulationMethod::CPU_SFC_BARNES_HUT : SimulationMethod::CPU_BARNES_HUT;
            break;

        case SimulationMethod::GPU_BARNES_HUT:
        default:
            newMethod = SimulationMethod::GPU_BARNES_HUT;
            break;
        }

        simulationState_.simulationMethod.store(newMethod);
        simulationState_.restart.store(true);
    }

    if (sfcEnabled)
    {
        ImGui::TextWrapped("SFC improves memory access patterns by organizing data along a space-filling curve.");

        // Curve Type Selection
        static const char *curveTypes[] = {"Morton (Z-order)", "Hilbert"};
        int curveTypeIndex = (simulationState_.sfcCurveType.load() == sfc::CurveType::MORTON) ? 0 : 1;

        if (ImGui::Combo("Curve Type", &curveTypeIndex, curveTypes, IM_ARRAYSIZE(curveTypes)))
        {
            simulationState_.sfcCurveType.store(curveTypeIndex == 0 ? sfc::CurveType::MORTON : sfc::CurveType::HILBERT);
            simulationState_.restart.store(true);
        }

        // Show curve type descriptions
        if (curveTypeIndex == 0)
        {
            ImGui::TextWrapped("Morton/Z-order: Simple bit-interleaving curve. Faster to compute but less spatial coherence.");
        }
        else
        {
            ImGui::TextWrapped("Hilbert curve: Better spatial coherence (neighboring points stay closer), slightly more expensive to compute.");
        }

        // Only show ordering mode for Barnes-Hut methods
        bool isBarnesHut = (currentMethod == SimulationMethod::CPU_BARNES_HUT ||
                            currentMethod == SimulationMethod::CPU_SFC_BARNES_HUT ||
                            currentMethod == SimulationMethod::GPU_BARNES_HUT);

        if (isBarnesHut)
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
            simulationState_.restart.store(true);
        }
        ImGui::TextWrapped("How often to recalculate the space-filling curve ordering.");
    }
    else
    {
        ImGui::TextWrapped("Enable Space-Filling Curve to access ordering options.");
    }
}

void SimulationUIManager::renderSFCOptions()
{
    SimulationMethod currentMethod = simulationState_.simulationMethod.load();

    // Only show SFC options for methods that support it
    bool supportsParticleOrdering = (currentMethod == SimulationMethod::CPU_SFC_DIRECT_SUM ||
                                     currentMethod == SimulationMethod::GPU_SFC_DIRECT_SUM ||
                                     currentMethod == SimulationMethod::CPU_SFC_BARNES_HUT ||
                                     currentMethod == SimulationMethod::CPU_BARNES_HUT ||
                                     currentMethod == SimulationMethod::GPU_BARNES_HUT);

    bool supportsOctantOrdering = (currentMethod == SimulationMethod::CPU_SFC_BARNES_HUT ||
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

void SimulationUIManager::renderOctreeVisualizationOptions()
{
    // Determinar si el método actual usa octree
    SimulationMethod currentMethod = simulationState_.simulationMethod.load();
    bool usesOctree = (currentMethod == SimulationMethod::CPU_BARNES_HUT ||
                       currentMethod == SimulationMethod::CPU_SFC_BARNES_HUT ||
                       currentMethod == SimulationMethod::GPU_BARNES_HUT);

    if (!usesOctree)
    {
        ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.5f, 1.0f),
                           "Octree visualization is only available for Barnes-Hut methods");
        return;
    }

    ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Octree Visualization");

    // Toggle para mostrar/ocultar octree
    bool showOctree = simulationState_.showOctree;
    if (ImGui::Checkbox("Show Octree Structure", &showOctree))
    {
        simulationState_.showOctree = showOctree;
    }

    ImGui::TextWrapped("Toggle octree visualization without affecting particle display");

    if (showOctree)
    {
        // Control de profundidad máxima
        int maxDepth = simulationState_.octreeMaxDepth;
        if (ImGui::SliderInt("Max Depth", &maxDepth, 1, 8))
        {
            simulationState_.octreeMaxDepth = maxDepth;
        }
        ImGui::TextWrapped("Lower depths show less detail but render faster");

        // Control de opacidad
        float opacity = simulationState_.octreeOpacity;
        if (ImGui::SliderFloat("Opacity", &opacity, 0.1f, 1.0f, "%.2f"))
        {
            simulationState_.octreeOpacity = opacity;
        }
        ImGui::TextWrapped("Adjust transparency of the octree lines");

        // Toggle para colorear por masa
        bool colorByMass = simulationState_.octreeColorByMass;
        if (ImGui::Checkbox("Color Nodes by Mass", &colorByMass))
        {
            simulationState_.octreeColorByMass = colorByMass;
        }

        if (colorByMass)
        {
            ImGui::TextWrapped("Red nodes contain more mass");
        }
        else
        {
            ImGui::TextWrapped("Uniform color for all octree nodes");
        }

        // Información sobre el octree
        ImGui::Separator();
        ImGui::TextColored(ImVec4(0.4f, 0.8f, 0.4f, 1.0f), "Octree Information");
        ImGui::Text("Bodies in simulation: %d", simulationState_.numBodies.load());

        // Nota: Aquí se podría añadir información real sobre el octree si estuviera disponible
        int approxDepth = static_cast<int>(log2(simulationState_.numBodies.load() / 8) + 1);
        ImGui::Text("Approximate depth: ~%d", approxDepth);
        ImGui::Text("Approximate node count: ~%d", (1 << (3 * approxDepth)) / 7); // Estimación basada en octree completo
    }
}
