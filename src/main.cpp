#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <functional>
#include <fstream>
#include <vector>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>

#include "../include/common/constants.cuh"

#include "../include/simulation/base/base.cuh"
#include "../include/simulation/simulation_thread.hpp"

#include "../include/simulation/implementations/cpu/direct_sum.hpp"
#include "../include/simulation/implementations/cpu/barnes_hut.hpp"
#include "../include/simulation/implementations/cpu/sfc_variants.hpp"

#include "../include/simulation/implementations/gpu/direct_sum.cuh"
#include "../include/simulation/implementations/gpu/barnes_hut.cuh"
#include "../include/simulation/implementations/gpu/sfc_variants.cuh"

#include "../include/ui/simulation_state.hpp"
#include "../include/ui/opengl_renderer.hpp"
#include "../include/ui/simulation_ui_manager.hpp"

// Define the global variables
double g_theta = 0.5; // Default theta value
int g_blockSize = 256; // Default block size

// NVIDIA GPU selection hint for Linux
extern "C"
{
    __attribute__((visibility("default"))) int NvOptimusEnablement = 1;
}

// Logging function
void logMessage(const std::string &message, bool isError = false)
{
    std::ostream &stream = isError ? std::cerr : std::cout;
    stream << "[" << (isError ? "ERROR" : "INFO") << "] " << message << std::endl;
}

// Configuration structure
struct SimulationConfig
{
    int initialBodies = 1024;
    int sortType = 0;        // 0: none, 1: hilbert, 2: morton
    int numSteps = 1000;     // Number of simulation steps
    int massDistribution = 0; // 0: uniform, 1: normal
    int algorithm = 0;       // 0: cpu-direct-sum, 1: cpu-barnes, 2: gpu-direct-sum, 3: gpu-barnes-hut
    float theta = 0.5f;      // Barnes-Hut parameter
    bool visualization = true; // 0: off, 1: on
    std::string energyOutput = ""; // Output file for energy data
    int numThreads = 1;      // Number of threads for CPU implementations
    int blockSize = 256;     // Block size for GPU implementations
    bool fullscreen = true;
    bool useSFC = false;     // Used for space-filling curve options
    bool verbose = false;
};

// Parse command-line arguments
SimulationConfig parseArgs(int argc, char **argv)
{
    SimulationConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-n" && i + 1 < argc) {
            config.initialBodies = std::stoi(argv[++i]);
        }
        else if (arg == "-sort" && i + 1 < argc) {
            config.sortType = std::stoi(argv[++i]);
            config.useSFC = (config.sortType > 0); // Enable SFC if using Hilbert or Morton
        }
        else if (arg == "-steps" && i + 1 < argc) {
            config.numSteps = std::stoi(argv[++i]);
        }
        else if (arg == "-mdist" && i + 1 < argc) {
            config.massDistribution = std::stoi(argv[++i]);
        }
        else if (arg == "-alg" && i + 1 < argc) {
            config.algorithm = std::stoi(argv[++i]);
        }
        else if (arg == "-theta" && i + 1 < argc) {
            config.theta = std::stof(argv[++i]);
        }
        else if (arg == "-visual" && i + 1 < argc) {
            config.visualization = (std::stoi(argv[++i]) != 0);
            config.fullscreen = config.visualization; // Only fullscreen if visualization is enabled
        }
        else if (arg == "-energy" && i + 1 < argc) {
            config.energyOutput = argv[++i];
        }
        else if (arg == "-nt" && i + 1 < argc) {
            config.numThreads = std::stoi(argv[++i]);
        }
        else if (arg == "-bs" && i + 1 < argc) {
            config.blockSize = std::stoi(argv[++i]);
        }
        else if (arg == "-verbose") {
            config.verbose = true;
        }
        else if (arg == "-help" || arg == "--help" || arg == "-h") {
            std::cout << "N-Body Simulation Usage:\n"
                      << "  ./prog [options]\n"
                      << "Options:\n"
                      << "  -n <particles>      : Number of particles (default: 1024)\n"
                      << "  -sort <type>        : Space-filling curve type (0: none, 1: hilbert, 2: morton) (default: 0)\n"
                      << "  -steps <steps>      : Number of simulation steps (default: 1000)\n"
                      << "  -mdist <type>       : Mass distribution (0: uniform, 1: normal) (default: 0)\n"
                      << "  -alg <algorithm>    : Algorithm (0: cpu-direct-sum, 1: cpu-barnes, 2: gpu-direct-sum, 3: gpu-barnes-hut) (default: 0)\n"
                      << "  -theta <float>      : Barnes-Hut theta parameter (default: 0.5)\n"
                      << "  -visual <0|1>       : Enable visualization (0: off, 1: on) (default: 1)\n"
                      << "  -energy <filename>  : Output energy data to file\n"
                      << "  -nt <threads>       : Number of threads for CPU algorithms (default: 1)\n"
                      << "  -bs <blocksize>     : Block size for GPU algorithms (default: 256)\n"
                      << "  -verbose            : Enable verbose output\n"
                      << "  -help, --help, -h   : Show this help message\n";
            exit(0);
        }
    }

    if (config.verbose) {
        std::cout << "Configuration:\n"
                  << "  Particles: " << config.initialBodies << "\n"
                  << "  Sort Type: " << config.sortType << "\n"
                  << "  Steps: " << config.numSteps << "\n"
                  << "  Mass Distribution: " << config.massDistribution << "\n"
                  << "  Algorithm: " << config.algorithm << "\n"
                  << "  Theta: " << config.theta << "\n"
                  << "  Visualization: " << (config.visualization ? "On" : "Off") << "\n"
                  << "  Energy Output: " << (config.energyOutput.empty() ? "None" : config.energyOutput) << "\n"
                  << "  Threads: " << config.numThreads << "\n"
                  << "  Block Size: " << config.blockSize << "\n";
    }

    return config;
}

// Global simulation state for callbacks
SimulationState *g_simulationState = nullptr;

// GLFW error callback
void glfw_error_callback(int error, const char *description)
{
    logMessage("GLFW Error: " + std::string(description), true);
}

// OpenGL debug callback
void APIENTRY glDebugOutput(GLenum source, GLenum type, GLuint id, GLenum severity,
                            GLsizei length, const GLchar *message, const void *userParam)
{
    // Ignore non-significant error/warning codes
    if (id == 131169 || id == 131185 || id == 131218 || id == 131204)
        return;

    std::cout << "OpenGL Debug: " << message << std::endl;
}

// Keyboard callback function to handle ESC key to exit the simulation
void key_callback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    // Check if escape key was pressed
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
    {
        // Set the window to close and stop the simulation
        glfwSetWindowShouldClose(window, GLFW_TRUE);

        // If we have access to the simulation state, also set running to false
        if (g_simulationState != nullptr)
        {
            g_simulationState->running.store(false);
        }
    }
}

// Initialize GLFW and create window
GLFWwindow *initializeGLFW(const SimulationConfig &config)
{
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
    {
        logMessage("Failed to initialize GLFW", true);
        return nullptr;
    }

    // OpenGL and window hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);

    // Get primary monitor and video mode
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);

    // Create window
    GLFWwindow *window = nullptr;
    if (config.fullscreen)
    {
        window = glfwCreateWindow(
            mode->width,
            mode->height,
            "N-Body Simulation",
            monitor, // Fullscreen mode
            nullptr);
    }
    else
    {
        window = glfwCreateWindow(
            1280,
            720,
            "N-Body Simulation",
            nullptr, // Windowed mode
            nullptr);
    }

    if (!window)
    {
        logMessage("Failed to create GLFW window", true);
        glfwTerminate();
        return nullptr;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    glfwSetKeyCallback(window, key_callback);

    return window;
}

// Initialize GLAD for OpenGL function loading
bool initializeGLAD()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        logMessage("Failed to initialize GLAD", true);
        return false;
    }

    logMessage("OpenGL Version: " + std::string((char *)glGetString(GL_VERSION)));
    logMessage("GLSL Version: " + std::string((char *)glGetString(GL_SHADING_LANGUAGE_VERSION)));
    logMessage("Renderer: " + std::string((char *)glGetString(GL_RENDERER)));

    return true;
}

// Setup ImGui context and style
void setupImGui(GLFWwindow *window)
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    (void)io;

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
}

// Main render loop
void renderLoop(GLFWwindow *window, SimulationState &simulationState, SimulationThread &simThread, OpenGLRenderer &renderer, SimulationUIManager &uiManager)
{
    int frameCounter = 0;
    const int OCTREE_UPDATE_FREQ = 5; // Actualizar visualización del octree cada N frames

    while (!glfwWindowShouldClose(window) && simulationState.running.load())
    {
        // Poll and handle events
        glfwPollEvents();

        // Render bodies if available
        {
            std::lock_guard<std::mutex> lock(simulationState.mtx);
            if (simulationState.sharedBodies && simulationState.currentBodiesCount > 0)
            {
                renderer.updateBodies(
                    simulationState.sharedBodies,
                    simulationState.currentBodiesCount);
            }
        }

        // Get window dimensions
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

        // Render bodies and octree
        renderer.render(aspectRatio);

        // Render UI
        uiManager.renderUI(window);

        // Swap front and back buffers
        glfwSwapBuffers(window);
    }
}

int main(int argc, char **argv)
{
    try
    {
        std::cout << "Attempting to use dedicated GPU..." << std::endl;
        checkCudaAvailability();
        // Parse command-line arguments
        SimulationConfig config = parseArgs(argc, argv);

        // Initialize GLFW and create window
        GLFWwindow *window = initializeGLFW(config);
        if (!window)
            return -1;

        // Initialize GLAD
        if (!initializeGLAD())
            return -1;

        // Configure OpenGL
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Setup ImGui
        setupImGui(window);

        // Create simulation state and apply all command-line parameters
        SimulationState simulationState;
        
        // Basic parameters
        simulationState.numBodies.store(config.initialBodies);
        simulationState.useSFC.store(config.useSFC);
        
        // Set sorting method based on command-line parameter
        if (config.sortType > 0) {
            // Set the SFC curve type
            simulationState.sfcCurveType.store(config.sortType == 1 ? 
                sfc::CurveType::HILBERT : sfc::CurveType::MORTON);
        }
        
        // Set the distribution type
        simulationState.massDistribution.store(config.massDistribution == 0 ? 
            MassDistribution::UNIFORM : MassDistribution::NORMAL);
        
        // Set the algorithm type
        switch (config.algorithm) {
            case 0: // CPU direct sum
                simulationState.simulationMethod.store(SimulationMethod::CPU_DIRECT_SUM);
                break;
            case 1: // CPU Barnes-Hut
                simulationState.simulationMethod.store(SimulationMethod::CPU_BARNES_HUT);
                break;
            case 2: // GPU direct sum
                simulationState.simulationMethod.store(SimulationMethod::GPU_DIRECT_SUM);
                break;
            case 3: // GPU Barnes-Hut
                simulationState.simulationMethod.store(SimulationMethod::GPU_BARNES_HUT);
                break;
        }
        
        // Set the Barnes-Hut theta parameter if applicable
        if (config.algorithm == 1 || config.algorithm == 3) {
            // Set the global theta parameter
            g_theta = config.theta;
            logMessage("Using Barnes-Hut with theta: " + std::to_string(g_theta));
        } else {
            // Reset to default for non-Barnes-Hut algorithms
            g_theta = 0.5; // Default theta value
        }
        
        // Set the block size for GPU kernels
        g_blockSize = config.blockSize;
        logMessage("Using CUDA block size: " + std::to_string(g_blockSize));
        
        // Set thread count for CPU implementations
        simulationState.openMPThreads.store(config.numThreads);
        simulationState.useOpenMP.store(config.numThreads > 1);
        
        // Initialize variables for octree visualization
        simulationState.showOctree = (config.algorithm == 1 || config.algorithm == 3); // Show octree for Barnes-Hut
        simulationState.octreeMaxDepth = 3;
        simulationState.octreeOpacity = 0.5f;
        simulationState.octreeColorByMass = true;

        // Create OpenGL renderer
        OpenGLRenderer renderer(simulationState);
        renderer.init();

        // Create UI manager
        SimulationUIManager uiManager(simulationState, renderer);

        SimulationThread simulationThread(&simulationState);
        simulationThread.start();

        g_simulationState = &simulationState;

        // Main render loop
        // Open energy output file if specified
        std::ofstream energyOutput;
        if (!config.energyOutput.empty()) {
            energyOutput.open(config.energyOutput);
            if (energyOutput.is_open()) {
                logMessage("Energy output will be written to: " + config.energyOutput);
                // Write header
                energyOutput << "Step,Time,KineticEnergy,PotentialEnergy,TotalEnergy" << std::endl;
            } else {
                logMessage("Failed to open energy output file: " + config.energyOutput, true);
            }
        }

        // If visualization is enabled, run the render loop
        if (config.visualization) {
            renderLoop(window, simulationState, simulationThread, renderer, uiManager);
        }
        // Otherwise, run the simulation for the specified number of steps
        else {
            logMessage("Running simulation without visualization for " + std::to_string(config.numSteps) + " steps");
            
            // Wait a moment for the simulation to initialize
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            // Run for the specified number of steps
            for (int step = 0; step < config.numSteps && simulationState.running.load(); ++step) {
                // Get current simulation data for energy calculation
                if (!config.energyOutput.empty() && energyOutput.is_open()) {
                    SimulationData simData = simulationThread.getSimulationData();
                    if (simData.valid && simData.simulation) {
                        // Calculate energies
                        double kineticEnergy = simData.simulation->getKineticEnergy();
                        double potentialEnergy = simData.simulation->getPotentialEnergy();
                        double totalEnergy = kineticEnergy + potentialEnergy;
                        
                        // Write to output file
                        energyOutput << step << "," 
                                    << simulationState.lastIterationTime << ","
                                    << kineticEnergy << ","
                                    << potentialEnergy << ","
                                    << totalEnergy << std::endl;
                    }
                }
                
                // Print progress every 10% of steps
                if (step % (config.numSteps / 10) == 0 || step == config.numSteps - 1) {
                    logMessage("Simulation progress: " + std::to_string(step + 1) + "/" + 
                              std::to_string(config.numSteps) + " steps");
                }
                
                // Sleep briefly to avoid consuming 100% CPU while getting status
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        // Close energy output file if opened
        if (energyOutput.is_open()) {
            energyOutput.close();
        }

        // Cleanup
        simulationState.running.store(false);
        simulationThread.join();

        // Shutdown ImGui
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        // Terminate GLFW
        glfwDestroyWindow(window);
        glfwTerminate();

        return 0;
    }
    catch (const std::exception &e)
    {
        logMessage("Fatal error: " + std::string(e.what()), true);
        return 1;
    }
}