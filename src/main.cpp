#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <functional>
#include <fstream>
#include <vector>
#include <omp.h>

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

double g_theta = 0.5;
int g_blockSize = 256;

// NVIDIA GPU selection hint for Linux
extern "C"
{
    __attribute__((visibility("default"))) int NvOptimusEnablement = 1;
}

// Function to configure optimal CUDA parameters based on detected GPU
void configureGPUParameters()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        return;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    
    std::string gpuName = deviceProp.name;
    int computeCapability = deviceProp.major * 10 + deviceProp.minor;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int multiProcessorCount = deviceProp.multiProcessorCount;
    
    // Log the detected GPU information
    std::cout << "Detected GPU: " << gpuName << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Number of SMs: " << multiProcessorCount << std::endl;
    
    // Configure block size based on GPU architecture
    // For most modern GPUs, multiples of 32 (warp size) are ideal
    // RTX 40 series (Ada Lovelace) performs better with larger blocks
    if (computeCapability >= 89) {  // RTX 40 series (Ada Lovelace - SM 8.9)
        g_blockSize = 512;  // Larger blocks for Ada Lovelace
    } 
    else if (computeCapability >= 86) {  // RTX 30 series (Ampere - SM 8.6)
        g_blockSize = 384;
    }
    else if (computeCapability >= 75) {  // RTX 20 series (Turing - SM 7.5)
        g_blockSize = 256;
    }
    else {
        // For older GPUs, stick with the default 256
        g_blockSize = 256;
    }
    
    // Ensure block size is within device limits and is a multiple of 32
    g_blockSize = std::min(g_blockSize, maxThreadsPerBlock);
    g_blockSize = (g_blockSize / 32) * 32;  // Round to multiple of warp size
    
    std::cout << "Configured block size: " << g_blockSize << std::endl;
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
    int numSteps = 10000000;     // Number of simulation steps
    int massDistribution = 0; // 0: uniform, 1: normal
    int algorithm = 0;       // 0: cpu-direct-sum, 1: cpu-barnes, 2: gpu-direct-sum, 3: gpu-barnes-hut
    float theta = 0.5f;      // Barnes-Hut parameter
    bool visualization = true; // 0: off, 1: on
    std::string energyOutput = ""; // Output file for energy data
    int numThreads = omp_get_max_threads();      // Default to max threads for CPU implementations
    int blockSize = 256;     // Block size for GPU implementations
    bool fullscreen = true;
    bool useSFC = false;     // Used for space-filling curve options
    bool verbose = false;
    bool headless = false;   // Run without any UI or visualization
    bool reportMetrics = false; // Report detailed metrics at the end
    unsigned int randomSeed = 12345; // Random seed for reproducibility
    bool dynamicReordering = true; // Use dynamic reordering for Barnes-Hut SFC
    int metricsWindowSize = 10; // Window size for dynamic reordering metrics
};

// Parse command-line arguments
SimulationConfig parseArgs(int argc, char **argv)
{
    SimulationConfig config;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-n" && i + 1 < argc || arg == "--bodies" && i + 1 < argc) {
            config.initialBodies = std::stoi(argv[++i]);
        }
        else if (arg == "-sort" && i + 1 < argc) {
            config.sortType = std::stoi(argv[++i]);
            config.useSFC = (config.sortType > 0); // Enable SFC if using Hilbert or Morton
        }
        else if (arg == "-steps" && i + 1 < argc || arg == "--iterations" && i + 1 < argc) {
            config.numSteps = std::stoi(argv[++i]);
        }
        else if (arg == "-mdist" && i + 1 < argc) {
            config.massDistribution = std::stoi(argv[++i]);
        }
        else if (arg == "-alg" && i + 1 < argc || arg == "--method" && i + 1 < argc) {
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
        else if (arg == "--headless") {
            config.headless = true;
            config.visualization = false; // Headless mode disables visualization
        }
        else if (arg == "--report-metrics") {
            config.reportMetrics = true;
        }
        else if (arg == "--use-sfc" && i + 1 < argc) {
            config.useSFC = (std::string(argv[++i]) == "true" || std::string(argv[i]) == "1");
            if (config.useSFC) {
                // Default to Morton curve if SFC is enabled but no specific curve is set
                if (config.sortType == 0) config.sortType = 2; // Set to Morton
            }
        }
        else if (arg == "--seed" && i + 1 < argc) {
            config.randomSeed = std::stoul(argv[++i]);
        }
        else if (arg == "--dynamic-reordering" && i + 1 < argc) {
            std::string val = argv[++i];
            config.dynamicReordering = (val == "true" || val == "1");
        } 
        else if (arg == "--metrics-window" && i + 1 < argc) {
            config.metricsWindowSize = std::stoi(argv[++i]);
        }
        else if (arg == "-verbose") {
            config.verbose = true;
        }
        else if (arg == "-help" || arg == "--help" || arg == "-h") {
            std::cout << "N-Body Simulation Usage:\n"
                      << "  ./prog [options]\n"
                      << "Options:\n"
                      << "  -n, --bodies <particles>     : Number of particles (default: 1024)\n"
                      << "  -sort <type>                 : Space-filling curve type (0: none, 1: hilbert, 2: morton) (default: 0)\n"
                      << "  -steps, --iterations <steps> : Number of simulation steps (default: 1000)\n"
                      << "  -mdist <type>                : Mass distribution (0: uniform, 1: normal) (default: 0)\n"
                      << "  -alg, --method <algorithm>   : Algorithm (0: cpu-direct-sum, 1: cpu-barnes, 2: gpu-direct-sum, 3: gpu-barnes-hut) (default: 0)\n"
                      << "  -theta <float>               : Barnes-Hut theta parameter (default: 0.5)\n"
                      << "  -visual <0|1>                : Enable visualization (0: off, 1: on) (default: 1)\n"
                      << "  -energy <filename>           : Output energy data to file\n"
                      << "  -nt <threads>                : Number of threads for CPU algorithms (default: 1)\n"
                      << "  -bs <blocksize>              : Block size for GPU algorithms (default: 256)\n"
                      << "  --headless                   : Run without visualization or UI\n"
                      << "  --report-metrics             : Report detailed performance metrics at the end\n"
                      << "  --use-sfc <true|false>       : Enable/disable space-filling curve ordering\n"
                      << "  --dynamic-reordering <true|false> : Enable/disable dynamic reordering for SFC Barnes-Hut\n"
                      << "  --metrics-window <size>      : Window size for dynamic reordering metrics (default: 10)\n"
                      << "  --seed <number>              : Random seed for reproducible simulations\n"
                      << "  -verbose                     : Enable verbose output\n"
                      << "  -help, --help, -h            : Show this help message\n";
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
                  << "  Headless: " << (config.headless ? "Yes" : "No") << "\n"
                  << "  Report Metrics: " << (config.reportMetrics ? "Yes" : "No") << "\n"
                  << "  SFC Enabled: " << (config.useSFC ? "Yes" : "No") << "\n"
                  << "  Dynamic Reordering: " << (config.dynamicReordering ? "Yes" : "No") << "\n"
                  << "  Metrics Window Size: " << config.metricsWindowSize << "\n"
                  << "  Energy Output: " << (config.energyOutput.empty() ? "None" : config.energyOutput) << "\n"
                  << "  Threads: " << config.numThreads << "\n"
                  << "  Block Size: " << config.blockSize << "\n"
                  << "  Random Seed: " << config.randomSeed << "\n";
    }

    return config;
}

// Global simulation state for callbacks
SimulationState *g_simulationState = nullptr;

// GLFW error callback
void glfw_error_callback(int error, const char *description)
{
    // logMessage("GLFW Error: " + std::string(description), true);
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
        // logMessage("Failed to initialize GLFW", true);
        return nullptr;
    }

    // OpenGL and window hints
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Get primary monitor and video mode
    GLFWmonitor *monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode *mode = glfwGetVideoMode(monitor);

    // Create window
    GLFWwindow *window = nullptr;

    int windowWidth = 1280;
    int windowHeight = 720;

    if (config.fullscreen)
    {
        // If fullscreen, get the primary monitor resolution
        GLFWmonitor *primaryMonitor = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = glfwGetVideoMode(primaryMonitor);

        windowWidth = mode->width;
        windowHeight = mode->height;

        // For windowed fullscreen (borderless)
        glfwWindowHint(GLFW_RED_BITS, mode->redBits);
        glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
        glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
        glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

        window = glfwCreateWindow(windowWidth, windowHeight, "N-Body Simulation", primaryMonitor, nullptr);
    }
    else
    {
        window = glfwCreateWindow(windowWidth, windowHeight, "N-Body Simulation", nullptr, nullptr);
    }

    if (!window)
    {
        glfwTerminate();
        return nullptr;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Set up callbacks
    glfwSetKeyCallback(window, key_callback);

    // Enable vsync
    glfwSwapInterval(1);

    return window;
}

// Initialize GLAD
bool initializeGLAD()
{
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        return false;
    }

    return true;
}

// Setup ImGui
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
    while (!glfwWindowShouldClose(window) && simulationState.running.load())
    {
        // Poll and handle events
        glfwPollEvents();

        // Clear the screen with a dark blue background to make particles visible
        glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Get window dimensions
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        float aspectRatio = static_cast<float>(width) / static_cast<float>(height);

        // Update renderer with latest body data if available
        {
            std::lock_guard<std::mutex> lock(simulationState.mtx);
            if (simulationState.sharedBodies != nullptr && simulationState.currentBodiesCount > 0) {
                renderer.updateBodies(simulationState.sharedBodies, simulationState.currentBodiesCount);
            }
        }

        // Render bodies and octree
        renderer.render(aspectRatio);

        // Render UI
        uiManager.renderUI(window);

        // Swap front and back buffers
        glfwSwapBuffers(window); 
    }
}

// Function to collect and report metrics
void reportMetrics(SimulationData &simData, double totalSimTime, int iterations)
{
    if (!simData.valid || !simData.simulation) {
        std::cout << "No valid simulation data available for metrics reporting" << std::endl;
        return;
    }
    
    // Basic metrics
    std::cout << "total_time_ms: " << totalSimTime << std::endl;
    std::cout << "iterations: " << iterations << std::endl;
    std::cout << "avg_time_per_iteration_ms: " << (totalSimTime / iterations) << std::endl;
    
    // Simulation-specific metrics
    SimulationMetrics metrics = simData.simulation->getMetrics();
    
    // Common metrics
    std::cout << "force_time_ms: " << metrics.forceTimeMs << std::endl;
    std::cout << "total_update_time_ms: " << metrics.totalTimeMs << std::endl;
    
    // Barnes-Hut specific metrics
    if (metrics.bboxTimeMs > 0) {
        std::cout << "bbox_time_ms: " << metrics.bboxTimeMs << std::endl;
    }
    if (metrics.resetTimeMs > 0) {
        std::cout << "reset_time_ms: " << metrics.resetTimeMs << std::endl;
    }
    
    // Energy metrics
    double kineticEnergy = simData.simulation->getKineticEnergy();
    double potentialEnergy = simData.simulation->getPotentialEnergy();
    double totalEnergy = kineticEnergy + potentialEnergy;
    
    std::cout << "kinetic_energy: " << kineticEnergy << std::endl;
    std::cout << "potential_energy: " << potentialEnergy << std::endl;
    std::cout << "total_energy: " << totalEnergy << std::endl;
}

int main(int argc, char **argv)
{
    try
    {
        std::cout << "Attempting to use dedicated GPU..." << std::endl;
        checkCudaAvailability();
        SimulationConfig config = parseArgs(argc, argv);

        auto startTime = std::chrono::high_resolution_clock::now();

        SimulationState simulationState;
        
        simulationState.numBodies.store(config.initialBodies);
        simulationState.useSFC.store(config.useSFC);
        simulationState.randomSeed.store(config.randomSeed);
        
        if (config.sortType > 0) {
            simulationState.sfcCurveType.store(config.sortType == 1 ? 
                sfc::CurveType::HILBERT : sfc::CurveType::MORTON);
        }
        
        simulationState.massDistribution.store(config.massDistribution == 0 ? 
            MassDistribution::UNIFORM : MassDistribution::NORMAL);
        
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
            case 4: // CPU SFC direct sum
                simulationState.simulationMethod.store(SimulationMethod::CPU_SFC_DIRECT_SUM);
                simulationState.useSFC.store(true);
                break;
            case 5: // CPU SFC Barnes-Hut
                simulationState.simulationMethod.store(SimulationMethod::CPU_SFC_BARNES_HUT);
                simulationState.useSFC.store(true);
                break;
            case 6: // GPU SFC direct sum
                simulationState.simulationMethod.store(SimulationMethod::GPU_SFC_DIRECT_SUM);
                simulationState.useSFC.store(true);
                break;
            case 7: // GPU SFC Barnes-Hut
                simulationState.simulationMethod.store(SimulationMethod::GPU_SFC_BARNES_HUT);
                simulationState.useSFC.store(true);
                break;
        }
        
        // Set the Barnes-Hut theta parameter if applicable
        if (config.algorithm == 1 || config.algorithm == 3 || 
            config.algorithm == 5 || config.algorithm == 7) {
            g_theta = config.theta;
            g_theta = 0.5; // Default theta value
        }
        
        simulationState.openMPThreads.store(config.numThreads);

        bool isCpuMethod = (config.algorithm == 0 || config.algorithm == 1 || 
                            config.algorithm == 4 || config.algorithm == 5);
        simulationState.useOpenMP.store(isCpuMethod || config.numThreads > 1);

        bool isGpuMethod = (config.algorithm == 2 || config.algorithm == 3 || 
                           config.algorithm == 6 || config.algorithm == 7);
        if (isGpuMethod) {
            configureGPUParameters();
            config.blockSize = g_blockSize;
        } 

        simulationState.showOctree = 
            (config.algorithm == 1 || config.algorithm == 3 || 
             config.algorithm == 5 || config.algorithm == 7); // Show octree for Barnes-Hut
        simulationState.octreeMaxDepth = 3;
        simulationState.octreeOpacity = 0.5f;
        simulationState.octreeColorByMass = true;

        // Set the dynamic reordering and metrics window size options for Barnes-Hut SFC methods
        if (config.algorithm == 5 || config.algorithm == 7) {
            simulationState.dynamicReordering.store(config.dynamicReordering);
            simulationState.metricsWindowSize.store(config.metricsWindowSize);
        }

        GLFWwindow *window = nullptr;
        OpenGLRenderer *renderer = nullptr;
        SimulationUIManager *uiManager = nullptr;

        // Only initialize visualization if not in headless mode
        if (!config.headless) {
            // Initialize GLFW and create window
            window = initializeGLFW(config);
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

            // Create OpenGL renderer
            renderer = new OpenGLRenderer(simulationState);
            renderer->init();

            // Create UI manager
            uiManager = new SimulationUIManager(simulationState, *renderer);
        }

        SimulationThread simulationThread(&simulationState);
        simulationThread.start();

        g_simulationState = &simulationState;

        // Main render loop
        std::ofstream energyOutput;
        if (!config.energyOutput.empty()) {
            energyOutput.open(config.energyOutput);
            if (energyOutput.is_open()) {
                energyOutput << "Step,Time,KineticEnergy,PotentialEnergy,TotalEnergy" << std::endl;
            }
        }

        // If visualization is enabled, run the render loop
        if (config.visualization && window && renderer && uiManager) {
            renderLoop(window, simulationState, simulationThread, *renderer, *uiManager);
        }

        else {
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
                // if (step % (config.numSteps / 10) == 0 || step == config.numSteps - 1) {
                //     if (!config.headless || config.verbose) {
                //         logMessage("Simulation progress: " + std::to_string(step + 1) + "/" + 
                //                 std::to_string(config.numSteps) + " steps");
                //     }
                // }
                
                // Sleep briefly to avoid consuming 100% CPU while getting status
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
        
        // Close energy output file if opened
        if (energyOutput.is_open()) {
            energyOutput.close();
        }

        // Get final simulation data for reporting
        SimulationData finalSimData = simulationThread.getSimulationData();
        
        // Calculate total simulation time
        auto endTime = std::chrono::high_resolution_clock::now();
        double totalSimTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        
        // Report metrics if requested
        if (config.reportMetrics) {
            reportMetrics(finalSimData, totalSimTimeMs, config.numSteps);
        }

        // Cleanup
        simulationState.running.store(false);
        simulationThread.join();

        // Clean up visualization resources
        if (!config.headless) {
            // Shutdown ImGui
            ImGui_ImplOpenGL3_Shutdown();
            ImGui_ImplGlfw_Shutdown();
            ImGui::DestroyContext();

            // Cleanup renderer and UI manager
            delete uiManager;
            delete renderer;

            // Terminate GLFW
            glfwDestroyWindow(window);
            glfwTerminate();
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        // logMessage("Fatal error: " + std::string(e.what()), true);
        return 1;
    }
}