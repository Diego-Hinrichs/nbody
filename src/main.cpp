#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <functional>
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
    bool fullscreen = true;
    bool useSFC = false;
    bool verbose = false;
};

// Parse command-line arguments
SimulationConfig parseArgs(int argc, char **argv)
{
    SimulationConfig config;

    if (argc > 1)
        config.initialBodies = std::stoi(argv[1]);
    if (argc > 2)
        config.fullscreen = std::stoi(argv[2]);

    return config;
}

// Global simulation state for callbacks
SimulationState *g_simulationState = nullptr;

// Forward declaration of octree visualization functions
void updateOctreeVisualization(SimulationThread *simThread, OpenGLRenderer &renderer);

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

        // Update octree visualization if enabled
        frameCounter++;
        if (simulationState.showOctree && frameCounter >= OCTREE_UPDATE_FREQ)
        {
            updateOctreeVisualization(&simThread, renderer);
            frameCounter = 0;
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

// Helper function to update octree visualization
void updateOctreeVisualization(SimulationThread *simThread, OpenGLRenderer &renderer)
{
    std::cout << "Intentando actualizar visualización de octree..." << std::endl;
    
    if (!simThread || !g_simulationState) {
        std::cout << "Error: simThread o g_simulationState es nullptr" << std::endl;
        return;
    }

    // Check if we are using Barnes-Hut method
    SimulationMethod method = g_simulationState->simulationMethod.load();
    bool isBarnesHut = (method == SimulationMethod::CPU_BARNES_HUT ||
                        method == SimulationMethod::CPU_SFC_BARNES_HUT ||
                        method == SimulationMethod::GPU_BARNES_HUT);

    if (!isBarnesHut)
    {
        std::cout << "Método actual no es Barnes-Hut: " << static_cast<int>(method) << std::endl;
        return;
    }

    // Obtener datos de simulación
    auto simData = simThread->getSimulationData();
    if (!simData.valid)
    {
        std::cout << "Error: simData no es válido" << std::endl;
        return;
    }

    // Get octree data based on simulation type
    std::vector<Node> octreeNodes;
    int numNodes = 0;
    int rootIndex = 0;

    if (method == SimulationMethod::GPU_BARNES_HUT)
    {
        BarnesHut *bhSim = dynamic_cast<BarnesHut *>(simData.simulation);
        if (bhSim)
        {
            numNodes = bhSim->getNumNodes();
            std::cout << "Número de nodos en el octree: " << numNodes << std::endl;

            octreeNodes.resize(numNodes);
            if (bhSim->getOctreeNodes(octreeNodes.data(), numNodes))
            {
                std::cout << "Nodos de octree obtenidos correctamente" << std::endl;
                rootIndex = bhSim->getRootNodeIndex();
            }
            else
            {
                std::cout << "Error al obtener nodos de octree" << std::endl;
                return;
            }
        }
        else
        {
            std::cout << "Error: dynamic_cast a BarnesHut falló" << std::endl;
            return;
        }
    }
    else if (method == SimulationMethod::CPU_BARNES_HUT ||
             method == SimulationMethod::CPU_SFC_BARNES_HUT)
    {
        // Handle CPU Barnes-Hut
        CPUBarnesHut *cpuBhSim = dynamic_cast<CPUBarnesHut *>(simData.simulation);
        if (cpuBhSim)
        {
            // Para CPU Barnes-Hut necesitaríamos añadir métodos similares
            // (getNumNodes, getOctreeNodes, etc.)
            // Si no están implementados, esta parte no se ejecutará
            return;
        }
    }

    // Update visualization if we have valid data
    if (numNodes > 0 && !octreeNodes.empty()) {
        std::cout << "Actualizando visualización con " << numNodes << " nodos" << std::endl;
        renderer.updateOctreeVisualization(
            octreeNodes.data(),
            numNodes,
            rootIndex,
            g_simulationState->octreeMaxDepth
        );
    } else {
        std::cout << "No hay nodos válidos para renderizar" << std::endl;
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

        // Create simulation state
        SimulationState simulationState;
        simulationState.numBodies.store(config.initialBodies);
        simulationState.useSFC.store(config.useSFC);

        // Inicializar variables para visualización de octree
        simulationState.showOctree = false;
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
        renderLoop(window, simulationState, simulationThread, renderer, uiManager);

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