#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_opengl3.h>
#include <glm/glm.hpp>

#include "../include/common/constants.cuh"
#include "../include/simulation/barnes_hut.cuh"
#include "../include/simulation/sfc_barnes_hut.cuh"
#include "../include/ui/simulation_state.h"
#include "../include/ui/opengl_renderer.h"
#include "../include/ui/simulation_ui_manager.h"

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
    bool useSFC = false;
    bool fullscreen = true;
    bool verbose = false;
};

// Parse command-line arguments
SimulationConfig parseArgs(int argc, char **argv)
{
    SimulationConfig config;
    // Add argument parsing logic if needed
    return config;
}

// Global simulation state and renderer
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

// Simulation thread function
void simulationThread(SimulationState *state)
{
    try
    {
        SimulationBase *simulation = nullptr;
        int currentNumBodies = state->numBodies.load();
        bool currentUseSFC = state->useSFC.load();
        SFCOrderingMode currentOrderingMode = state->sfcOrderingMode.load();
        int currentReorderFreq = state->reorderFrequency.load();
        BodyDistribution currentDistribution = state->bodyDistribution.load();
        unsigned int currentSeed = state->randomSeed.load();

        // Configuración para reducir transferencias CPU-GPU
        const int VISUALIZATION_FREQUENCY = 24; // Actualizar visualización cada 24 cuadros
        int frameCounter = 0;

        // Create initial simulation with distribution parameters
        if (currentUseSFC)
        {
            simulation = new SFCBarnesHut(
                currentNumBodies,
                true,                // Use SFC
                currentOrderingMode, // Ordering mode
                currentReorderFreq,  // Reorder frequency
                currentDistribution, // Distribution type
                currentSeed          // Random seed
            );
            // Set distribution parameters
            simulation->setDistributionAndSeed(currentDistribution, currentSeed);
        }
        else
        {
            simulation = new BarnesHut(
                currentNumBodies,
                currentDistribution, // Distribution type
                currentSeed          // Random seed
            );
            // Set distribution parameters
            simulation->setDistributionAndSeed(currentDistribution, currentSeed);
        }

        if (!simulation)
        {
            logMessage("Failed to create simulation", true);
            return;
        }

        // Setup initial conditions
        simulation->setup();

        // Variables for performance measurement
        auto lastTime = std::chrono::steady_clock::now();
        double frameTimeAccum = 0.0;
        int frameCount = 0;

        // Main simulation loop
        while (state->running.load())
        {
            auto frameStart = std::chrono::steady_clock::now();

            // If paused, wait and continue
            if (state->isPaused.load())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                lastTime = std::chrono::steady_clock::now();
                continue;
            }

            // Check if simulation restart is needed
            if (state->restart.load() ||
                currentNumBodies != state->numBodies.load() ||
                currentUseSFC != state->useSFC.load() ||
                currentOrderingMode != state->sfcOrderingMode.load() ||
                currentReorderFreq != state->reorderFrequency.load() ||
                currentDistribution != state->bodyDistribution.load() ||
                (state->seedWasChanged && currentSeed != state->randomSeed.load()))
            {
                // Update parameters
                currentNumBodies = state->numBodies.load();
                currentUseSFC = state->useSFC.load();
                currentOrderingMode = state->sfcOrderingMode.load();
                currentReorderFreq = state->reorderFrequency.load();
                currentDistribution = state->bodyDistribution.load();
                currentSeed = state->randomSeed.load();

                // Reset the seed change flag
                state->seedWasChanged = false;

                // Recreate the simulation
                delete simulation;
                simulation = nullptr; // Avoid dangling pointer

                try
                {
                    if (currentUseSFC)
                    {
                        simulation = new SFCBarnesHut(
                            currentNumBodies,
                            true,                // Use SFC
                            currentOrderingMode, // Ordering mode
                            currentReorderFreq,  // Reorder frequency
                            currentDistribution, // Distribution type
                            currentSeed          // Random seed
                        );
                    }
                    else
                    {
                        simulation = new BarnesHut(
                            currentNumBodies,
                            currentDistribution, // Distribution type
                            currentSeed          // Random seed
                        );
                    }

                    if (!simulation)
                    {
                        logMessage("Failed to recreate simulation", true);
                        break;
                    }

                    // Set distribution parameters
                    simulation->setDistributionAndSeed(currentDistribution, currentSeed);
                    simulation->setup();
                }
                catch (const std::exception &e)
                {
                    logMessage("Exception during simulation restart: " + std::string(e.what()), true);
                    break;
                }

                state->restart.store(false);

                // Reset time and frame counter
                lastTime = std::chrono::steady_clock::now();
                frameCounter = 0;
            }

            // Update simulation
            simulation->update();

            // Incrementar contador de cuadros
            frameCounter++;

            // Leer cuerpos desde el dispositivo solo cuando sea necesario
            if (frameCounter >= VISUALIZATION_FREQUENCY)
            {
                frameCounter = 0;

                // Copiar datos desde GPU a CPU
                simulation->copyBodiesFromDevice();

                // Update shared bodies for rendering
                try
                {
                    std::lock_guard<std::mutex> lock(state->mtx);

                    // Cleanup old data
                    delete[] state->sharedBodies;
                    state->sharedBodies = nullptr;

                    // Allocate and copy new data
                    state->sharedBodies = new Body[currentNumBodies];
                    memcpy(state->sharedBodies, simulation->getBodies(), currentNumBodies * sizeof(Body));
                    state->currentBodiesCount = currentNumBodies;
                }
                catch (const std::exception &e)
                {
                    logMessage("Exception during body data update: " + std::string(e.what()), true);
                    break;
                }
            }

            // Calculate performance metrics
            auto now = std::chrono::steady_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(now - frameStart).count();
            state->lastIterationTime = frameTime;

            frameTimeAccum += frameTime;
            frameCount++;

            if (std::chrono::duration<double>(now - lastTime).count() >= 1.0)
            {
                // Update FPS every second
                state->fps = frameCount / std::chrono::duration<double>(now - lastTime).count();
                frameTimeAccum = 0.0;
                frameCount = 0;
                lastTime = now;
            }
        }

        // Cleanup
        delete simulation;
    }
    catch (const std::exception &e)
    {
        logMessage("Fatal error in simulation thread: " + std::string(e.what()), true);
    }
}

int main(int argc, char **argv)
{
    try
    {
        std::cout << "Attempting to use dedicated GPU..." << std::endl;

        // Parse command-line arguments
        SimulationConfig config = parseArgs(argc, argv);

        // Initialize GLFW
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
        {
            logMessage("Failed to initialize GLFW", true);
            return -1;
        }

        // OpenGL and window hints
        // En el método de inicialización de GLFW
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
        glfwWindowHint(GLFW_CONTEXT_CREATION_API, GLFW_NATIVE_CONTEXT_API);

        // Get primary monitor and video mode
        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);

        // Create window (windowed mode for easier debugging)
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
            return -1;
        }

        // Make the window's context current
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync

        glfwSetKeyCallback(window, key_callback);

        // Load OpenGL functions with GLAD
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            logMessage("Failed to initialize GLAD", true);
            return -1;
        }
        // Extensive OpenGL context and capability checking
        std::cout << "OpenGL Initialization Details:" << std::endl;
        logMessage("OpenGL Version: " + std::string((char *)glGetString(GL_VERSION)));
        logMessage("GLSL Version: " + std::string((char *)glGetString(GL_SHADING_LANGUAGE_VERSION)));
        logMessage("Renderer: " + std::string((char *)glGetString(GL_RENDERER)));

        // OpenGL configuration
        glEnable(GL_DEPTH_TEST);
        glEnable(GL_PROGRAM_POINT_SIZE);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        (void)io;

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 330");

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Create simulation state
        SimulationState simulationState;
        simulationState.numBodies.store(config.initialBodies);
        simulationState.useSFC.store(config.useSFC);

        // Create OpenGL renderer
        OpenGLRenderer renderer(simulationState);
        renderer.init();

        // Create UI manager
        SimulationUIManager uiManager(simulationState, renderer);

        // Set global simulation state for callbacks
        g_simulationState = &simulationState;

        // Start simulation thread
        std::thread simThread(simulationThread, &simulationState);

        // Main render loop
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

            // Render bodies
            renderer.render(aspectRatio);

            // Render UI
            uiManager.renderUI(window);

            // Swap front and back buffers
            glfwSwapBuffers(window);
        }

        // Cleanup
        simulationState.running.store(false);
        simThread.join();

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