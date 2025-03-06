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

// Logging function
void logMessage(const std::string &message, bool isError = false)
{
    std::ostream &stream = isError ? std::cerr : std::cout;
    stream << "[" << (isError ? "ERROR" : "INFO") << "] " << message << std::endl;
}

// Configuration structure
struct SimulationConfig
{
    int initialBodies = 1000;
    bool useSFC = true;
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

// Simulation thread function
void simulationThread(SimulationState *state)
{
    SimulationBase *simulation = nullptr;
    int currentNumBodies = state->numBodies.load();
    bool currentUseSFC = state->useSFC.load();

    // Create initial simulation
    if (currentUseSFC)
    {
        simulation = new SFCBarnesHut(currentNumBodies, true);
    }
    else
    {
        simulation = new BarnesHut(currentNumBodies);
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
            currentUseSFC != state->useSFC.load())
        {
            // Update parameters
            currentNumBodies = state->numBodies.load();
            currentUseSFC = state->useSFC.load();

            // Recreate the simulation
            delete simulation;

            if (currentUseSFC)
            {
                simulation = new SFCBarnesHut(currentNumBodies, true);
            }
            else
            {
                simulation = new BarnesHut(currentNumBodies);
            }

            simulation->setup();
            state->restart.store(false);

            // Reset time
            lastTime = std::chrono::steady_clock::now();
        }

        // Update simulation
        simulation->update();

        // Read bodies from device
        simulation->copyBodiesFromDevice();

        // Calculate performance metrics
        auto now = std::chrono::steady_clock::now();
        double frameTime = std::chrono::duration<double, std::milli>(now - frameStart).count();
        state->lastIterationTime = frameTime;

        frameTimeAccum += frameTime;
        frameCount++;

        if (std::chrono::duration<double>(now - lastTime).count() >= 1.0)
        {
            // Update FPS every second
            state->fps = 1000.0 / (frameTimeAccum / frameCount);
            frameTimeAccum = 0.0;
            frameCount = 0;
            lastTime = now;
        }

        // Limit speed to avoid overloading the GPU
        if (frameTime < 16.67) // Approximately 60 FPS
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(16.67 - frameTime)));
        }
    }

    // Cleanup
    delete simulation;
}

int main(int argc, char **argv)
{
    try
    {
        // Initialize GLFW
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
        {
            logMessage("Failed to initialize GLFW", true);
            return -1;
        }

        // OpenGL and window hints
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        // Get primary monitor and video mode
        GLFWmonitor *monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode *mode = glfwGetVideoMode(monitor);

        // Create fullscreen window
        GLFWwindow *window = glfwCreateWindow(
            mode->width,
            mode->height,
            "N-Body Simulation",
            monitor, // Fullscreen
            nullptr);

        if (!window)
        {
            logMessage("Failed to create GLFW window", true);
            glfwTerminate();
            return -1;
        }

        // Make the window's context current
        glfwMakeContextCurrent(window);
        glfwSwapInterval(1); // Enable vsync

        // Load OpenGL functions with GLAD
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
        {
            logMessage("Failed to initialize GLAD", true);
            return -1;
        }

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

        // Parse command-line arguments
        SimulationConfig config = parseArgs(argc, argv);

        // Create simulation state
        SimulationState simulationState;
        simulationState.numBodies.store(config.initialBodies);
        simulationState.useSFC.store(config.useSFC);

        // Set global simulation state for callbacks
        g_simulationState = &simulationState;

        // Start simulation thread
        std::thread simThread(simulationThread, &simulationState);

        // Main render loop
        while (!glfwWindowShouldClose(window) && simulationState.running.load())
        {
            // Poll and handle events
            glfwPollEvents();

            // Start ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            // Render simulation UI
            ImGui::Begin("Simulation Controls");

            // Basic simulation info
            ImGui::Text("Bodies: %d", simulationState.numBodies.load());
            ImGui::Text("FPS: %.1f", simulationState.fps);

            // Simulation controls
            if (ImGui::Button("Pause/Resume"))
            {
                simulationState.isPaused.store(!simulationState.isPaused.load());
            }

            ImGui::SameLine();
            if (ImGui::Button("Restart"))
            {
                simulationState.restart.store(true);
            }

            // SFC Toggle
            bool sfcEnabled = simulationState.useSFC.load();
            if (ImGui::Checkbox("Space-Filling Curve", &sfcEnabled))
            {
                simulationState.useSFC.store(sfcEnabled);
                simulationState.restart.store(true);
            }

            ImGui::End();

            // Render ImGui
            ImGui::Render();

            // Clear screen
            glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);

            // Render ImGui draw data
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

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