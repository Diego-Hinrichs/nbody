#include <iostream>
#include <thread>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <getopt.h>

#include "../include/common/constants.cuh"
#include "../include/simulation/barnes_hut.cuh"
#include "../include/simulation/sfc_barnes_hut.cuh"

#include "../include/ui/simulation_state.h"
#include "../include/ui/simulation_ui_manager.h"

// Configuration structure
struct SimulationConfig
{
    int initialBodies = 1000;
    bool useSFC = true;
    bool fullscreen = false;
    bool verbose = false;
};

// Logging function
void log(const std::string &message, bool isError = false)
{
    std::ostream &stream = isError ? std::cerr : std::cout;
    stream << "[" << (isError ? "ERROR" : "INFO") << "] " << message << std::endl;
}

// Parse command-line arguments
SimulationConfig parseArgs(int argc, char **argv)
{
    SimulationConfig config;
    int opt;

    struct option long_options[] = {
        {"bodies", required_argument, 0, 'n'},
        {"sfc", no_argument, 0, 's'},
        {"no-sfc", no_argument, 0, 'S'},
        {"fullscreen", no_argument, 0, 'f'},
        {"verbose", no_argument, 0, 'v'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}};

    while ((opt = getopt_long(argc, argv, "n:sfSvh", long_options, NULL)) != -1)
    {
        switch (opt)
        {
        case 'n':
            config.initialBodies = std::max(100, std::min(50000, atoi(optarg)));
            break;
        case 's':
            config.useSFC = true;
            break;
        case 'S':
            config.useSFC = false;
            break;
        case 'f':
            config.fullscreen = true;
            break;
        case 'v':
            config.verbose = true;
            break;
        case 'h':
            std::cout << "Barnes-Hut N-body Simulation\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  -n, --bodies N     Number of bodies (100-50000)\n"
                      << "  -s, --sfc          Enable Space-Filling Curve (default)\n"
                      << "  -S, --no-sfc       Disable Space-Filling Curve\n"
                      << "  -f, --fullscreen   Start in fullscreen mode\n"
                      << "  -v, --verbose      Enable verbose logging\n"
                      << "  -h, --help         Show this help message\n";
            exit(0);
        }
    }

    return config;
}

// Simulation thread function
void simulationThread(SimulationState *state)
{
    // Use base class pointers to support both simulation types
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
            lastTime = std::chrono::steady_clock::now(); // Reset timer when resumed
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

        // Update shared buffer with new body data
        {
            std::lock_guard<std::mutex> lock(state->mtx);
            Body *bodies = simulation->getBodies();
            int n = simulation->getNumBodies();

            // If buffer size has changed, recreate it
            if (n != state->currentBodiesCount && state->sharedBodies != nullptr)
            {
                delete[] state->sharedBodies;
                state->sharedBodies = nullptr;
            }

            // Create buffer if needed
            if (state->sharedBodies == nullptr)
            {
                state->sharedBodies = new Body[n];
            }

            // Copy data
            memcpy(state->sharedBodies, bodies, n * sizeof(Body));
            state->currentBodiesCount = n;
        }

        // Calculate FPS
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
        // Parse command-line arguments
        SimulationConfig config = parseArgs(argc, argv);

        // Logging configuration
        if (config.verbose)
        {
            log("Initializing Barnes-Hut N-body Simulation");
            log("Number of bodies: " + std::to_string(config.initialBodies));
            log("Space-Filling Curve: " + std::string(config.useSFC ? "Enabled" : "Disabled"));
        }

        // Create simulation state
        SimulationState simulationState;
        simulationState.numBodies.store(config.initialBodies);
        simulationState.useSFC.store(config.useSFC);

        // Create UI manager
        SimulationUIManager uiManager(simulationState);

        // Setup window
        uiManager.setupWindow(config.fullscreen);

        // Create display image
        cv::Mat display;

        // Start simulation thread
        std::thread simThread(simulationThread, &simulationState);

        // Main UI loop
        auto lastUIUpdate = std::chrono::steady_clock::now();
        while (simulationState.running.load())
        {
            // Create image for visualization
            display = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

            // Draw bodies and show information
            uiManager.drawBodies(display);

            // Show the image
            cv::imshow("Barnes-Hut Simulation", display);

            // Process keyboard events (with short timeout for quick response)
            int key = cv::waitKey(15);
            uiManager.handleKeyboardEvents(key);

            // Limit UI update rate
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration<double, std::milli>(now - lastUIUpdate).count();
            if (elapsed < 16.67) // Approximately 60 FPS
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(16.67 - elapsed)));
            }
            lastUIUpdate = std::chrono::steady_clock::now();
        }

        // Wait for simulation thread to finish
        simThread.join();

        // Cleanup done in SimulationState destructor
        cv::destroyAllWindows();

        if (config.verbose)
        {
            log("Simulation completed successfully");
        }

        return 0;
    }
    catch (const std::exception &e)
    {
        log("Fatal error: " + std::string(e.what()), true);
        return 1;
    }
}