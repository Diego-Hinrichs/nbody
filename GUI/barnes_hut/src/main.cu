#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <getopt.h>
#include <opencv2/opencv.hpp>

#include "../include/common/types.cuh"
#include "../include/common/constants.cuh"
#include "../include/common/error_handling.cuh"
#include "../include/simulation/barnes_hut.cuh"
#include "../include/simulation/sfc_barnes_hut.cuh"

using Clock = std::chrono::steady_clock;

// Logging function
void log(const std::string &message, bool isError = false)
{
    std::ostream &stream = isError ? std::cerr : std::cout;
    stream << "[" << (isError ? "ERROR" : "INFO") << "] " << message << std::endl;
}

// Simulation configuration structure
struct SimulationConfig
{
    int initialBodies = 1000;
    bool useSFC = true;
    bool fullscreen = false;
    bool verbose = false;
};

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

// Structure for sharing state between threads
struct SimulationState
{
    std::atomic<bool> running;
    std::atomic<bool> restart;
    std::atomic<bool> useSFC;
    std::atomic<int> numBodies;
    std::atomic<double> zoomFactor;
    std::atomic<bool> isPaused;
    std::mutex mtx;
    double offsetX;
    double offsetY;

    // Shared buffer for bodies
    Body *sharedBodies;
    int currentBodiesCount;
    double fps;
    double lastIterationTime;

    // Floating menu state
    bool showCommandMenu;
    int selectedCommandIndex;
    std::vector<std::string> commandOptions;
    std::vector<std::string> particleOptions;
    int selectedParticleOption;
};

// Convert 3D position to 2D coordinates with zoom and offset
cv::Point scaleToWindow(const Vector &pos3D, double zoomFactor, double offsetX, double offsetY)
{
    double centerX = WINDOW_WIDTH / 2.0;
    double centerY = WINDOW_HEIGHT / 2.0;

    double scaleX = WINDOW_WIDTH / (NBODY_WIDTH * 2.0) * zoomFactor;
    double scaleY = WINDOW_HEIGHT / (NBODY_HEIGHT * 2.0) * zoomFactor;

    double screenX = centerX + (pos3D.x * scaleX) - offsetX;
    double screenY = centerY - (pos3D.y * scaleY) - offsetY;

    return cv::Point((int)screenX, (int)screenY);
}

// Function to draw the floating command menu
void drawCommandMenu(cv::Mat &image, SimulationState &state)
{
    if (!state.showCommandMenu)
        return;

    // Define menu dimensions and position
    int menuWidth = 300;
    int menuHeight = 300;
    int menuX = WINDOW_WIDTH - menuWidth - 20;
    int menuY = 20;

    // Create semi-transparent background overlay
    cv::Mat overlay;
    image.copyTo(overlay);
    cv::Rect menuRect(menuX, menuY, menuWidth, menuHeight);
    cv::rectangle(overlay, menuRect, cv::Scalar(20, 20, 50), -1);

    // Menu title
    cv::putText(overlay, "Comandos de Simulacion", cv::Point(menuX + 20, menuY + 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200, 200, 250), 1);

    // Divider line
    cv::line(overlay, cv::Point(menuX + 10, menuY + 45),
             cv::Point(menuX + menuWidth - 10, menuY + 45),
             cv::Scalar(100, 100, 150), 1);

    // Menu options
    int optionY = menuY + 80;
    int spacing = 35;

    for (size_t i = 0; i < state.commandOptions.size(); i++)
    {
        cv::Scalar textColor = (i == state.selectedCommandIndex) ? cv::Scalar(120, 230, 255) : cv::Scalar(180, 180, 200);

        // Highlight selected option
        if (i == state.selectedCommandIndex)
        {
            cv::Rect highlightRect(menuX + 10, optionY - 20, menuWidth - 20, 30);
            cv::rectangle(overlay, highlightRect, cv::Scalar(40, 60, 100), -1);
        }

        // Display option text
        cv::putText(overlay, state.commandOptions[i], cv::Point(menuX + 25, optionY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);

        // If it's the particles option, show current value
        if (i == 1) // Assuming particles option is the second one
        {
            std::string particleText = ": " + state.particleOptions[state.selectedParticleOption];
            cv::putText(overlay, particleText, cv::Point(menuX + 170, optionY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);
        }

        // If it's the SFC option, show current state
        if (i == 2) // Assuming SFC option is the third one
        {
            std::string sfcText = ": " + std::string(state.useSFC.load() ? "Activo" : "Inactivo");
            cv::putText(overlay, sfcText, cv::Point(menuX + 170, optionY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);
        }

        optionY += spacing;
    }

    // Menu usage instructions
    cv::putText(overlay, "↑/↓: Navegar   Enter: Seleccionar",
                cv::Point(menuX + 20, menuY + menuHeight - 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 180), 1);
    cv::putText(overlay, "M: Cerrar menu",
                cv::Point(menuX + 20, menuY + menuHeight - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(150, 150, 180), 1);

    // Apply the overlay with transparency
    double alpha = 0.9;
    cv::addWeighted(overlay, alpha, image, 1 - alpha, 0, image);

    // Menu border
    cv::rectangle(image, menuRect, cv::Scalar(100, 100, 180), 2);
}

// Draw celestial bodies on the image with information
void drawBodies(cv::Mat &image, SimulationState &state)
{
    // Protection for accessing the shared buffer
    std::lock_guard<std::mutex> lock(state.mtx);

    // Black background with star gradient
    image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

    // Add some background stars
    int numStars = 1000;
    for (int i = 0; i < numStars; i++)
    {
        int x = rand() % WINDOW_WIDTH;
        int y = rand() % WINDOW_HEIGHT;
        int intensity = 50 + rand() % 150;
        image.at<cv::Vec3b>(y, x) = cv::Vec3b(intensity, intensity, intensity);
    }

    // Get current values
    int n = state.currentBodiesCount;
    double zoomFactor = state.zoomFactor.load();
    double offsetX = state.offsetX;
    double offsetY = state.offsetY;
    Body *bodies = state.sharedBodies;

    if (!bodies || n <= 0)
        return;

    // Draw bodies
    int visibleBodies = 0;

    // Draw other bodies
    for (int i = 0; i < n; i++)
    {
        cv::Point center = scaleToWindow(bodies[i].position, zoomFactor, offsetX, offsetY);
        // Check if the planet is on screen
        if (center.x >= -10 && center.x < WINDOW_WIDTH + 10 &&
            center.y >= -10 && center.y < WINDOW_HEIGHT + 10)
        {
            int planetRadius = std::max(1, std::min(5, (int)(2 * zoomFactor)));

            // Vary color based on velocity
            double speed = sqrt(
                bodies[i].velocity.x * bodies[i].velocity.x +
                bodies[i].velocity.y * bodies[i].velocity.y +
                bodies[i].velocity.z * bodies[i].velocity.z);

            int b = 150 + std::min(105, (int)(speed * 2000));
            int g = 150 + std::min(105, (int)(speed * 1000));
            int r = 150;

            cv::circle(image, center, planetRadius, cv::Scalar(b, g, r), -1);
            visibleBodies++;
        }
    }

    // Information panel
    cv::Rect panel(10, 10, 300, 130);
    cv::rectangle(image, panel, cv::Scalar(0, 0, 0), -1);
    cv::rectangle(image, panel, cv::Scalar(100, 100, 100), 1);

    // Display information on screen
    std::stringstream ss;
    ss << "Cuerpos: " << n << " (Visibles: " << visibleBodies << ")";
    cv::putText(image, ss.str(), cv::Point(20, 35), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    ss.str("");
    ss << "Zoom: " << std::fixed << std::setprecision(2) << zoomFactor << "x";
    cv::putText(image, ss.str(), cv::Point(20, 65), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    ss.str("");
    ss << "SFC: " << (state.useSFC.load() ? "Activo" : "Inactivo");
    cv::putText(image, ss.str(), cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    ss.str("");
    ss << "FPS: " << std::fixed << std::setprecision(1) << state.fps;
    cv::putText(image, ss.str(), cv::Point(20, 125), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    // Keyboard shortcuts panel (more compact)
    cv::Rect keysPanel(10, WINDOW_HEIGHT - 40, 380, 30);
    cv::rectangle(image, keysPanel, cv::Scalar(0, 0, 0, 120), -1);
    cv::rectangle(image, keysPanel, cv::Scalar(100, 100, 100), 1);

    // More concise keyboard shortcuts
    cv::putText(image, "ESC: Salir | M: Menú | ESPACIO: Pausa | +/-: Zoom | Flechas: Mover",
                cv::Point(20, WINDOW_HEIGHT - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    // Draw the floating command menu if it's active
    drawCommandMenu(image, state);

    // Pause indicator
    if (state.isPaused.load())
    {
        cv::Rect pauseBox(WINDOW_WIDTH / 2 - 100, WINDOW_HEIGHT / 2 - 40, 200, 80);
        cv::rectangle(image, pauseBox, cv::Scalar(0, 0, 50), -1);
        cv::rectangle(image, pauseBox, cv::Scalar(0, 0, 200), 2);
        cv::putText(image, "SIMULACION", cv::Point(WINDOW_WIDTH / 2 - 85, WINDOW_HEIGHT / 2 - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
        cv::putText(image, "PAUSADA", cv::Point(WINDOW_WIDTH / 2 - 65, WINDOW_HEIGHT / 2 + 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);
    }
}

// Execute the selected action in the command menu
void executeMenuCommand(SimulationState *state, int commandIndex)
{
    switch (commandIndex)
    {
    case 0: // Restart simulation
        state->restart.store(true);
        break;

    case 1: // Change particle count
    {
        // Advance to the next option in the particle list
        state->selectedParticleOption = (state->selectedParticleOption + 1) % state->particleOptions.size();
        int particleCount = std::stoi(state->particleOptions[state->selectedParticleOption]);
        state->numBodies.store(particleCount);
        state->restart.store(true);
    }
    break;

    case 2: // Toggle SFC
        state->useSFC.store(!state->useSFC.load());
        break;

    case 3: // Pause/Resume
        state->isPaused.store(!state->isPaused.load());
        break;

    case 4: // Close menu
        state->showCommandMenu = false;
        break;
    }
}

// Handle mouse events
void mouseCallback(int event, int x, int y, int flags, void *userdata)
{
    SimulationState *state = static_cast<SimulationState *>(userdata);

    static bool dragging = false;
    static int lastX = 0, lastY = 0;

    if (event == cv::EVENT_LBUTTONDOWN)
    {
        dragging = true;
        lastX = x;
        lastY = y;
    }
    else if (event == cv::EVENT_LBUTTONUP)
    {
        dragging = false;
    }
    else if (event == cv::EVENT_MOUSEMOVE && dragging)
    {
        // Move view according to mouse drag
        std::lock_guard<std::mutex> lock(state->mtx);
        state->offsetX += (x - lastX);
        state->offsetY += (y - lastY);
        lastX = x;
        lastY = y;
    }
    else if (event == cv::EVENT_MOUSEWHEEL)
    {
        // Zoom with mouse wheel
        int delta = cv::getMouseWheelDelta(flags);
        if (delta > 0)
        {
            // Use explicit atomic operation to multiply
            double currentZoom = state->zoomFactor.load();
            state->zoomFactor.store(currentZoom * 1.1);
        }
        else
        {
            // Use explicit atomic operation to divide
            double currentZoom = state->zoomFactor.load();
            double newZoom = currentZoom / 1.1;
            if (newZoom < 0.1)
                newZoom = 0.1;
            state->zoomFactor.store(newZoom);
        }
    }
}

// Function for simulation thread
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
    auto lastTime = Clock::now();
    double frameTimeAccum = 0.0;
    int frameCount = 0;

    // Main simulation loop
    while (state->running.load())
    {
        auto frameStart = Clock::now();

        // If paused, wait and continue
        if (state->isPaused.load())
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            lastTime = Clock::now(); // Reset timer when resumed
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
            lastTime = Clock::now();
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
        auto now = Clock::now();
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

        // Configure simulation state
        SimulationState state;
        state.running.store(true);
        state.restart.store(false);
        state.numBodies.store(config.initialBodies);
        state.useSFC.store(config.useSFC);
        state.isPaused.store(false);
        state.zoomFactor.store(1.0);
        state.offsetX = 0.0;
        state.offsetY = 0.0;
        state.sharedBodies = nullptr;
        state.currentBodiesCount = 0;
        state.fps = 0.0;
        state.lastIterationTime = 0.0;

        // Command menu configuration (similar to previous implementation)
        state.showCommandMenu = false;
        state.selectedCommandIndex = 0;
        state.commandOptions = {
            "Reiniciar Simulacion",
            "Numero de Particulas",
            "Activar/Desactivar SFC",
            "Pausar/Reanudar",
            "Cerrar Menu"};
        state.particleOptions = {"1000", "5000", "10000", "15000", "20000"};

        // Find the index corresponding to the initial particle count
        state.selectedParticleOption = 0;
        for (size_t i = 0; i < state.particleOptions.size(); i++)
        {
            if (std::stoi(state.particleOptions[i]) == config.initialBodies)
            {
                state.selectedParticleOption = i;
                break;
            }
        }

        // Create main window
        cv::Mat display;
        cv::namedWindow("Barnes-Hut Simulation", cv::WINDOW_NORMAL);
        cv::resizeWindow("Barnes-Hut Simulation", WINDOW_WIDTH, WINDOW_HEIGHT);

        // Fullscreen handling
        if (config.fullscreen)
        {
            const char *wayland_display = getenv("WAYLAND_DISPLAY");
            if (!wayland_display)
            {
                cv::setWindowProperty("Barnes-Hut Simulation", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
            }
        }

        // Setup mouse and other UI elements (same as previous implementation)
        cv::setMouseCallback("Barnes-Hut Simulation", mouseCallback, &state);

        // Start simulation thread
        std::thread simThread(simulationThread, &state);
        // Main UI loop
        auto lastUIUpdate = Clock::now();
        while (state.running.load())
        {
            // Create image for visualization
            display = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);

            // Draw bodies and show information
            drawBodies(display, state);

            // Show the image
            cv::imshow("Barnes-Hut Simulation", display);

            // Process keyboard events (with short timeout for quick response)
            int key = cv::waitKey(15);
            if (key == 27) // ESC
            {
                state.running.store(false);
                break;
            }
            else if (key == ' ') // Space
            {
                state.isPaused.store(!state.isPaused.load());
            }
            else if (key == 'm' || key == 'M') // Show/hide command menu
            {
                state.showCommandMenu = !state.showCommandMenu;
            }
            else if (key == 13 && state.showCommandMenu) // Enter to select menu option
            {
                executeMenuCommand(&state, state.selectedCommandIndex);
            }
            else if ((key == 82 || key == 2490368) && state.showCommandMenu) // Up arrow - menu navigation
            {
                state.selectedCommandIndex = (state.selectedCommandIndex > 0) ? state.selectedCommandIndex - 1 : state.commandOptions.size() - 1;
            }
            else if ((key == 84 || key == 2621440) && state.showCommandMenu) // Down arrow - menu navigation
            {
                state.selectedCommandIndex = (state.selectedCommandIndex < state.commandOptions.size() - 1) ? state.selectedCommandIndex + 1 : 0;
            }
            else if (key == 'r' || key == 'R') // Restart (keyboard shortcut)
            {
                state.restart.store(true);
            }
            else if (key == 's' || key == 'S') // Toggle SFC (keyboard shortcut)
            {
                state.useSFC.store(!state.useSFC.load());
            }
            else if (key == 'p' || key == 'P') // Add/remove bodies (keyboard shortcut)
            {
                // Advance to the next option in the particle list
                state.selectedParticleOption = (state.selectedParticleOption + 1) % state.particleOptions.size();
                int particleCount = std::stoi(state.particleOptions[state.selectedParticleOption]);
                state.numBodies.store(particleCount);
                state.restart.store(true);
            }
            else if (key == '+' || key == '=') // Increase zoom
            {
                // Use explicit atomic operation to multiply
                double currentZoom = state.zoomFactor.load();
                state.zoomFactor.store(currentZoom * 1.1);
            }
            else if (key == '-' || key == '_') // Decrease zoom
            {
                // Use explicit atomic operation to divide
                double currentZoom = state.zoomFactor.load();
                double newZoom = currentZoom / 1.1;
                if (newZoom < 0.1)
                    newZoom = 0.1;
                state.zoomFactor.store(newZoom);
            }
            else if (key == 81 || key == 2490368) // Up arrow (when not in menu)
            {
                if (!state.showCommandMenu)
                {
                    std::lock_guard<std::mutex> lock(state.mtx);
                    double zoomFactor = state.zoomFactor.load();
                    state.offsetY -= 50.0 / zoomFactor;
                }
            }
            else if (key == 82 || key == 2621440) // Down arrow (when not in menu)
            {
                if (!state.showCommandMenu)
                {
                    std::lock_guard<std::mutex> lock(state.mtx);
                    double zoomFactor = state.zoomFactor.load();
                    state.offsetY += 50.0 / zoomFactor;
                }
            }
            else if (key == 83 || key == 2555904) // Right arrow
            {
                std::lock_guard<std::mutex> lock(state.mtx);
                double zoomFactor = state.zoomFactor.load();
                state.offsetX += 50.0 / zoomFactor;
            }
            else if (key == 84 || key == 2424832) // Left arrow
            {
                std::lock_guard<std::mutex> lock(state.mtx);
                double zoomFactor = state.zoomFactor.load();
                state.offsetX -= 50.0 / zoomFactor;
            }

            // Limit UI update rate
            auto now = Clock::now();
            auto elapsed = std::chrono::duration<double, std::milli>(now - lastUIUpdate).count();
            if (elapsed < 16.67) // Approximately 60 FPS
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(16.67 - elapsed)));
            }
            lastUIUpdate = Clock::now();
        }

        // Wait for simulation thread to finish
        simThread.join();

        // Cleanup
        if (state.sharedBodies != nullptr)
        {
            delete[] state.sharedBodies;
        }
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