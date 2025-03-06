#include "../include/common/constants.cuh"
#include "../include/ui/simulation_ui_manager.h"

SimulationUIManager::SimulationUIManager(SimulationState &state)
    : simulationState_(state) {}

cv::Point SimulationUIManager::scaleToWindow(const Vector &pos3D, double zoomFactor, double offsetX, double offsetY)
{
    double centerX = WINDOW_WIDTH / 2.0;
    double centerY = WINDOW_HEIGHT / 2.0;

    double scaleX = WINDOW_WIDTH / (NBODY_WIDTH * 2.0) * zoomFactor;
    double scaleY = WINDOW_HEIGHT / (NBODY_HEIGHT * 2.0) * zoomFactor;

    double screenX = centerX + (pos3D.x * scaleX) - offsetX;
    double screenY = centerY - (pos3D.y * scaleY) - offsetY;

    return cv::Point((int)screenX, (int)screenY);
}

void SimulationUIManager::drawCommandMenu(cv::Mat &image)
{
    if (!simulationState_.showCommandMenu)
        return;

    // Define menu dimensions and position (same as original implementation)
    int menuWidth = 600;
    int menuHeight = 600;
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

    for (size_t i = 0; i < simulationState_.commandOptions.size(); i++)
    {
        cv::Scalar textColor = (i == simulationState_.selectedCommandIndex) ? cv::Scalar(120, 230, 255) : cv::Scalar(180, 180, 200);

        // Highlight selected option
        if (i == simulationState_.selectedCommandIndex)
        {
            cv::Rect highlightRect(menuX + 10, optionY - 20, menuWidth - 20, 30);
            cv::rectangle(overlay, highlightRect, cv::Scalar(40, 60, 100), -1);
        }

        // Display option text
        cv::putText(overlay, simulationState_.commandOptions[i], cv::Point(menuX + 25, optionY),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);

        // If it's the particles option, show current value
        if (i == 1)
        {
            std::string particleText = ": " + simulationState_.particleOptions[simulationState_.selectedParticleOption];
            cv::putText(overlay, particleText, cv::Point(menuX + 170, optionY),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 1);
        }

        // If it's the SFC option, show current state
        if (i == 2)
        {
            std::string sfcText = ": " + std::string(simulationState_.useSFC.load() ? "Activo" : "Inactivo");
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

void SimulationUIManager::drawBodies(cv::Mat &image)
{
    // Protection for accessing the shared buffer
    std::lock_guard<std::mutex> lock(simulationState_.mtx);

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
    int n = simulationState_.currentBodiesCount;
    double zoomFactor = simulationState_.zoomFactor.load();
    double offsetX = simulationState_.offsetX;
    double offsetY = simulationState_.offsetY;
    Body *bodies = simulationState_.sharedBodies;

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
    ss << "SFC: " << (simulationState_.useSFC.load() ? "Activo" : "Inactivo");
    cv::putText(image, ss.str(), cv::Point(20, 95), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    ss.str("");
    ss << "FPS: " << std::fixed << std::setprecision(1) << simulationState_.fps;
    cv::putText(image, ss.str(), cv::Point(20, 125), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 1);

    // Keyboard shortcuts panel
    cv::Rect keysPanel(10, WINDOW_HEIGHT - 40, 380, 30);
    cv::rectangle(image, keysPanel, cv::Scalar(0, 0, 0, 120), -1);
    cv::rectangle(image, keysPanel, cv::Scalar(100, 100, 100), 1);

    // More concise keyboard shortcuts
    cv::putText(image, "ESC: Salir | M: Menú | ESPACIO: Pausa | +/-: Zoom | Flechas: Mover",
                cv::Point(20, WINDOW_HEIGHT - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);

    // Draw the floating command menu if it's active
    drawCommandMenu(image);

    // Pause indicator
    if (simulationState_.isPaused.load())
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

void SimulationUIManager::setupWindow(bool fullscreen)
{
    cv::namedWindow("Barnes-Hut Simulation", cv::WINDOW_NORMAL);
    cv::resizeWindow("Barnes-Hut Simulation", WINDOW_WIDTH, WINDOW_HEIGHT);

    // Fullscreen handling
    if (fullscreen)
    {
        const char *wayland_display = getenv("WAYLAND_DISPLAY");
        if (!wayland_display)
        {
            cv::setWindowProperty("Barnes-Hut Simulation", cv::WND_PROP_FULLSCREEN, cv::WINDOW_FULLSCREEN);
        }
    }

    // Setup mouse callback
    cv::setMouseCallback("Barnes-Hut Simulation", mouseCallback, &simulationState_);
}

void SimulationUIManager::mouseCallback(int event, int x, int y, int flags, void *userdata)
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

void SimulationUIManager::executeMenuCommand(int commandIndex)
{
    switch (commandIndex)
    {
    case 0: // Restart simulation
        simulationState_.restart.store(true);
        break;

    case 1: // Change particle count
    {
        // Advance to the next option in the particle list
        simulationState_.selectedParticleOption =
            (simulationState_.selectedParticleOption + 1) % simulationState_.particleOptions.size();
        int particleCount = std::stoi(simulationState_.particleOptions[simulationState_.selectedParticleOption]);
        simulationState_.numBodies.store(particleCount);
        simulationState_.restart.store(true);
    }
    break;

    case 2: // Toggle SFC
        simulationState_.useSFC.store(!simulationState_.useSFC.load());
        break;

    case 3: // Pause/Resume
        simulationState_.isPaused.store(!simulationState_.isPaused.load());
        break;

    case 4: // Close menu
        simulationState_.showCommandMenu = false;
        break;
    }
}

void SimulationUIManager::handleKeyboardEvents(int key)
{
    if (key == 27) // ESC
    {
        simulationState_.running.store(false);
        return;
    }
    else if (key == ' ') // Space
    {
        simulationState_.isPaused.store(!simulationState_.isPaused.load());
    }
    else if (key == 'm' || key == 'M') // Show/hide command menu
    {
        simulationState_.showCommandMenu = !simulationState_.showCommandMenu;
    }
    else if (key == 13 && simulationState_.showCommandMenu) // Enter to select menu option
    {
        executeMenuCommand(simulationState_.selectedCommandIndex);
    }
    else if ((key == 82 || key == 2490368) && simulationState_.showCommandMenu) // Up arrow - menu navigation
    {
        simulationState_.selectedCommandIndex =
            (simulationState_.selectedCommandIndex > 0)
                ? simulationState_.selectedCommandIndex - 1
                : simulationState_.commandOptions.size() - 1;
    }
    else if ((key == 84 || key == 2621440) && simulationState_.showCommandMenu) // Down arrow - menu navigation
    {
        simulationState_.selectedCommandIndex =
            (simulationState_.selectedCommandIndex < simulationState_.commandOptions.size() - 1)
                ? simulationState_.selectedCommandIndex + 1
                : 0;
    }
    else if (key == 'r' || key == 'R') // Restart (keyboard shortcut)
    {
        simulationState_.restart.store(true);
    }
    else if (key == 's' || key == 'S') // Toggle SFC (keyboard shortcut)
    {
        simulationState_.useSFC.store(!simulationState_.useSFC.load());
    }
    else if (key == 'p' || key == 'P') // Add/remove bodies (keyboard shortcut)
    {
        // Advance to the next option in the particle list
        simulationState_.selectedParticleOption =
            (simulationState_.selectedParticleOption + 1) % simulationState_.particleOptions.size();
        int particleCount = std::stoi(simulationState_.particleOptions[simulationState_.selectedParticleOption]);
        simulationState_.numBodies.store(particleCount);
        simulationState_.restart.store(true);
    }
    else if (key == '+' || key == '=') // Increase zoom
    {
        // Use explicit atomic operation to multiply
        double currentZoom = simulationState_.zoomFactor.load();
        simulationState_.zoomFactor.store(currentZoom * 1.1);
    }
    else if (key == '-' || key == '_') // Decrease zoom
    {
        // Use explicit atomic operation to divide
        double currentZoom = simulationState_.zoomFactor.load();
        double newZoom = currentZoom / 1.1;
        if (newZoom < 0.1)
            newZoom = 0.1;
        simulationState_.zoomFactor.store(newZoom);
    }
    else if (key == 81 || key == 2490368) // Up arrow (when not in menu)
    {
        if (!simulationState_.showCommandMenu)
        {
            std::lock_guard<std::mutex> lock(simulationState_.mtx);
            double zoomFactor = simulationState_.zoomFactor.load();
            simulationState_.offsetY -= 50.0 / zoomFactor;
        }
    }
    else if (key == 82 || key == 2621440) // Down arrow (when not in menu)
    {
        if (!simulationState_.showCommandMenu)
        {
            std::lock_guard<std::mutex> lock(simulationState_.mtx);
            double zoomFactor = simulationState_.zoomFactor.load();
            simulationState_.offsetY += 50.0 / zoomFactor;
        }
    }
    else if (key == 83 || key == 2555904) // Right arrow
    {
        std::lock_guard<std::mutex> lock(simulationState_.mtx);
        double zoomFactor = simulationState_.zoomFactor.load();
        simulationState_.offsetX += 50.0 / zoomFactor;
    }
    else if (key == 84 || key == 2424832) // Left arrow
    {
        std::lock_guard<std::mutex> lock(simulationState_.mtx);
        double zoomFactor = simulationState_.zoomFactor.load();
        simulationState_.offsetX -= 50.0 / zoomFactor;
    }
}