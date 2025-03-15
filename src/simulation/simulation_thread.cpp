#include "../../include/simulation/simulation_thread.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

SimulationThread::SimulationThread(SimulationState *simulationState)
    : state(simulationState),
      simulation(nullptr),
      frameCounter(0),
      frameTimeAccum(0.0),
      frameCount(0)
{
    // Initialize current parameters from state
    updateCurrentParameters();
}

SimulationThread::~SimulationThread()
{
    // Make sure thread is stopped properly
    if (thread.joinable())
    {
        stop();
        join();
    }
}

void SimulationThread::start()
{
    // Start the simulation thread
    thread = std::thread(&SimulationThread::run, this);
}

void SimulationThread::stop()
{
    // Signal the thread to stop
    if (state)
    {
        state->running.store(false);
    }
}

void SimulationThread::join()
{
    // Wait for thread to finish
    if (thread.joinable())
    {
        thread.join();
    }
}

void SimulationThread::run()
{
    try
    {
        // Create initial simulation
        simulation = SimulationFactory::createFromState(*state);

        if (!simulation)
        {
            std::cerr << "Failed to create simulation" << std::endl;
            return;
        }

        // Setup initial conditions
        simulation->setup();

        // Initialize performance tracking
        lastTime = std::chrono::steady_clock::now();

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

            // Check if restart is needed
            bool shouldRestart = state->restart.load() || checkForParameterChanges();

            // Restart simulation if needed
            if (shouldRestart)
            {
                // Reset the seed change flag
                state->seedWasChanged = false;

                // Update current parameters
                updateCurrentParameters();

                // Recreate the simulation
                try
                {
                    simulation = SimulationFactory::createFromState(*state);

                    if (!simulation)
                    {
                        std::cerr << "Failed to recreate simulation" << std::endl;
                        break;
                    }

                    // Setup the simulation
                    simulation->setup();
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Exception during simulation restart: " << e.what() << std::endl;
                    break;
                }

                state->restart.store(false);

                // Reset time and frame counter
                lastTime = std::chrono::steady_clock::now();
                frameCounter = 0;
            }

            // Update simulation
            simulation->update();

            // Increment frame counter
            frameCounter++;

            // Update visualization data when needed
            if (frameCounter >= VISUALIZATION_FREQUENCY)
            {
                frameCounter = 0;
                updateVisualizationData();
            }

            // Calculate performance metrics
            auto now = std::chrono::steady_clock::now();
            double frameTime = std::chrono::duration<double, std::milli>(now - frameStart).count();

            // Update state with last iteration time
            updatePerformanceMetrics(frameTime);
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error in simulation thread: " << e.what() << std::endl;
    }
}

bool SimulationThread::checkForParameterChanges()
{
    bool shouldRestart = false;

    // Check basic parameters
    shouldRestart = shouldRestart ||
                    currentNumBodies != state->numBodies.load() ||
                    currentMethod != state->simulationMethod.load() ||
                    currentDistribution != state->bodyDistribution.load() ||
                    (state->seedWasChanged && currentSeed != state->randomSeed.load());

    // Check method-specific parameters if no restart needed yet
    if (!shouldRestart)
    {
        switch (currentMethod)
        {
        case SimulationMethod::CPU_DIRECT_SUM:
        case SimulationMethod::CPU_BARNES_HUT:
            shouldRestart = (currentUseOpenMP != state->useOpenMP.load() ||
                             currentOpenMPThreads != state->openMPThreads.load());
            break;

        case SimulationMethod::GPU_BARNES_HUT:
            shouldRestart = (currentUseSFC != state->useSFC.load());
            // Only check SFC parameters if SFC is enabled
            if (currentUseSFC && state->useSFC.load())
            {
                shouldRestart = shouldRestart ||
                                currentOrderingMode != state->sfcOrderingMode.load() ||
                                currentReorderFreq != state->reorderFrequency.load();
            }
            break;

        case SimulationMethod::GPU_SFC_BARNES_HUT:
            shouldRestart = (currentOrderingMode != state->sfcOrderingMode.load() ||
                             currentReorderFreq != state->reorderFrequency.load());
            break;

        default:
            break;
        }
    }

    return shouldRestart;
}

void SimulationThread::updateCurrentParameters()
{
    currentNumBodies = state->numBodies.load();
    currentMethod = state->simulationMethod.load();
    currentUseSFC = state->useSFC.load();
    currentOrderingMode = state->sfcOrderingMode.load();
    currentReorderFreq = state->reorderFrequency.load();
    currentDistribution = state->bodyDistribution.load();
    currentSeed = state->randomSeed.load();
    currentUseOpenMP = state->useOpenMP.load();
    currentOpenMPThreads = state->openMPThreads.load();
}

void SimulationThread::updateVisualizationData()
{
    if (!simulation)
        return;

    // Copy data from GPU to CPU
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
        std::cerr << "Exception during body data update: " << e.what() << std::endl;
    }
}

void SimulationThread::updatePerformanceMetrics(double frameTime)
{
    // Track the simulation iteration time separately from FPS
    state->lastIterationTime = frameTime;
    
    // Update FPS tracking based on visualization frequency
    if (frameCounter == 0) {
        // We've just updated the visualization, so count this as a rendered frame
        frameCount++;
    }
    
    auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - lastTime).count() >= 1.0)
    {
        // Update FPS every second - this now represents actual visualization updates
        state->fps = frameCount / std::chrono::duration<double>(now - lastTime).count();
        frameCount = 0;
        lastTime = now;
    }
}
