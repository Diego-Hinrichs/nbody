#include "../../include/simulation/simulation_thread.hpp"
#include <iostream>
#include <cstring>
#include <stdexcept>

SimulationThread::SimulationThread(SimulationState *simulationState)
    : state(simulationState),
      simulation(nullptr),
      frameTimeAccum(0.0)
{
    if (!state) {
        throw std::runtime_error("Null simulation state provided");
    }

    std::cout << "Initializing simulation thread with parameters:" << std::endl;
    std::cout << "  Bodies: " << state->numBodies.load() << std::endl;
    std::cout << "  Method: " << static_cast<int>(state->simulationMethod.load()) << std::endl;
    std::cout << "  SFC Enabled: " << (state->useSFC.load() ? "Yes" : "No") << std::endl;
    std::cout << "  OpenMP: " << (state->useOpenMP.load() ? "Yes" : "No") << std::endl;
    if (state->useOpenMP.load()) {
        std::cout << "  Threads: " << state->openMPThreads.load() << std::endl;
    }

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
        // Initialize current parameters
        updateCurrentParameters();

        // Initialize simulation immediately
        try
        {
            std::cout << "Creating initial simulation..." << std::endl;
            {
                std::lock_guard<std::mutex> lock(simulationMutex);
                simulation = SimulationFactory::createFromState(*state);

                if (!simulation)
                {
                    std::cerr << "Failed to create initial simulation" << std::endl;
                    return;
                }

                // Setup the simulation
                simulation->setup();
                std::cout << "Initial simulation setup completed" << std::endl;
            }
        }
        catch (const std::exception &e)
        {
            std::cerr << "Exception during initial simulation setup: " << e.what() << std::endl;
            return;
        }

        // Main simulation loop
        while (state->running.load())
        {
            auto frameStart = std::chrono::steady_clock::now();

            // Check if we need to restart the simulation
            bool shouldRestart = checkForParameterChanges();

            if (shouldRestart)
            {
                std::cout << "Restarting simulation due to parameter changes..." << std::endl;
                // Reset the seed change flag
                state->seedWasChanged = false;

                // Update current parameters
                updateCurrentParameters();

                // Recreate the simulation
                try
                {
                    {
                        std::lock_guard<std::mutex> lock(simulationMutex);
                        simulation = SimulationFactory::createFromState(*state);

                        if (!simulation)
                        {
                            std::cerr << "Failed to recreate simulation" << std::endl;
                            break;
                        }

                        // Setup the simulation
                        simulation->setup();
                        std::cout << "Simulation restarted successfully" << std::endl;
                    }
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Exception during simulation restart: " << e.what() << std::endl;
                    break;
                }

                state->restart.store(false);
            }

            // Update simulation
            {
                std::lock_guard<std::mutex> lock(simulationMutex);
                if (simulation)
                {
                    simulation->update();
                }
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

void SimulationThread::updatePerformanceMetrics(double frameTime)
{
    // Track the simulation iteration time
    state->lastIterationTime = frameTime;
}

SimulationData SimulationThread::getSimulationData()
{
    SimulationData data;

    // Acceder a la simulación bajo protección de mutex
    {
        std::lock_guard<std::mutex> lock(simulationMutex);
        if (simulation)
        {
            data.simulation = simulation.get();
            data.valid = true;
        }
        else
        {
            data.simulation = nullptr;
            data.valid = false;
        }
    }

    return data;
}
