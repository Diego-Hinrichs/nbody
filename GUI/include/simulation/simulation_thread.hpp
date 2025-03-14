#ifndef SIMULATION_THREAD_HPP
#define SIMULATION_THREAD_HPP

#include "base/base.cuh"
#include "simulation_factory.hpp"

#include "../common/types.cuh"
#include "../ui/simulation_state.hpp"

#include <thread>
#include <atomic>
#include <memory>
#include <chrono>

/**
 * @brief Class to encapsulate simulation thread logic
 *
 * This class manages the simulation thread, handling state updates
 * and coordinating with the rendering thread.
 */
class SimulationThread
{
private:
    SimulationState *state;
    std::thread thread;
    std::unique_ptr<SimulationBase> simulation;

    // Current simulation parameters (for detecting changes)
    int currentNumBodies;
    SimulationMethod currentMethod;
    bool currentUseSFC;
    SFCOrderingMode currentOrderingMode;
    int currentReorderFreq;
    BodyDistribution currentDistribution;
    unsigned int currentSeed;
    bool currentUseOpenMP;
    int currentOpenMPThreads;

    // Visualization parameters
    const int VISUALIZATION_FREQUENCY = 24; // Update visualization every 24 frames
    int frameCounter;

    // Performance tracking
    std::chrono::steady_clock::time_point lastTime;
    double frameTimeAccum;
    int frameCount;

    /**
     * @brief Main thread function that runs the simulation
     */
    void run();

    /**
     * @brief Check if simulation parameters have changed
     * @return true if restart is needed, false otherwise
     */
    bool checkForParameterChanges();

    /**
     * @brief Update current parameter values from state
     */
    void updateCurrentParameters();

    /**
     * @brief Update visualization data for rendering
     */
    void updateVisualizationData();

    /**
     * @brief Update performance metrics
     * @param frameTime Time taken to process this frame in milliseconds
     */
    void updatePerformanceMetrics(double frameTime);

public:
    /**
     * @brief Constructor
     * @param simulationState Pointer to shared simulation state
     */
    explicit SimulationThread(SimulationState *simulationState);

    /**
     * @brief Destructor
     */
    ~SimulationThread();

    /**
     * @brief Start the simulation thread
     */
    void start();

    /**
     * @brief Stop the simulation thread and wait for it to finish
     */
    void stop();

    /**
     * @brief Join the simulation thread (wait for it to finish)
     */
    void join();
};

#endif // SIMULATION_THREAD_HPP