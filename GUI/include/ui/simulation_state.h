#ifndef SIMULATION_STATE_H
#define SIMULATION_STATE_H

#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include "../include/common/types.cuh"

struct SimulationState
{
    // Simulation control flags
    std::atomic<bool> running;
    std::atomic<bool> restart;
    std::atomic<bool> useSFC;
    std::atomic<bool> isPaused;

    // Simulation parameters
    std::atomic<int> numBodies;
    std::atomic<double> zoomFactor;

    // Thread synchronization
    std::mutex mtx;

    // Simulation parameters
    double offsetX;
    double offsetY;

    // Visualization state
    Body *sharedBodies;
    int currentBodiesCount;
    double fps;
    double lastIterationTime;

    // UI state
    bool showCommandMenu;
    int selectedCommandIndex;
    std::vector<std::string> commandOptions;
    std::vector<std::string> particleOptions;
    int selectedParticleOption;

    // Constructor with default initialization
    SimulationState();

    // Destructor to clean up resources
    ~SimulationState();

    // Disable copy constructor and assignment operator
    SimulationState(const SimulationState &) = delete;
    SimulationState &operator=(const SimulationState &) = delete;
};

#endif // SIMULATION_STATE_H