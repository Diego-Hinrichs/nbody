#ifndef SIMULATION_STATE_H
#define SIMULATION_STATE_H

#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include "../common/types.cuh"

// Enum for SFC ordering mode
enum class SFCOrderingMode
{
    PARTICLES, // Order particles using SFC
    OCTANTS    // Order octants using SFC
};

// Enum for initial body distribution
enum class BodyDistribution
{
    SOLAR_SYSTEM,   // Solar system like distribution
    GALAXY,         // Galaxy-like spiral
    BINARY_SYSTEM,  // Binary star system
    UNIFORM_SPHERE, // Uniform distribution in sphere
    RANDOM_CLUSTERS // Random clusters of bodies
};

struct SimulationState
{
    // Simulation control flags
    std::atomic<bool> running;
    std::atomic<bool> restart;
    std::atomic<bool> useSFC;
    std::atomic<bool> isPaused;

    // SFC specific parameters
    std::atomic<SFCOrderingMode> sfcOrderingMode;
    std::atomic<int> reorderFrequency; // How often to reorder (iterations)

    // Distribution parameters
    std::atomic<BodyDistribution> bodyDistribution;
    std::atomic<unsigned int> randomSeed; // Seed for random number generation
    bool seedWasChanged;                  // Flag to indicate seed was manually set

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
    int selectedParticleOption;
    char seedInputBuffer[16]; // Buffer for seed input text

    // Constructor with default initialization
    SimulationState();

    // Destructor to clean up resources
    ~SimulationState();

    // Disable copy constructor and assignment operator
    SimulationState(const SimulationState &) = delete;
    SimulationState &operator=(const SimulationState &) = delete;
};

#endif // SIMULATION_STATE_H