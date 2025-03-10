#ifndef SIMULATION_STATE_H
#define SIMULATION_STATE_H

#include <atomic>
#include <mutex>
#include <vector>
#include <string>
#include "../common/types.cuh"
#include "../sfc/sfc_framework.cuh"

// Enum for SFC ordering mode
enum class SFCOrderingMode
{
    PARTICLES, // Order particles using SFC
    OCTANTS    // Order octants using SFC
};

// Enum for initial body distribution
enum class BodyDistribution
{
    SOLAR_SYSTEM,
    GALAXY,
    BINARY_SYSTEM,
    UNIFORM_SPHERE,
    RANDOM_CLUSTERS
};

// Enum for simulation method
enum class SimulationMethod
{
    CPU_DIRECT_SUM,
    CPU_SFC_DIRECT_SUM,

    GPU_DIRECT_SUM,
    GPU_SFC_DIRECT_SUM,

    CPU_BARNES_HUT,
    CPU_SFC_BARNES_HUT,

    GPU_BARNES_HUT,
    GPU_SFC_BARNES_HUT
};

struct SimulationState
{
    // Simulation control flags
    std::atomic<bool> running;
    std::atomic<bool> restart;
    std::atomic<bool> useSFC;
    std::atomic<bool> isPaused;

    // Simulation method
    std::atomic<SimulationMethod> simulationMethod;
    std::atomic<bool> useOpenMP;    // For CPU methods
    std::atomic<int> openMPThreads; // For CPU methods

    // SFC specific parameters
    std::atomic<SFCOrderingMode> sfcOrderingMode;
    std::atomic<int> reorderFrequency;        // How often to reorder (iterations)
    std::atomic<sfc::CurveType> sfcCurveType; // Type of SFC (Morton or Hilbert)

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