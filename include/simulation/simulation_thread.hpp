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
#include <mutex>

// Estructura para compartir datos de simulación de forma segura entre hilos
struct SimulationData {
    SimulationBase* simulation;  // Puntero no administrado (raw pointer)
    bool valid;                  // Indica si los datos son válidos
};

class SimulationThread
{
private:
    SimulationState *state;
    std::thread thread;
    std::unique_ptr<SimulationBase> simulation;
    std::mutex simulationMutex;  // Mutex para proteger el acceso a simulation

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

    void run();

    bool checkForParameterChanges();

    void updateCurrentParameters();

    void updateVisualizationData();

    void updatePerformanceMetrics(double frameTime);

public:
    explicit SimulationThread(SimulationState *simulationState);

    ~SimulationThread();

    void start();

    void stop();

    void join();

    SimulationData getSimulationData();
};

#endif // SIMULATION_THREAD_HPP