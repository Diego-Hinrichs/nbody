#include "../../include/ui/simulation_state.hpp"
#include <omp.h>

SimulationState::SimulationState() : running(true),
                                     restart(false),
                                     isPaused(false),
                                     simulationMethod(SimulationMethod::GPU_BARNES_HUT), // Default to Barnes-Hut
                                     useOpenMP(true),                                    // Enable OpenMP by default
                                     openMPThreads(omp_get_max_threads()),               // Use all available cores by default
                                     useSFC(false),
                                     sfcOrderingMode(SFCOrderingMode::PARTICLES),
                                     reorderFrequency(10),                 // Reorder every 10 iterations by default
                                     sfcCurveType(sfc::CurveType::MORTON), // Default to Morton curve
                                     bodyDistribution(BodyDistribution::SOLAR_SYSTEM),
                                     randomSeed(static_cast<unsigned int>(time(nullptr))),
                                     seedWasChanged(false),
                                     numBodies(1024),
                                     zoomFactor(1.0),
                                     offsetX(0.0),
                                     offsetY(0.0),
                                     sharedBodies(nullptr),
                                     currentBodiesCount(0),
                                     fps(0.0),
                                     lastIterationTime(0.0),
                                     showCommandMenu(false),
                                     selectedCommandIndex(0),
                                     selectedParticleOption(0),
                                     showOctree(false),
                                     octreeMaxDepth(3),
                                     octreeOpacity(0.5f),
                                     octreeColorByMass(true)
{
    // Initialize seed input buffer with current seed
    snprintf(seedInputBuffer, sizeof(seedInputBuffer), "%u", randomSeed.load());
}

SimulationState::~SimulationState()
{
    // Free shared bodies if allocated
    if (sharedBodies != nullptr)
    {
        delete[] sharedBodies;
        sharedBodies = nullptr;
    }
}