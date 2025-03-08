#include "../../include/ui/simulation_state.h"

SimulationState::SimulationState() : running(true),
                                     restart(false),
                                     useSFC(false),
                                     isPaused(false),
                                     sfcOrderingMode(SFCOrderingMode::PARTICLES),
                                     reorderFrequency(10), // Reorder every 10 iterations by default
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
                                     selectedParticleOption(0)
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