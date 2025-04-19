#include "../../include/ui/simulation_state.hpp"
#include <omp.h>

SimulationState::SimulationState() : running(true),
                                     restart(false),
                                     isPaused(false),
                                     simulationMethod(SimulationMethod::GPU_BARNES_HUT),
                                     useOpenMP(true),
                                     openMPThreads(omp_get_max_threads()),
                                     useSFC(false),
                                     sfcOrderingMode(SFCOrderingMode::PARTICLES),
                                     reorderFrequency(10),
                                     sfcCurveType(sfc::CurveType::MORTON),
                                     bodyDistribution(BodyDistribution::RANDOM_BODIES),
                                     massDistribution(MassDistribution::UNIFORM),
                                     randomSeed(static_cast<unsigned int>(time(nullptr))),
                                     seedWasChanged(false),
                                     numBodies(1024),
                                     lastIterationTime(0.0),
                                     useDynamicReordering(false),
                                     dynamicReorderFrequency(10),
                                     reorderTimeMs(0.0f),
                                     simTimeMs(0.0f)
{
}
SimulationState::~SimulationState()
{
}