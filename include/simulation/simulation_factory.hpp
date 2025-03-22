#ifndef SIMULATION_FACTORY_HPP
#define SIMULATION_FACTORY_HPP


#include "base/base.cuh"
#include "implementations/cpu/direct_sum.hpp"
#include "implementations/cpu/sfc_variants.hpp"
#include "implementations/cpu/barnes_hut.hpp"

#include "implementations/gpu/direct_sum.cuh"
#include "implementations/gpu/sfc_variants.cuh"
#include "implementations/gpu/barnes_hut.cuh"

#include "../sfc/sfc_framework.cuh"
#include "../ui/simulation_state.hpp"

#include <memory>
#include <functional>

class SimulationFactory
{
public:
    static std::unique_ptr<SimulationBase> createSimulation(
        SimulationMethod method,
        int numBodies,
        bool useSFC,
        SFCOrderingMode orderingMode,
        int reorderFreq,
        sfc::CurveType curveType,
        BodyDistribution distribution,
        unsigned int seed,
        bool useOpenMP,
        int numThreads);

    static std::unique_ptr<SimulationBase> createFromState(const SimulationState &state);
};

#endif // SIMULATION_FACTORY_HPP