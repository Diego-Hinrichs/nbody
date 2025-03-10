#ifndef SIMULATION_FACTORY_HPP
#define SIMULATION_FACTORY_HPP

#include <memory>
#include <functional>
#include "../common/types.cuh"
#include "../ui/simulation_state.hpp"
#include "simulation_base.cuh"
#include "cpu_direct_sum.hpp"
#include "cpu_sfc_direct_sum.hpp"
#include "cpu_barnes_hut.hpp"
#include "gpu_direct_sum.cuh"
#include "gpu_sfc_direct_sum.cuh"
#include "gpu_barnes_hut.cuh"
#include "gpu_sfc_barnes_hut.cuh"
#include "../sfc/sfc_framework.cuh"

/**
 * @brief Factory class to create simulation instances
 *
 * This class encapsulates the logic for creating different types of
 * simulations based on the configuration parameters.
 */
class SimulationFactory
{
public:
    /**
     * @brief Create a simulation instance based on parameters
     *
     * @param method Simulation method to use
     * @param numBodies Number of bodies in the simulation
     * @param useSFC Flag to enable/disable Space-Filling Curve
     * @param orderingMode SFC ordering mode (particles or octants)
     * @param reorderFreq Reordering frequency for SFC
     * @param curveType Type of SFC (Morton or Hilbert)
     * @param distribution Distribution type for bodies
     * @param seed Random seed
     * @param useOpenMP Flag to enable/disable OpenMP parallelization
     * @param numThreads Number of OpenMP threads
     * @return std::unique_ptr<SimulationBase> Smart pointer to created simulation
     */
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

    /**
     * @brief Create a simulation from simulation state
     *
     * @param state Current simulation state
     * @return std::unique_ptr<SimulationBase> Smart pointer to created simulation
     */
    static std::unique_ptr<SimulationBase> createFromState(const SimulationState &state);
};

#endif // SIMULATION_FACTORY_HPP