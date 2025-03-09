#include "../../include/simulation/simulation_factory.hpp"
#include <iostream>

std::unique_ptr<SimulationBase> SimulationFactory::createSimulation(
    SimulationMethod method,
    int numBodies,
    bool useSFC,
    SFCOrderingMode orderingMode,
    int reorderFreq,
    BodyDistribution distribution,
    unsigned int seed,
    bool useOpenMP,
    int numThreads)
{
    std::unique_ptr<SimulationBase> simulation;

    // Log creation
    std::cout << "Creating simulation: "
              << "Method=" << static_cast<int>(method)
              << ", Bodies=" << numBodies
              << ", SFC=" << (useSFC ? "enabled" : "disabled");

    if (useSFC)
    {
        std::cout << ", Ordering=" << (orderingMode == SFCOrderingMode::PARTICLES ? "particles" : "octants")
                  << ", ReorderFreq=" << reorderFreq;
    }

    std::cout << ", Seed=" << seed << std::endl;

    // Create appropriate simulation based on method
    switch (method)
    {
    case SimulationMethod::CPU_DIRECT_SUM:
        simulation = std::make_unique<CPUDirectSum>(
            numBodies,
            useOpenMP,    // Use OpenMP
            numThreads,   // Thread count
            distribution, // Distribution type
            seed          // Random seed
        );
        break;

    case SimulationMethod::CPU_SFC_DIRECT_SUM:
        simulation = std::make_unique<SFCCPUDirectSum>(
            numBodies,
            useOpenMP,    // Use OpenMP
            numThreads,   // Thread count
            true,         // Enable SFC
            reorderFreq,  // Reorder frequency
            distribution, // Distribution type
            seed          // Random seed
        );
        break;

    case SimulationMethod::GPU_DIRECT_SUM:
        simulation = std::make_unique<GPUDirectSum>(
            numBodies,
            distribution, // Distribution type
            seed          // Random seed
        );
        break;

    case SimulationMethod::GPU_SFC_DIRECT_SUM:
        simulation = std::make_unique<SFCGPUDirectSum>(
            numBodies,
            true,         // SFC is always enabled for this method
            reorderFreq,  // Reorder frequency
            distribution, // Distribution type
            seed          // Random seed
        );
        break;

    case SimulationMethod::CPU_BARNES_HUT:
        simulation = std::make_unique<CPUBarnesHut>(
            numBodies,
            useOpenMP,    // Use OpenMP
            numThreads,   // Thread count
            distribution, // Distribution type
            seed          // Random seed
        );
        break;

    case SimulationMethod::CPU_SFC_BARNES_HUT:
        simulation = std::make_unique<CPUBarnesHut>(
            numBodies,
            useOpenMP,    // Use OpenMP
            numThreads,   // Thread count
            distribution, // Distribution type
            seed,         // Random seed
            true,         // Enable SFC
            orderingMode, // Ordering mode
            reorderFreq   // Reorder frequency
        );
        break;

    case SimulationMethod::GPU_SFC_BARNES_HUT:
        simulation = std::make_unique<SFCBarnesHut>(
            numBodies,
            true,         // SFC is always enabled for this method
            orderingMode, // Ordering mode
            reorderFreq,  // Reorder frequency
            distribution, // Distribution type
            seed          // Random seed
        );
        break;

    case SimulationMethod::GPU_BARNES_HUT:
    default:
        if (useSFC)
        {
            // Use SFCBarnesHut if SFC is enabled
            simulation = std::make_unique<SFCBarnesHut>(
                numBodies,
                true,         // Enable SFC
                orderingMode, // Ordering mode
                reorderFreq,  // Reorder frequency
                distribution, // Distribution type
                seed          // Random seed
            );
        }
        else
        {
            // Use regular Barnes-Hut
            simulation = std::make_unique<BarnesHut>(
                numBodies,
                distribution, // Distribution type
                seed          // Random seed
            );
        }
        break;
    }

    return simulation;
}

std::unique_ptr<SimulationBase> SimulationFactory::createFromState(const SimulationState &state)
{
    // Extract parameters from state
    int numBodies = state.numBodies.load();
    SimulationMethod method = state.simulationMethod.load();
    bool useSFC = state.useSFC.load();
    SFCOrderingMode orderingMode = state.sfcOrderingMode.load();
    int reorderFreq = state.reorderFrequency.load();
    BodyDistribution distribution = state.bodyDistribution.load();
    unsigned int seed = state.randomSeed.load();
    bool useOpenMP = state.useOpenMP.load();
    int numThreads = state.openMPThreads.load();

    // Create simulation using the factory method
    return createSimulation(
        method,
        numBodies,
        useSFC,
        orderingMode,
        reorderFreq,
        distribution,
        seed,
        useOpenMP,
        numThreads);
}