#include "../../include/simulation/simulation_factory.hpp"
#include <iostream>

std::unique_ptr<SimulationBase> SimulationFactory::createFromState(const SimulationState &state)
{
    // Extract parameters from state
    int numBodies = state.numBodies.load();
    SimulationMethod method = state.simulationMethod.load();
    bool useSFC = state.useSFC.load();
    SFCOrderingMode orderingMode = state.sfcOrderingMode.load();
    int reorderFreq = state.reorderFrequency.load();
    sfc::CurveType curveType = state.sfcCurveType.load();
    BodyDistribution distribution = state.bodyDistribution.load();
    MassDistribution massDistribution = state.massDistribution.load();
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
        curveType,
        distribution,
        massDistribution,
        seed,
        useOpenMP,
        numThreads);
}

std::unique_ptr<SimulationBase> SimulationFactory::createSimulation(
    SimulationMethod method,
    int numBodies,
    bool useSFC,
    SFCOrderingMode orderingMode,
    int reorderFreq,
    sfc::CurveType curveType,
    BodyDistribution distribution,
    MassDistribution massDistribution,
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
                  << ", Curve=" << (curveType == sfc::CurveType::MORTON ? "Morton" : "Hilbert")
                  << ", ReorderFreq=" << reorderFreq;
    }

    std::cout << ", Seed=" << seed << std::endl;

    // Create appropriate simulation based on method
    switch (method)
    {
    case SimulationMethod::CPU_DIRECT_SUM:
        simulation = std::make_unique<CPUDirectSum>(
            numBodies,
            useOpenMP,
            numThreads,
            distribution,
            seed,
            massDistribution);
        break;

    case SimulationMethod::CPU_SFC_DIRECT_SUM:
    {
        auto sfcSim = std::make_unique<SFCCPUDirectSum>(
            numBodies,
            useOpenMP,
            numThreads,
            true,
            reorderFreq,
            distribution,
            seed,
            massDistribution);

        simulation = std::move(sfcSim);
    }
    break;

    case SimulationMethod::GPU_DIRECT_SUM:
        simulation = std::make_unique<GPUDirectSum>(
            numBodies,
            distribution,
            seed,
            massDistribution);
        break;

    case SimulationMethod::GPU_SFC_DIRECT_SUM:
    {
        auto sfcSim = std::make_unique<SFCGPUDirectSum>(
            numBodies,
            true, // SFC is always enabled for this method
            reorderFreq,
            distribution,
            seed,
            massDistribution);

        sfcSim->setCurveType(curveType);

        simulation = std::move(sfcSim);
    }
    break;

    case SimulationMethod::CPU_BARNES_HUT:
        simulation = std::make_unique<CPUBarnesHut>(
            numBodies,
            useOpenMP,
            numThreads,
            distribution,
            seed,
            massDistribution);
        break;

    case SimulationMethod::CPU_SFC_BARNES_HUT:
    {
        auto sfcSim = std::make_unique<CPUBarnesHut>(
            numBodies,
            useOpenMP,
            numThreads,
            distribution,
            seed,
            massDistribution,
            true,
            orderingMode, // Ordering mode
            reorderFreq);

        simulation = std::move(sfcSim);
    }
    break;

    case SimulationMethod::GPU_SFC_BARNES_HUT:
    {
        auto sfcSim = std::make_unique<SFCBarnesHut>(
            numBodies,
            true,
            orderingMode,
            reorderFreq,
            distribution,
            seed,
            massDistribution);

        // Set curve type
        sfcSim->setCurveType(curveType);

        simulation = std::move(sfcSim);
    }
    break;

    case SimulationMethod::GPU_BARNES_HUT:
    default:
        // Use regular Barnes-Hut
        simulation = std::make_unique<BarnesHut>(
            numBodies,
            distribution,
            seed,
            massDistribution);
        break;
    }

    return simulation;
}