#include "../../include/utils/simulation_runner.hpp"
#include "../../include/utils/metrics.hpp"
#include "../../include/ui/simulation_state.hpp"
#include "../../include/simulation/simulation_thread.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>

// External declaration for global output file
extern std::string globalCsvFilename;
extern std::ofstream globalOutputFile;

void runSimulationOnce(const SimulationConfig &config)
{
    std::cout << "Starting simulation with " << config.initialBodies << " bodies..." << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    SimulationState simulationState;

    // Set simulation parameters
    simulationState.numBodies.store(config.initialBodies);
    simulationState.useSFC.store(config.useSFC);
    simulationState.randomSeed.store(config.randomSeed);
    simulationState.simulationMethod.store(static_cast<SimulationMethod>(config.algorithm));
    simulationState.massDistribution.store(config.massDistribution == 0 ? MassDistribution::UNIFORM : MassDistribution::NORMAL);
    simulationState.useOpenMP.store(config.numThreads > 1);
    simulationState.openMPThreads.store(config.numThreads);
    simulationState.dynamicReordering.store(config.dynamicReordering);
    simulationState.metricsWindowSize.store(config.metricsWindowSize);

    if (config.sortType > 0)
    {
        simulationState.sfcCurveType.store(config.sortType == 1 ? sfc::CurveType::HILBERT : sfc::CurveType::MORTON);
    }

    // Make sure the global CSV file is initialized
    if (!globalOutputFile.is_open())
    {
        initializeGlobalCsv();
    }

    // Generate a unique simulation ID
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "sim_" << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S");
    std::string simulationId = ss.str();

    std::cout << "Simulation ID: " << simulationId << std::endl;

    // Start simulation thread
    SimulationThread simulationThread(&simulationState);

    try
    {
        simulationThread.start();

        // Wait for simulation to initialize
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        // Start running the simulation
        simulationState.running.store(true);
        simulationState.isPaused.store(false);

        // Configure number of steps
        int targetSteps = config.numSteps > 0 ? config.numSteps : 1000;
        std::cout << "Running for " << targetSteps << " steps..." << std::endl;

        // Main simulation loop
        for (int step = 0; step < targetSteps; step++)
        {
            SimulationData simData = simulationThread.getSimulationData();
            if (!simData.valid || !simData.simulation)
            {
                std::cerr << "Invalid simulation data at step " << step << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                continue;
            }

            try
            {
                double kinetic = simData.simulation->getKineticEnergy();
                double potential = simData.simulation->getPotentialEnergy();
                double total = kinetic + potential;

                // Write data for this step to the global CSV
                globalOutputFile << "simulation,"
                                 << simulationId << ","
                                 << config.algorithm << ","
                                 << config.initialBodies << ","
                                 << config.sortType << ","
                                 << (config.useSFC ? "true" : "false") << ","
                                 << config.theta << ","
                                 << config.numThreads << ","
                                 << config.blockSize << ","
                                 << (config.dynamicReordering ? "true" : "false") << ","
                                 << config.metricsWindowSize << ","
                                 << config.randomSeed << ","
                                 << config.massDistribution << ","
                                 << step << ","
                                 << simulationState.lastIterationTime << ","
                                 << kinetic << ","
                                 << potential << ","
                                 << total << ",NA,NA,NA,NA,NA,NA,NA\n";

                globalOutputFile.flush();

                if (step % 100 == 0)
                {
                    std::cout << "Completed step " << step << std::endl;
                }
            }
            catch (const std::exception &e)
            {
                std::cerr << "Error processing step " << step << ": " << e.what() << std::endl;
                continue;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        std::cout << "Main simulation loop completed" << std::endl;

        SimulationData finalSimData = simulationThread.getSimulationData();
        if (!finalSimData.valid || !finalSimData.simulation)
        {
            std::cerr << "Invalid final simulation data" << std::endl;
        }
        else
        {
            auto endTime = std::chrono::high_resolution_clock::now();
            double totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

            if (config.reportMetrics)
            {
                try
                {
                    // Get simulation metrics
                    SimulationMetrics metrics = finalSimData.simulation->getMetrics();

                    // Write final metrics to global CSV
                    globalOutputFile << "final,"
                                     << simulationId << ","
                                     << config.algorithm << ","
                                     << config.initialBodies << ","
                                     << config.sortType << ","
                                     << (config.useSFC ? "true" : "false") << ","
                                     << config.theta << ","
                                     << config.numThreads << ","
                                     << config.blockSize << ","
                                     << (config.dynamicReordering ? "true" : "false") << ","
                                     << config.metricsWindowSize << ","
                                     << config.randomSeed << ","
                                     << config.massDistribution << ","
                                     << "final" << ","
                                     << totalTimeMs << ","
                                     << finalSimData.simulation->getKineticEnergy() << ","
                                     << finalSimData.simulation->getPotentialEnergy() << ","
                                     << (finalSimData.simulation->getKineticEnergy() + finalSimData.simulation->getPotentialEnergy()) << ","
                                     << metrics.forceTimeMs << ","
                                     << metrics.totalTimeMs << ","
                                     << metrics.bboxTimeMs << ","
                                     << metrics.resetTimeMs << ","
                                     << metrics.octreeTimeMs << ","
                                     << metrics.reorderTimeMs << ","
                                     << metrics.sortTimeMs << "\n";

                    globalOutputFile.flush();

                    // Also call reportMetrics to print to console
                    reportMetrics(finalSimData, totalTimeMs, targetSteps, simulationId);
                }
                catch (const std::exception &e)
                {
                    std::cerr << "Error writing final metrics: " << e.what() << std::endl;
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Fatal error in simulation: " << e.what() << std::endl;
    }

    std::cout << "Simulation complete. Results saved to: " << globalCsvFilename << std::endl;

    std::cout << "Stopping simulation thread..." << std::endl;
    simulationState.running.store(false);
    simulationThread.join();
    std::cout << "Simulation completed." << std::endl;
} 