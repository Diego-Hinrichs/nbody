#include "../../include/utils/metrics.hpp"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

// Initialize global variables
std::string globalCsvFilename;
std::ofstream globalOutputFile;

void initializeGlobalCsv()
{
    // Create a timestamp for the global output file
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "nbody_results_" << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".csv";
    globalCsvFilename = ss.str();

    // Open the global CSV file for writing
    globalOutputFile.open(globalCsvFilename);
    if (!globalOutputFile.is_open())
    {
        std::cerr << "Failed to open global CSV file for writing" << std::endl;
        return;
    }

    // Write header
    globalOutputFile << "type,simulation_id,algorithm,bodies,sort_type,use_sfc,theta,threads,block_size,dynamic_reordering,metrics_window,random_seed,mass_distribution,step,time_ms,kinetic_energy,potential_energy,total_energy,force_time,total_update_time,bbox_time,reset_time,octree_time,reorder_time,sort_time\n";
    globalOutputFile.flush();

    std::cout << "Global CSV file initialized: " << globalCsvFilename << std::endl;
}

void reportMetrics(SimulationData &simData, double totalSimTime, int iterations, const std::string &simulationId)
{
    if (!simData.valid || !simData.simulation)
    {
        std::cout << "No valid simulation data available for metrics reporting" << std::endl;
        return;
    }

    // Get simulation metrics
    SimulationMetrics metrics = simData.simulation->getMetrics();

    // Make sure the global file is open
    if (!globalOutputFile.is_open())
    {
        initializeGlobalCsv();
    }

    // Write metrics data to the global CSV
    globalOutputFile << "metrics," << simulationId << ",NA," << simData.simulation->getNumBodies() << ",NA,NA,NA,NA,NA,NA,NA,NA,NA,"
                     << "summary," << totalSimTime << ",NA,NA,NA,"
                     << metrics.forceTimeMs << "," << metrics.totalTimeMs << ","
                     << metrics.bboxTimeMs << "," << metrics.resetTimeMs << ","
                     << metrics.octreeTimeMs << "," << metrics.reorderTimeMs << ","
                     << metrics.sortTimeMs << "\n";

    // Energy metrics
    double kineticEnergy = simData.simulation->getKineticEnergy();
    double potentialEnergy = simData.simulation->getPotentialEnergy();
    double totalEnergy = kineticEnergy + potentialEnergy;

    globalOutputFile << "energy," << simulationId << ",NA," << simData.simulation->getNumBodies() << ",NA,NA,NA,NA,NA,NA,NA,NA,NA,"
                     << "summary," << totalSimTime << "," << kineticEnergy << ","
                     << potentialEnergy << "," << totalEnergy << ",NA,NA,NA,NA,NA,NA,NA\n";

    globalOutputFile.flush();

    // Also print to console for immediate feedback
    std::cout << "Metrics saved to global CSV: " << globalCsvFilename << std::endl;
    std::cout << "total_time_ms: " << totalSimTime << std::endl;
    std::cout << "iterations: " << iterations << std::endl;
    std::cout << "avg_time_per_iteration_ms: " << (totalSimTime / iterations) << std::endl;
    std::cout << "force_time_ms: " << metrics.forceTimeMs << std::endl;
    std::cout << "total_update_time_ms: " << metrics.totalTimeMs << std::endl;
    if (metrics.bboxTimeMs > 0)
        std::cout << "bbox_time_ms: " << metrics.bboxTimeMs << std::endl;
    if (metrics.resetTimeMs > 0)
        std::cout << "reset_time_ms: " << metrics.resetTimeMs << std::endl;
    if (metrics.octreeTimeMs > 0)
        std::cout << "octree_time_ms: " << metrics.octreeTimeMs << std::endl;
    if (metrics.reorderTimeMs > 0)
        std::cout << "reorder_time_ms: " << metrics.reorderTimeMs << std::endl;
    if (metrics.sortTimeMs > 0)
        std::cout << "sort_time_ms: " << metrics.sortTimeMs << std::endl;
    std::cout << "kinetic_energy: " << kineticEnergy << std::endl;
    std::cout << "potential_energy: " << potentialEnergy << std::endl;
    std::cout << "total_energy: " << totalEnergy << std::endl;
} 