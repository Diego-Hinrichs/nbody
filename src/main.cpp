#include <iostream>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <memory>
#include <functional>
#include <fstream>
#include <vector>
#include <omp.h>
#include <iomanip>
#include <sstream>

#include "../include/common/constants.cuh"

#include "../include/simulation/base/base.cuh"
#include "../include/simulation/simulation_thread.hpp"

#include "../include/simulation/implementations/cpu/direct_sum.hpp"
#include "../include/simulation/implementations/cpu/barnes_hut.hpp"
#include "../include/simulation/implementations/cpu/sfc_variants.hpp"

#include "../include/simulation/implementations/gpu/direct_sum.cuh"
#include "../include/simulation/implementations/gpu/barnes_hut.cuh"
#include "../include/simulation/implementations/gpu/sfc_variants.cuh"

#include "../include/ui/simulation_state.hpp"

// Define the global variables
double g_theta = 0.5;  // Default theta value
int g_blockSize = 256; // Default block size

// Global variables for CSV management
std::string globalCsvFilename;
std::ofstream globalOutputFile;

void initializeGlobalCsv() {
    // Create a timestamp for the global output file
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::stringstream ss;
    ss << "nbody_results_" << std::put_time(std::localtime(&now_c), "%Y%m%d_%H%M%S") << ".csv";
    globalCsvFilename = ss.str();

    // Open the global CSV file for writing
    globalOutputFile.open(globalCsvFilename);
    if (!globalOutputFile.is_open()) {
        std::cerr << "Failed to open global CSV file for writing" << std::endl;
        return;
    }

    // Write header
    globalOutputFile << "type,simulation_id,algorithm,bodies,sort_type,use_sfc,theta,threads,block_size,dynamic_reordering,metrics_window,random_seed,mass_distribution,step,time_ms,kinetic_energy,potential_energy,total_energy,force_time,total_update_time,bbox_time,reset_time,octree_time,reorder_time,sort_time\n";
    globalOutputFile.flush();
    
    std::cout << "Global CSV file initialized: " << globalCsvFilename << std::endl;
}

// Configuration structure
struct SimulationConfig
{
    int initialBodies = 1024;
    int sortType = 0;                       // 0: none, 1: hilbert, 2: morton
    int numSteps = 10000000;                // Number of simulation steps
    int massDistribution = 0;               // 0: uniform, 1: normal
    int algorithm = 0;                      // 0: cpu-direct-sum, 1: cpu-barnes, 2: gpu-direct-sum, 3: gpu-barnes-hut
    float theta = 0.5f;                     // Barnes-Hut parameter
    bool visualization = false;              // Always false in headless mode
    std::string energyOutput = "";          // Output file for energy data
    int numThreads = omp_get_max_threads(); // Default to max threads for CPU implementations
    int blockSize = 256;                    // Block size for GPU implementations
    bool fullscreen = false;                // Not used in headless mode
    bool useSFC = false;                    // Used for space-filling curve options
    bool verbose = false;
    bool headless = true;                   // Always true in headless mode
    bool reportMetrics = false;             // Report detailed metrics at the end
    unsigned int randomSeed = 12345;        // Random seed for reproducibility
    bool dynamicReordering = true;          // Use dynamic reordering for Barnes-Hut SFC
    int metricsWindowSize = 10;             // Window size for dynamic reordering metrics
    bool benchmark = false;
};

void runSimulationOnce(const SimulationConfig &config);

// Parse command-line arguments
SimulationConfig parseArgs(int argc, char **argv)
{
    SimulationConfig config;

    for (int i = 1; i < argc; i++)
    {
        std::string arg = argv[i];

        if (arg == "-n" && i + 1 < argc || arg == "--bodies" && i + 1 < argc)
        {
            config.initialBodies = std::stoi(argv[++i]);
        }
        else if (arg == "-sort" && i + 1 < argc)
        {
            config.sortType = std::stoi(argv[++i]);
            config.useSFC = (config.sortType > 0); // Enable SFC if using Hilbert or Morton
        }
        else if (arg == "-steps" && i + 1 < argc || arg == "--iterations" && i + 1 < argc)
        {
            config.numSteps = std::stoi(argv[++i]);
        }
        else if (arg == "-mdist" && i + 1 < argc)
        {
            config.massDistribution = std::stoi(argv[++i]);
        }
        else if (arg == "-alg" && i + 1 < argc || arg == "--method" && i + 1 < argc)
        {
            config.algorithm = std::stoi(argv[++i]);
        }
        else if (arg == "-theta" && i + 1 < argc)
        {
            config.theta = std::stof(argv[++i]);
        }
        else if (arg == "-energy" && i + 1 < argc)
        {
            config.energyOutput = argv[++i];
        }
        else if (arg == "-nt" && i + 1 < argc)
        {
            config.numThreads = std::stoi(argv[++i]);
        }
        else if (arg == "-bs" && i + 1 < argc)
        {
            config.blockSize = std::stoi(argv[++i]);
        }
        else if (arg == "--headless")
        {
            config.headless = true;
            config.visualization = false; // Headless mode disables visualization
        }
        else if (arg == "--report-metrics")
        {
            config.reportMetrics = true;
        }
        else if (arg == "--use-sfc" && i + 1 < argc)
        {
            config.useSFC = (std::string(argv[++i]) == "true" || std::string(argv[i]) == "1");
            if (config.useSFC)
            {
                // Default to Morton curve if SFC is enabled but no specific curve is set
                if (config.sortType == 0)
                    config.sortType = 2; // Set to Morton
            }
        }
        else if (arg == "--seed" && i + 1 < argc)
        {
            config.randomSeed = std::stoul(argv[++i]);
        }
        else if (arg == "--dynamic-reordering" && i + 1 < argc)
        {
            std::string val = argv[++i];
            config.dynamicReordering = (val == "true" || val == "1");
        }
        else if (arg == "--metrics-window" && i + 1 < argc)
        {
            config.metricsWindowSize = std::stoi(argv[++i]);
        }
        else if (arg == "-verbose")
        {
            config.verbose = true;
        }
        else if (arg == "--benchmark")
        {
            config.benchmark = true;
        }
        else if (arg == "-help" || arg == "--help" || arg == "-h")
        {
            std::cout << "N-Body Simulation Usage:\n"
                      << "  ./prog [options]\n"
                      << "Options:\n"
                      << "  -n, --bodies <particles>     : Number of particles (default: 1024)\n"
                      << "  -sort <type>                 : Space-filling curve type (0: none, 1: hilbert, 2: morton) (default: 0)\n"
                      << "  -steps, --iterations <steps> : Number of simulation steps (default: 1000)\n"
                      << "  -mdist <type>                : Mass distribution (0: uniform, 1: normal) (default: 0)\n"
                      << "  -alg, --method <algorithm>   : Algorithm (0: cpu-direct-sum, 1: cpu-barnes, 2: gpu-direct-sum, 3: gpu-barnes-hut) (default: 0)\n"
                      << "  -theta <float>               : Barnes-Hut theta parameter (default: 0.5)\n"
                      << "  -energy <filename>           : Output energy data to file\n"
                      << "  -nt <threads>                : Number of threads for CPU algorithms (default: 1)\n"
                      << "  -bs <blocksize>              : Block size for GPU algorithms (default: 256)\n"
                      << "  --headless                   : Run without visualization or UI\n"
                      << "  --report-metrics             : Report detailed performance metrics at the end\n"
                      << "  --use-sfc <true|false>       : Enable/disable space-filling curve ordering\n"
                      << "  --dynamic-reordering <true|false> : Enable/disable dynamic reordering for SFC Barnes-Hut\n"
                      << "  --metrics-window <size>      : Window size for dynamic reordering metrics (default: 10)\n"
                      << "  --seed <number>              : Random seed for reproducible simulations\n"
                      << "  -verbose                     : Enable verbose output\n"
                      << "  -help, --help, -h            : Show this help message\n";
            exit(0);
        }
    }

    if (config.verbose)
    {
        std::cout << "Configuration:\n"
                  << "  Particles: " << config.initialBodies << "\n"
                  << "  Sort Type: " << config.sortType << "\n"
                  << "  Steps: " << config.numSteps << "\n"
                  << "  Mass Distribution: " << config.massDistribution << "\n"
                  << "  Algorithm: " << config.algorithm << "\n"
                  << "  Theta: " << config.theta << "\n"
                  << "  Visualization: " << (config.visualization ? "On" : "Off") << "\n"
                  << "  Headless: " << (config.headless ? "Yes" : "No") << "\n"
                  << "  Report Metrics: " << (config.reportMetrics ? "Yes" : "No") << "\n"
                  << "  SFC Enabled: " << (config.useSFC ? "Yes" : "No") << "\n"
                  << "  Dynamic Reordering: " << (config.dynamicReordering ? "Yes" : "No") << "\n"
                  << "  Metrics Window Size: " << config.metricsWindowSize << "\n"
                  << "  Energy Output: " << (config.energyOutput.empty() ? "None" : config.energyOutput) << "\n"
                  << "  Threads: " << config.numThreads << "\n"
                  << "  Block Size: " << config.blockSize << "\n"
                  << "  Random Seed: " << config.randomSeed << "\n";
    }

    return config;
}

// Global simulation state for callbacks
SimulationState *g_simulationState = nullptr;

void reportMetrics(SimulationData &simData, double totalSimTime, int iterations, const std::string &simulationId = "")
{
    if (!simData.valid || !simData.simulation)
    {
        std::cout << "No valid simulation data available for metrics reporting" << std::endl;
        return;
    }

    // Get simulation metrics
    SimulationMetrics metrics = simData.simulation->getMetrics();

    // Make sure the global file is open
    if (!globalOutputFile.is_open()) {
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
    if (metrics.bboxTimeMs > 0) std::cout << "bbox_time_ms: " << metrics.bboxTimeMs << std::endl;
    if (metrics.resetTimeMs > 0) std::cout << "reset_time_ms: " << metrics.resetTimeMs << std::endl;
    if (metrics.octreeTimeMs > 0) std::cout << "octree_time_ms: " << metrics.octreeTimeMs << std::endl;
    if (metrics.reorderTimeMs > 0) std::cout << "reorder_time_ms: " << metrics.reorderTimeMs << std::endl;
    if (metrics.sortTimeMs > 0) std::cout << "sort_time_ms: " << metrics.sortTimeMs << std::endl;
    std::cout << "kinetic_energy: " << kineticEnergy << std::endl;
    std::cout << "potential_energy: " << potentialEnergy << std::endl;
    std::cout << "total_energy: " << totalEnergy << std::endl;
}

int main(int argc, char **argv)
{
    std::cout << "Attempting to use dedicated GPU..." << std::endl;
    checkCudaAvailability();
    
    // Parse command-line arguments
    SimulationConfig config = parseArgs(argc, argv);
    
    if (config.benchmark)
    {
        std::cout << "Starting benchmark mode..." << std::endl;
        
        // Reduced number of steps for benchmarking
        const int BENCHMARK_STEPS = 100;
        
        std::vector<int> methods = {0, 1, 2, 3};  // Reduced set of methods for testing
        std::vector<int> bodyCounts = {1000, 5000, 10000};
        
        for (int method : methods)
        {
            std::cout << "\nTesting method " << method << std::endl;
            bool useSFC = (method == 1 || method == 3);
            
            for (int bodies : bodyCounts)
            {
                std::cout << "\nTesting with " << bodies << " bodies" << std::endl;
                
                for (int i = 0; i < 5; ++i)
                {
                    std::cout << "\nRun " << (i + 1) << " of 5" << std::endl;
                    
                    // Create a new configuration for each run
                    SimulationConfig benchConfig = config;
                    benchConfig.algorithm = method;
                    benchConfig.initialBodies = bodies;
                    benchConfig.useSFC = useSFC;
                    benchConfig.sortType = useSFC ? 2 : 0; // Morton by default
                    benchConfig.visualization = false;
                    benchConfig.headless = true;
                    benchConfig.reportMetrics = true;
                    benchConfig.numSteps = BENCHMARK_STEPS;
                    
                    try {
                        std::cout << "Starting benchmark: method=" << method 
                                 << " bodies=" << bodies 
                                 << " run=" << (i + 1) << std::endl;
                        
                        runSimulationOnce(benchConfig);
                        
                        std::cout << "Benchmark completed successfully" << std::endl;
                        
                        // Add a small delay between runs
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    }
                    catch (const std::exception& e) {
                        std::cerr << "Error in benchmark run: " << e.what() << std::endl;
                        continue;
                    }
                }
            }
        }
        
        std::cout << "Benchmark completed." << std::endl;
    }
    else {
        // If not in benchmark mode, run a single simulation
        runSimulationOnce(config);
    }
    
    // Close the global CSV file if it's open
    if (globalOutputFile.is_open()) {
        std::cout << "Closing global CSV file: " << globalCsvFilename << std::endl;
        globalOutputFile.close();
    }
    
    return 0;
}

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
    if (!globalOutputFile.is_open()) {
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

            try {
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

                if (step % 100 == 0) {
                    std::cout << "Completed step " << step << std::endl;
                }
            }
            catch (const std::exception& e) {
                std::cerr << "Error processing step " << step << ": " << e.what() << std::endl;
                continue;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        std::cout << "Main simulation loop completed" << std::endl;

        SimulationData finalSimData = simulationThread.getSimulationData();
        if (!finalSimData.valid || !finalSimData.simulation) {
            std::cerr << "Invalid final simulation data" << std::endl;
        } else {
            auto endTime = std::chrono::high_resolution_clock::now();
            double totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

            if (config.reportMetrics)
            {
                try {
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
                catch (const std::exception& e) {
                    std::cerr << "Error writing final metrics: " << e.what() << std::endl;
                }
            }
        }
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error in simulation: " << e.what() << std::endl;
    }

    std::cout << "Simulation complete. Results saved to: " << globalCsvFilename << std::endl;

    std::cout << "Stopping simulation thread..." << std::endl;
    simulationState.running.store(false);
    simulationThread.join();
    std::cout << "Simulation completed." << std::endl;
}
