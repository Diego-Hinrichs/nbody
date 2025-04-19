#pragma once

#include <string>
#include <omp.h>

// Configuration structure
struct SimulationConfig
{
    int initialBodies = 1024;
    int sortType = 0;                       // 0: none, 1: hilbert, 2: morton
    int numSteps = 10000000;                // Number of simulation steps
    int massDistribution = 0;               // 0: uniform, 1: normal
    int algorithm = 0;                      // 0: cpu-direct-sum, 1: cpu-barnes, 2: gpu-direct-sum, 3: gpu-barnes-hut
    float theta = 0.5f;                     // Barnes-Hut parameter
    bool visualization = false;             // Always false in headless mode
    std::string energyOutput = "";          // Output file for energy data
    int numThreads = omp_get_max_threads(); // Default to max threads for CPU implementations
    int blockSize = 256;                    // Block size for GPU implementations
    bool fullscreen = false;                // Not used in headless mode
    bool useSFC = false;                    // Used for space-filling curve options
    bool verbose = false;
    bool headless = true;            // Always true in headless mode
    bool reportMetrics = false;      // Report detailed metrics at the end
    unsigned int randomSeed = 12345; // Random seed for reproducibility
    bool dynamicReordering = true;   // Use dynamic reordering for Barnes-Hut SFC
    int metricsWindowSize = 10;      // Window size for dynamic reordering metrics
    bool benchmark = false;
};

// Parse command-line arguments
SimulationConfig parseArgs(int argc, char **argv);

// Configure GPU parameters based on detected hardware
void configureGPUParameters(); 