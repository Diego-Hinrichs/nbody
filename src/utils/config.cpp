#include "../../include/utils/config.hpp"
#include <iostream>
#include <string>
#include <cuda_runtime.h>

// Define the global variables
double g_theta = 0.5;  // Default theta value
int g_blockSize = 256; // Default block size

void configureGPUParameters()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        return;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    std::string gpuName = deviceProp.name;
    int computeCapability = deviceProp.major * 10 + deviceProp.minor;
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    int multiProcessorCount = deviceProp.multiProcessorCount;

    // Log the detected GPU information
    std::cout << "Detected GPU: " << gpuName << std::endl;
    std::cout << "Compute capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
    std::cout << "Number of SMs: " << multiProcessorCount << std::endl;

    if (computeCapability >= 89)
    {                      // RTX 40 series (Ada Lovelace - SM 8.9)
        g_blockSize = 512; // Larger blocks for Ada Lovelace
    }
    else if (computeCapability >= 86)
    { // RTX 30 series (Ampere - SM 8.6)
        g_blockSize = 384;
    }
    else if (computeCapability >= 75)
    { // RTX 20 series (Turing - SM 7.5)
        g_blockSize = 256;
    }
    else
    {
        // For older GPUs, stick with the default 256
        g_blockSize = 256;
    }

    // Ensure block size is within device limits and is a multiple of 32
    g_blockSize = std::min(g_blockSize, maxThreadsPerBlock);
    g_blockSize = (g_blockSize / 32) * 32; // Round to multiple of warp size

    std::cout << "Configured block size: " << g_blockSize << std::endl;
}

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