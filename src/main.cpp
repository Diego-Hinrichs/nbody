#include <iostream>
#include <thread>
#include <chrono>
#include <memory>
#include <vector>
#include <string>
#include <map>

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

#include "../include/utils/config.hpp"
#include "../include/utils/metrics.hpp"
#include "../include/utils/simulation_runner.hpp"

// Global simulation state for callbacks
SimulationState *g_simulationState = nullptr;

// External global variables from the config module
extern int g_blockSize;

// Función para obtener el nombre del método a partir del valor numérico
std::string getMethodName(int method) {
    std::map<int, std::string> methodNames = {
        {0, "CPU Direct Sum"},
        {1, "CPU SFC Direct Sum"},
        {2, "GPU Direct Sum"},
        {3, "GPU SFC Direct Sum"},
        {4, "CPU Barnes-Hut"},
        {5, "CPU SFC Barnes-Hut"},
        {6, "GPU Barnes-Hut"},
        {7, "GPU SFC Barnes-Hut"}
    };

    auto it = methodNames.find(method);
    if (it != methodNames.end()) {
        return it->second;
    }
    return "Unknown Method";
}

// Función para determinar si un método usa SFC
bool methodUsesSFC(int method) {
    return (method == 1 || method == 3 || method == 5 || method == 7);
}

// Función para determinar si un método es GPU
bool isGPUMethod(int method) {
    return (method == 2 || method == 3 || method == 6 || method == 7);
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

        // Todos los métodos disponibles:
        // 0: CPU_DIRECT_SUM
        // 1: CPU_SFC_DIRECT_SUM
        // 2: GPU_DIRECT_SUM
        // 3: GPU_SFC_DIRECT_SUM
        // 4: CPU_BARNES_HUT
        // 5: CPU_SFC_BARNES_HUT
        // 6: GPU_BARNES_HUT
        // 7: GPU_SFC_BARNES_HUT
        std::vector<int> methods = {0, 1, 2, 3, 4, 5, 6, 7}; // Todos los métodos
        std::vector<int> bodyCounts = {1000, 5000, 10000, 50000}; // Varios tamaños de problema

        for (int method : methods)
        {
            if (isGPUMethod(method))
            {
                configureGPUParameters();
            }
            std::cout << "\nTesting method " << method << ": " << getMethodName(method) << std::endl;
            bool useSFC = methodUsesSFC(method);

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
                    benchConfig.blockSize = g_blockSize;
                    benchConfig.useSFC = useSFC;
                    benchConfig.sortType = useSFC ? 2 : 0; // Morton by default
                    benchConfig.visualization = false;
                    benchConfig.headless = true;
                    benchConfig.reportMetrics = true;
                    benchConfig.numSteps = BENCHMARK_STEPS;

                    try
                    {
                        std::cout << "Starting benchmark: method=" << getMethodName(method)
                                  << " bodies=" << bodies
                                  << " run=" << (i + 1) << std::endl;

                        runSimulationOnce(benchConfig);

                        std::cout << "Benchmark completed successfully" << std::endl;

                        // Add a small delay between runs
                        std::this_thread::sleep_for(std::chrono::seconds(2));
                    }
                    catch (const std::exception &e)
                    {
                        std::cerr << "Error in benchmark run: " << e.what() << std::endl;
                        continue;
                    }
                }
            }
        }

        std::cout << "Benchmark completed." << std::endl;
    }
    else
    {
        // If not in benchmark mode, run a single simulation
        runSimulationOnce(config);
    }

    // Close the global CSV file if it's open
    if (globalOutputFile.is_open())
    {
        std::cout << "Closing global CSV file: " << globalCsvFilename << std::endl;
        globalOutputFile.close();
    }

    return 0;
}
