#pragma once

#include <string>
#include <fstream>
#include "../simulation/simulation_thread.hpp"  // Incluir para obtener SimulationData

// Global variables for CSV management
extern std::string globalCsvFilename;
extern std::ofstream globalOutputFile;

// Initialize the global CSV file for metrics reporting
void initializeGlobalCsv();

// Report simulation metrics
void reportMetrics(SimulationData &simData, double totalSimTime, int iterations, const std::string &simulationId = ""); 