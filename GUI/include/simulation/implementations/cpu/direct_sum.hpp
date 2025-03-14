#ifndef CPU_DIRECT_SUM_CUH
#define CPU_DIRECT_SUM_CUH

#include "../../../common/constants.cuh"
#include "../../base/base.cuh"
#include <omp.h>

/**
 * @brief CPU-based Direct Sum N-body simulation implementation
 *
 * This class implements the Direct Sum algorithm for N-body gravitational
 * interactions on the CPU, optionally using OpenMP for parallelization.
 */
class CPUDirectSum : public SimulationBase
{
protected:
    bool useOpenMP; // Flag to enable/disable OpenMP parallelization
    int numThreads; // Number of OpenMP threads to use

    /**
     * @brief Compute forces between bodies using direct summation
     */
    virtual void computeForces();

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param useParallelization Flag to enable/disable OpenMP
     * @param threads Number of OpenMP threads (0 for auto-detect)
     * @param dist Distribution type for initial body positions
     * @param seed Random seed for reproducibility
     */
    CPUDirectSum(int numBodies,
                 bool useParallelization = true,
                 int threads = 0,
                 BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                 unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    /**
     * @brief Destructor
     */
    virtual ~CPUDirectSum();

    /**
     * @brief Update the simulation
     *
     * Performs one simulation step by computing forces and updating positions.
     */
    virtual void update() override;

    /**
     * @brief Enable or disable OpenMP parallelization
     * @param enable Flag to enable/disable OpenMP parallelization
     */
    void enableOpenMP(bool enable)
    {
        useOpenMP = enable;
    }

    /**
     * @brief Set the number of OpenMP threads
     * @param threads Number of threads (0 for auto-detect)
     */
    void setThreadCount(int threads)
    {
        if (threads <= 0)
        {
            // Auto-detect number of available threads
            numThreads = omp_get_max_threads();
        }
        else
        {
            numThreads = threads;
        }
    }
};

#endif // CPU_DIRECT_SUM_CUH