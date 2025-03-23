#ifndef CPU_DIRECT_SUM_CUH
#define CPU_DIRECT_SUM_CUH

#include "../../../common/constants.cuh"
#include "../../base/base.cuh"
#include <omp.h>

class CPUDirectSum : public SimulationBase
{
protected:
    bool useOpenMP; // Flag to enable/disable OpenMP parallelization
    int numThreads; // Number of OpenMP threads to use
    virtual void computeForces();

public:
    CPUDirectSum(
        int numBodies,
        bool useParallelization = true,
        int threads = 0,
        BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM
    );

    virtual ~CPUDirectSum();
    virtual void update() override;
    void enableOpenMP(bool enable) { useOpenMP = enable; }

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