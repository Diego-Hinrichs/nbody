#ifndef CPU_SFC_DIRECT_SUM_HPP
#define CPU_SFC_DIRECT_SUM_HPP

#include "direct_sum.hpp"
#include "../../../common/constants.cuh"
#include "../../../sfc/sfc_framework.cuh"
#include <omp.h>
#include <vector>
#include <algorithm>

class SFCCPUDirectSum : public CPUDirectSum
{
private:
    bool useSFC;          // Flag to enable/disable SFC
    int reorderFrequency; // How often to reorder (in iterations)
    int iterationCounter; // Iterations since last reordering

    // Morton code ordering data
    std::vector<uint64_t> mortonCodes; // Morton codes for each body
    std::vector<int> orderedIndices;   // Ordered indices

    // Bounding box data
    Vector minBound; // Minimum bounds of the domain
    Vector maxBound; // Maximum bounds of the domain

    void computeBoundingBox();
    void orderBodiesBySFC();

protected:
    virtual void computeForces() override;

public:
    SFCCPUDirectSum(
        int numBodies,
        bool useParallelization = true,
        int threads = 0,
        bool enableSFC = true,
        int reorderFreq = 10,
        BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM);

    virtual ~SFCCPUDirectSum();
    virtual void update() override;
    void enableSFC(bool enable) { useSFC = enable; }
    void setReorderFrequency(int frequency) { reorderFrequency = frequency; }
};

#endif // SFC_CPU_DIRECT_SUM_HPP