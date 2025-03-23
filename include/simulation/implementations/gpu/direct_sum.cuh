#ifndef DIRECT_SUM_CUH
#define DIRECT_SUM_CUH

#include "../../../common/types.cuh"
#include "../../../common/constants.cuh"
#include "../../../common/error_handling.cuh"
#include "../../base/base.cuh"

class GPUDirectSum : public SimulationBase
{
protected:
    virtual void computeForces();

public:
    GPUDirectSum(
        int numBodies,
        BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution masDist = MassDistribution::UNIFORM);

    virtual ~GPUDirectSum();
    virtual void update() override;
};

#endif // GPU_DIRECT_SUM_CUH