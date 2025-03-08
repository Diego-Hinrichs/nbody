#ifndef GPU_DIRECT_SUM_CUH
#define GPU_DIRECT_SUM_CUH

#include "../common/types.cuh"
#include "../common/constants.cuh"
#include "../common/error_handling.cuh"
#include "../simulation/simulation_base.cuh"

// Forward declaration of CUDA kernel
__global__ void DirectSumForceKernel(Body *bodies, int nBodies);

/**
 * @brief GPU-based Direct Sum N-body simulation implementation
 *
 * This class implements the Direct Sum algorithm for N-body gravitational
 * interactions on the GPU using CUDA.
 */
class GPUDirectSum : public SimulationBase
{
private:
    /**
     * @brief Compute forces between bodies using direct summation on GPU
     */
    void computeForces();

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param dist Distribution type for initial body positions
     * @param seed Random seed for reproducibility
     */
    GPUDirectSum(int numBodies,
                 BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                 unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    /**
     * @brief Destructor
     */
    virtual ~GPUDirectSum();

    /**
     * @brief Update the simulation
     *
     * Performs one simulation step by computing forces and updating positions.
     */
    virtual void update() override;
};

#endif // GPU_DIRECT_SUM_CUH