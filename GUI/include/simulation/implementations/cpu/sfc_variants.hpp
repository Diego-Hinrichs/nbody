#ifndef CPU_SFC_DIRECT_SUM_HPP
#define CPU_SFC_DIRECT_SUM_HPP

#include "direct_sum.hpp"
#include "barnes_hut.hpp"
#include "../../../common/constants.cuh"
#include "../../../sfc/sfc_framework.cuh"
#include <omp.h>
#include <vector>
#include <algorithm>

/**
 * @brief GPU-based Direct Sum N-body simulation implementation
 *
 * This class implements the Direct Sum algorithm for N-body gravitational
 * interactions on the GPU using CUDA.
 */
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

    /**
     * @brief Compute the bounding box for SFC calculations
     */
    void computeBoundingBox();

    /**
     * @brief Order bodies by their Morton codes
     */
    void orderBodiesBySFC();

protected:
    /**
     * @brief Override force computation to use SFC ordering
     */
    virtual void computeForces() override;

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param useParallelization Flag to enable/disable OpenMP
     * @param threads Number of OpenMP threads (0 for auto-detect)
     * @param enableSFC Flag to enable/disable Space-Filling Curve ordering
     * @param reorderFreq How often to reorder (in iterations)
     * @param dist Distribution type for initial body positions
     * @param seed Random seed for reproducibility
     */
    SFCCPUDirectSum(int numBodies,
                    bool useParallelization = true,
                    int threads = 0,
                    bool enableSFC = true,
                    int reorderFreq = 10,
                    BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                    unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    /**
     * @brief Destructor
     */
    virtual ~SFCCPUDirectSum();

    /**
     * @brief Update the simulation
     *
     * Extends the direct sum update method by applying SFC ordering.
     */
    virtual void update() override;

    /**
     * @brief Enable or disable SFC ordering
     * @param enable Flag to enable/disable SFC ordering
     */
    void enableSFC(bool enable)
    {
        useSFC = enable;
    }

    /**
     * @brief Set the reordering frequency
     * @param frequency New reordering frequency in iterations
     */
    void setReorderFrequency(int frequency)
    {
        reorderFrequency = frequency;
    }
};

/**
 * @brief CPU-based Barnes-Hut N-body simulation with Space-Filling Curve ordering
 *
 * This class extends the CPU Barnes-Hut algorithm with Space-Filling Curve
 * ordering optimization to improve cache locality and performance.
 */
class SFCCPUBarnesHut : public CPUBarnesHut
{
private:
    // Morton code ordering data
    std::vector<uint64_t> mortonCodes; // Morton codes for each body
    std::vector<int> orderedIndices;   // Ordered indices

    /**
     * @brief Order bodies by their Morton codes
     */
    void orderBodiesBySFC();

    /**
     * @brief Reorder octants in the tree based on SFC
     * @param node The current node to process
     */
    void reorderOctants(CPUOctreeNode *node);

    /**
     * @brief Override buildOctree to use SFC ordering
     */
    virtual void buildOctree();

    /**
     * @brief Override computeForces to use SFC ordering
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
     * @param enableSFC Flag to enable/disable Space-Filling Curve ordering
     * @param sfcOrderingMode Ordering mode for SFC (particles or octants)
     * @param reorderFreq How often to reorder (in iterations)
     */
    SFCCPUBarnesHut(int numBodies,
                    bool useParallelization = true,
                    int threads = 0,
                    BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                    unsigned int seed = static_cast<unsigned int>(time(nullptr)),
                    bool enableSFC = true,
                    SFCOrderingMode sfcOrderingMode = SFCOrderingMode::PARTICLES,
                    int reorderFreq = 10);

    /**
     * @brief Destructor
     */
    virtual ~SFCCPUBarnesHut();

    /**
     * @brief Update the simulation
     *
     * Extends the Barnes-Hut update method by applying SFC ordering.
     */
    virtual void update() override;

    /**
     * @brief Enable or disable SFC ordering
     * @param enable Flag to enable/disable SFC ordering
     */
    void enableSFC(bool enable)
    {
        useSFC = enable;
    }

    /**
     * @brief Set the SFC ordering mode
     * @param mode The ordering mode to use (particles or octants)
     */
    void setOrderingMode(SFCOrderingMode mode)
    {
        orderingMode = mode;
    }

    /**
     * @brief Set the reordering frequency
     * @param frequency New reordering frequency in iterations
     */
    void setReorderFrequency(int frequency)
    {
        reorderFrequency = frequency;
    }
};

#endif // SFC_CPU_DIRECT_SUM_HPP