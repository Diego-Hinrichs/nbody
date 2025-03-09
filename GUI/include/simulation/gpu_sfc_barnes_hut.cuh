#ifndef SFC_BARNES_HUT_CUH
#define SFC_BARNES_HUT_CUH

#include "../../include/simulation/gpu_barnes_hut.cuh"
#include "../sfc/body_sorter.cuh"
#include "../ui/simulation_state.hpp"

/**
 * @brief Space-Filling Curve enhanced Barnes-Hut simulation
 *
 * This class extends the standard Barnes-Hut implementation by sorting
 * bodies or octants according to their position on a space-filling curve (Morton code),
 * which improves memory coherence during tree traversal.
 */
class SFCBarnesHut : public BarnesHut
{
private:
    bool useSFC;             // Flag to enable/disable SFC ordering
    Vector minBound;         // Minimum domain bounds
    Vector maxBound;         // Maximum domain bounds
    sfc::BodySorter *sorter; // Body sorter for SFC ordering
    int *d_orderedIndices;   // Device array for SFC-ordered indices
    int *d_octantIndices;    // Device array for octant ordering

    // Reordering parameters
    SFCOrderingMode orderingMode; // Current ordering mode
    int reorderFrequency;         // How often to reorder (in iterations)
    int iterationCounter;         // Counts iterations between reordering

    /**
     * @brief Update domain bounds based on root node
     */
    void updateBoundingBox();

    /**
     * @brief Order bodies according to their Morton codes
     */
    void orderBodiesBySFC();

    /**
     * @brief Order octants according to their Morton codes
     * @param nodes Device pointer to octree nodes
     * @param nNodes Number of nodes
     */
    void orderOctantsBySFC(Node *nodes, int nNodes);

protected:
    /**
     * @brief Override to construct the octree with SFC ordering
     */
    virtual void constructOctree() override;

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param useSpaceFillingCurve Flag to enable/disable SFC ordering
     * @param initialOrderingMode Initial ordering mode
     * @param initialReorderFreq Initial reordering frequency
     * @param dist Distribution type for initial body positions
     * @param seed Random seed for reproducibility
     */
    SFCBarnesHut(int numBodies,
                 bool useSpaceFillingCurve = true,
                 SFCOrderingMode initialOrderingMode = SFCOrderingMode::PARTICLES,
                 int initialReorderFreq = 10,
                 BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                 unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    /**
     * @brief Destructor
     */
    virtual ~SFCBarnesHut();

    /**
     * @brief Update the simulation
     *
     * Extends the Barnes-Hut update method by applying SFC ordering
     * before constructing the octree.
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
     * @brief Set the ordering mode
     * @param mode New ordering mode
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

    /**
     * @brief Check if SFC ordering is enabled
     * @return true if SFC ordering is enabled, false otherwise
     */
    bool isSFCEnabled() const
    {
        return useSFC;
    }

    /**
     * @brief Get the SFC-ordered indices array
     * @return Pointer to the device array of SFC-ordered indices
     */
    int *getOrderedIndices() const override
    {
        return (orderingMode == SFCOrderingMode::PARTICLES) ? d_orderedIndices : d_octantIndices;
    }

    /**
     * @brief Check if SFC ordering is being used
     * @return True if SFC ordering is enabled and indices are valid
     */
    bool isUsingSFC() const override
    {
        return useSFC && getOrderedIndices() != nullptr;
    }
};

#endif // SFC_BARNES_HUT_CUH
