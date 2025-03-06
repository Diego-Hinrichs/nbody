#ifndef SFC_BARNES_HUT_CUH
#define SFC_BARNES_HUT_CUH

#include "barnes_hut.cuh"
#include "../sfc/body_sorter.cuh"

/**
 * @brief Space-Filling Curve enhanced Barnes-Hut simulation
 *
 * This class extends the standard Barnes-Hut implementation by sorting
 * bodies according to their position on a space-filling curve (Morton code),
 * which improves memory coherence during tree traversal.
 */
class SFCBarnesHut : public BarnesHut
{
private:
    bool useSFC;             // Flag to enable/disable SFC ordering
    Vector minBound;         // Minimum domain bounds
    Vector maxBound;         // Maximum domain bounds
    sfc::BodySorter *sorter; // Body sorter for SFC ordering

    /**
     * @brief Update domain bounds based on root node
     */
    void updateBoundingBox();

    /**
     * @brief Order bodies according to their Morton codes
     */
    void orderBodiesBySFC();

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param useSpaceFillingCurve Flag to enable/disable SFC ordering
     */
    SFCBarnesHut(int numBodies, bool useSpaceFillingCurve = true);

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
     * @brief Check if SFC ordering is enabled
     * @return true if SFC ordering is enabled, false otherwise
     */
    bool isSFCEnabled() const
    {
        return useSFC;
    }
};

#endif // SFC_BARNES_HUT_CUH