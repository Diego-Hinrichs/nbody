#ifndef SFC_VARIANTS_CUH
#define SFC_VARIANTS_CUH

#include "barnes_hut.cuh"
#include "direct_sum.cuh"
#include "../../../ui/simulation_state.hpp"

// Forward declarations
namespace sfc
{
    class BodySorter;
    class OctantSorter;
    enum class CurveType;
}

class SFCGPUDirectSum : public GPUDirectSum
{
private:
    bool useSFC;              // Flag to enable/disable SFC ordering
    Vector minBound;          // Minimum domain bounds
    Vector maxBound;          // Maximum domain bounds
    sfc::BodySorter *sorter;  // Body sorter for SFC ordering
    int *d_orderedIndices;    // Device array for SFC-ordered indices
    sfc::CurveType curveType; // Type of SFC (Morton or Hilbert)

    // Reordering parameters
    int reorderFrequency; // How often to reorder (in iterations)
    int iterationCounter; // Counts iterations between reordering

    /**
     * @brief Update domain bounds
     */
    void updateBoundingBox();

    /**
     * @brief Order bodies according to their SFC position
     */
    void orderBodiesBySFC();

protected:
    /**
     * @brief Compute forces with SFC ordering
     */
    virtual void computeForces() override;

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param useSpaceFillingCurve Flag to enable/disable SFC ordering
     * @param initialReorderFreq Initial reordering frequency
     * @param dist Distribution type for initial body positions
     * @param seed Random seed for reproducibility
     */
    SFCGPUDirectSum(int numBodies,
                    bool useSpaceFillingCurve = true,
                    int initialReorderFreq = 10,
                    BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                    unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    /**
     * @brief Destructor
     */
    virtual ~SFCGPUDirectSum();

    /**
     * @brief Update the simulation
     *
     * Extends the Direct Sum update method by applying SFC ordering
     * before computing forces.
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
     * @brief Set the curve type (Morton or Hilbert)
     * @param type New curve type
     */
    void setCurveType(sfc::CurveType type);

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
    int *getOrderedIndices() const
    {
        return d_orderedIndices;
    }
};

#endif // SFC_BARNES_HUT_CUH