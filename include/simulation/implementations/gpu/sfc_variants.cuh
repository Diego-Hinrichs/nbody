#ifndef SFC_VARIANTS_CUH
#define SFC_VARIANTS_CUH

#include "barnes_hut.cuh"
#include "direct_sum.cuh"
#include "../../../ui/simulation_state.hpp"
#include "../../../sfc/sfc_framework.cuh"

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
    void updateBoundingBox();
    void orderBodiesBySFC();

protected:
    virtual void computeForces() override;

public:
    SFCGPUDirectSum(
        int numBodies,
        bool useSpaceFillingCurve = true,
        int initialReorderFreq = 10,
        BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM);

    virtual ~SFCGPUDirectSum();

    virtual void update() override;

    void enableSFC(bool enable) { useSFC = enable; }
    void setCurveType(sfc::CurveType type);
    void setReorderFrequency(int frequency) { reorderFrequency = frequency; }
    bool isSFCEnabled() const { return useSFC; }
    int *getOrderedIndices() const { return d_orderedIndices; }
};

class SFCBarnesHut : public BarnesHut
{
private:
    bool useSFC;
    sfc::CurveType curveType;
    SFCOrderingMode orderingMode;
    sfc::BodySorter *bodySorter;
    sfc::OctantSorter *octantSorter;
    int *d_orderedBodyIndices;
    int *d_orderedNodeIndices;

    Vector minBound;
    Vector maxBound;

    int reorderFrequency;
    int iterationCounter;

    void updateBoundingBox();
    void orderBodiesBySFC();
    void orderOctantsBySFC();

protected:
    virtual void resetOctree() override;
    virtual void computeBoundingBox() override;
    virtual void constructOctree() override;
    virtual void computeForces() override;

public:
    SFCBarnesHut(
        int numBodies,
        bool enableSFC = true,
        SFCOrderingMode ordMode = SFCOrderingMode::PARTICLES,
        int reorderFreq = 10,
        BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM);

    virtual ~SFCBarnesHut();

    virtual void update() override;

    void enableSFC(bool enable) { useSFC = enable; }
    void setCurveType(sfc::CurveType type);
    void setOrderingMode(SFCOrderingMode mode) { orderingMode = mode; }
    void setReorderFrequency(int frequency) { reorderFrequency = frequency; }

    bool isSFCEnabled() const { return useSFC; }
    SFCOrderingMode getOrderingMode() const { return orderingMode; }
    int getReorderFrequency() const { return reorderFrequency; }
};

#endif // SFC_BARNES_HUT_CUH