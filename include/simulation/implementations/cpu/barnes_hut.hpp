#ifndef CPU_BARNES_HUT_CUH
#define CPU_BARNES_HUT_CUH

#include "../../../common/constants.cuh"
#include "../../base/base.cuh"
#include <omp.h>
#include <vector>
#include <memory>

// Forward declaration for the CPU octree node
struct CPUOctreeNode;

class CPUBarnesHut : public SimulationBase
{
private:
    bool useOpenMP; // Flag to enable/disable OpenMP parallelization
    int numThreads; // Number of OpenMP threads to use

    // SFC-related parameters
    bool useSFC;                  // Flag to enable/disable SFC
    SFCOrderingMode orderingMode; // Ordering mode for SFC
    int reorderFrequency;         // How often to reorder (in iterations)
    int iterationCounter;         // Iterations since last reordering

    // Root of the octree
    std::unique_ptr<CPUOctreeNode> root;

    // Temporary variables for tree construction
    Vector minBound; // Minimum bounds of the domain
    Vector maxBound; // Maximum bounds of the domain

    void computeBoundingBox();
    void buildOctree();
    void computeCenterOfMass(CPUOctreeNode *node);
    void computeForces();
    void computeForceFromNode(Body &body, const CPUOctreeNode *node);

public:
    CPUBarnesHut(
        int numBodies,
        bool useParallelization = true,
        int threads = 0,
        BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
        unsigned int seed = static_cast<unsigned int>(time(nullptr)),
        MassDistribution massDist = MassDistribution::UNIFORM,
        bool enableSFC = false,
        SFCOrderingMode sfcOrderingMode = SFCOrderingMode::PARTICLES,
        int reorderFreq = 10);

    virtual ~CPUBarnesHut();
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

// Definition of CPU Octree Node
struct CPUOctreeNode
{
    Vector center;    // Center of this node's region
    double halfWidth; // Half width of this node's region

    bool isLeaf;   // Whether this is a leaf node
    int bodyIndex; // Index of the body if this is a leaf

    Vector centerOfMass; // Center of mass for this node and children
    double totalMass;    // Total mass for this node and children

    std::vector<int> bodies;    // Bodies contained in this node (if not leaf)
    CPUOctreeNode *children[8]; // Child octants

    // Constructor
    CPUOctreeNode() : center(),
                      halfWidth(0.0),
                      isLeaf(true),
                      bodyIndex(-1),
                      centerOfMass(),
                      totalMass(0.0)
    {
        for (int i = 0; i < 8; i++)
        {
            children[i] = nullptr;
        }
    }

    // Destructor - recursively destroys the octree
    ~CPUOctreeNode()
    {
        for (int i = 0; i < 8; i++)
        {
            if (children[i])
            {
                delete children[i];
                children[i] = nullptr;
            }
        }
    }

    // Determine which octant a position falls into
    int getOctant(const Vector &pos) const
    {
        int oct = 0;
        if (pos.x >= center.x)
            oct |= 1;
        if (pos.y >= center.y)
            oct |= 2;
        if (pos.z >= center.z)
            oct |= 4;
        return oct;
    }

    // Get the center position for a specific octant
    Vector getOctantCenter(int octant) const
    {
        Vector oct_center = center;
        double offset = halfWidth * 0.5;

        if (octant & 1)
            oct_center.x += offset;
        else
            oct_center.x -= offset;

        if (octant & 2)
            oct_center.y += offset;
        else
            oct_center.y -= offset;

        if (octant & 4)
            oct_center.z += offset;
        else
            oct_center.z -= offset;

        return oct_center;
    }

    // Check if a position is within this node's bounds
    bool contains(const Vector &pos) const
    {
        return (fabs(pos.x - center.x) <= halfWidth &&
                fabs(pos.y - center.y) <= halfWidth &&
                fabs(pos.z - center.z) <= halfWidth);
    }
};

#endif // CPU_BARNES_HUT_CUH