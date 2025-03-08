#ifndef CPU_BARNES_HUT_CUH
#define CPU_BARNES_HUT_CUH

#include "../common/types.cuh"
#include "../common/constants.cuh"
#include "../simulation/simulation_base.cuh"
#include <omp.h>
#include <vector>
#include <memory>

// Forward declaration for the CPU octree node
struct CPUOctreeNode;

/**
 * @brief CPU-based Barnes-Hut N-body simulation implementation
 *
 * This class implements the Barnes-Hut algorithm for N-body gravitational
 * interactions on the CPU, using an octree for space partitioning and optionally
 * using OpenMP for parallelization.
 */
class CPUBarnesHut : public SimulationBase
{
private:
    bool useOpenMP; // Flag to enable/disable OpenMP parallelization
    int numThreads; // Number of OpenMP threads to use

    // Root of the octree
    std::unique_ptr<CPUOctreeNode> root;

    // Temporary variables for tree construction
    Vector minBound; // Minimum bounds of the domain
    Vector maxBound; // Maximum bounds of the domain

    /**
     * @brief Compute the bounding box for the entire simulation
     */
    void computeBoundingBox();

    /**
     * @brief Build the octree for Barnes-Hut simulation
     */
    void buildOctree();

    /**
     * @brief Compute the center of mass for a node and its children
     * @param node The octree node
     */
    void computeCenterOfMass(CPUOctreeNode *node);

    /**
     * @brief Compute forces between bodies using the Barnes-Hut approximation
     */
    void computeForces();

    /**
     * @brief Compute force on a body from a node
     * @param body The body to compute force for
     * @param node The octree node
     */
    void computeForceFromNode(Body &body, const CPUOctreeNode *node);

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param useParallelization Flag to enable/disable OpenMP
     * @param threads Number of OpenMP threads (0 for auto-detect)
     * @param dist Distribution type for initial body positions
     * @param seed Random seed for reproducibility
     */
    CPUBarnesHut(int numBodies,
                 bool useParallelization = true,
                 int threads = 0,
                 BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                 unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    /**
     * @brief Destructor
     */
    virtual ~CPUBarnesHut();

    /**
     * @brief Update the simulation
     *
     * Performs one simulation step by computing forces and updating positions.
     */
    virtual void update() override;

    /**
     * @brief Enable or disable OpenMP parallelization
     * @param enable Flag to enable/disable OpenMP parallelization
     */
    void enableOpenMP(bool enable)
    {
        useOpenMP = enable;
    }

    /**
     * @brief Set the number of OpenMP threads
     * @param threads Number of threads (0 for auto-detect)
     */
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