#ifndef BARNES_HUT_CUH
#define BARNES_HUT_CUH

#include "simulation_base.cuh"

// Forward declarations of kernel functions
__global__ void ResetKernel(Node *nodes, int *mutex, int nNodes, int nBodies);
__global__ void ComputeBoundingBoxKernel(Node *nodes, Body *bodies, int *mutex, int nBodies);
__global__ void ConstructOctTreeKernel(Node *nodes, Body *bodies, Body *bodiesBuffer,
                                       int nodeIndex, int nNodes, int nBodies, int leafLimit);
__global__ void ComputeForceKernel(Node *nodes, Body *bodies, int nNodes, int nBodies, int leafLimit);

/**
 * @brief Barnes-Hut N-body simulation implementation
 *
 * This class implements the Barnes-Hut algorithm for approximating
 * N-body gravitational interactions using an octree structure.
 */
class BarnesHut : public SimulationBase
{
protected:
    int nNodes;    // Total number of nodes in the octree
    int leafLimit; // Leaf node threshold

    Node *h_nodes; // Host nodes array
    Node *d_nodes; // Device nodes array
    int *d_mutex;  // Device mutex for synchronization

    Body *d_bodiesBuffer; // Temporary buffer for octree construction

    // Kernel wrapper methods
    /**
     * @brief Reset the octree structure
     */
    void resetOctree();

    /**
     * @brief Compute the bounding box for the simulation domain
     */
    void computeBoundingBox();

    /**
     * @brief Construct the octree from current body positions
     */
    void constructOctree();

    /**
     * @brief Compute gravitational forces using the octree
     */
    void computeForces();

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     */
    BarnesHut(int numBodies);

    /**
     * @brief Destructor
     */
    virtual ~BarnesHut();

    /**
     * @brief Update the simulation
     *
     * Performs one simulation step:
     * 1. Reset octree
     * 2. Compute bounding box
     * 3. Construct octree
     * 4. Compute forces and update positions
     */
    virtual void update() override;
};

#endif // BARNES_HUT_CUH