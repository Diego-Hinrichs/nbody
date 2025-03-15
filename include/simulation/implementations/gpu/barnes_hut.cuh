
#ifndef BARNES_HUT_CUH
#define BARNES_HUT_CUH

#include "../../base/base.cuh"

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

    // Nuevos métodos para manejar índices SFC (definidos con valores por defecto)
    virtual int *getOrderedIndices() const { return nullptr; }
    virtual bool isUsingSFC() const { return false; }

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
     * Made virtual to allow derived classes to override it
     */
    virtual void constructOctree();

    /**
     * @brief Compute gravitational forces using the octree
     */
    void computeForces();

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param dist Distribution type for initial body positions
     * @param seed Random seed for reproducibility
     */
    BarnesHut(int numBodies,
              BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
              unsigned int seed = static_cast<unsigned int>(time(nullptr)));

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