
#ifndef BARNES_HUT_CUH
#define BARNES_HUT_CUH

#include "../../base/base.cuh"

class BarnesHut : public SimulationBase
{
protected:
    int nNodes;    // Total number of nodes in the octree
    int leafLimit; // Leaf node threshold

    Node *h_nodes; // Host nodes array
    Node *d_nodes; // Device nodes array
    int *d_mutex;  // Device mutex for synchronization

    Body *d_bodiesBuffer; // Temporary buffer for octree construction

    virtual int *getOrderedIndices() const { return nullptr; }
    
    virtual bool isUsingSFC() const { return false; }

    void resetOctree();

    void computeBoundingBox();

    virtual void constructOctree();

    void computeForces();

    void swapBodyBuffers();

public:
    BarnesHut(int numBodies,
              BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
              unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    virtual ~BarnesHut();
    virtual void update() override;
};

#endif // BARNES_HUT_CUH