
#ifndef BARNES_HUT_CUH
#define BARNES_HUT_CUH

#include "../../base/base.cuh"
#include <cstring>
class BarnesHut : public SimulationBase
{
protected:
    int nNodes;    // Total number of nodes in the octree
    int leafLimit; // Leaf node threshold

    Node *h_nodes; // Host nodes array
    Node *d_nodes; // Device nodes array
    int *d_mutex;  // Device mutex for synchronization

    Body *d_bodiesBuffer; // Temporary buffer for octree construction

    virtual void resetOctree();
    virtual void computeBoundingBox();
    virtual void constructOctree();
    virtual void computeForces();

public:
    BarnesHut(int numBodies,
              BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
              unsigned int seed = static_cast<unsigned int>(time(nullptr)));
    virtual ~BarnesHut();

    bool getOctreeNodes(Node* destNodes, int maxNodes) {
        if (!d_nodes || !h_nodes || nNodes <= 0)
            return false;
            
        try {
            cudaMemcpy(h_nodes, d_nodes, nNodes * sizeof(Node), cudaMemcpyDeviceToHost);
            int nodesToCopy = std::min(maxNodes, nNodes);
            memcpy(destNodes, h_nodes, nodesToCopy * sizeof(Node));
            return true;
        } catch(...) {
            return false;
        }
    }
    
    int getNumNodes() const {
        return nNodes;
    }
    
    int getRootNodeIndex() const {
        return 0;
    }

    virtual void update() override;
};

#endif // BARNES_HUT_CUH