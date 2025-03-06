#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

/**
 * @brief Reset the octree nodes to their initial state
 *
 * @param nodes Array of octree nodes
 * @param mutex Mutex for synchronization
 * @param nNodes Number of nodes in the array
 * @param nBodies Number of bodies in the simulation
 */
__global__ void ResetKernel(Node *nodes, int *mutex, int nNodes, int nBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nNodes)
    {
        // Reset node to initial values
        nodes[idx].topLeftFront = Vector(INFINITY, INFINITY, INFINITY);
        nodes[idx].botRightBack = Vector(-INFINITY, -INFINITY, -INFINITY);
        nodes[idx].centerMass = Vector(0.0, 0.0, 0.0);
        nodes[idx].totalMass = 0.0;
        nodes[idx].isLeaf = true;
        nodes[idx].start = -1;
        nodes[idx].end = -1;
        mutex[idx] = 0;
    }

    // First thread initializes the root node to include all bodies
    if (idx == 0)
    {
        nodes[0].start = 0;
        nodes[0].end = nBodies - 1;
    }
}