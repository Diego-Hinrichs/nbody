#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

__global__ void ResetKernel(Node *node, int *mutex, int nNodes, int nBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nNodes)
    {
        // Resetear nodo a valores iniciales
        node[idx].topLeftFront = {INFINITY, INFINITY, INFINITY};
        node[idx].botRightBack = {-INFINITY, -INFINITY, -INFINITY};
        node[idx].centerMass = {0.0, 0.0, 0.0};
        node[idx].totalMass = 0.0;
        node[idx].isLeaf = true;
        node[idx].start = -1;
        node[idx].end = -1;
        mutex[idx] = 0;
    }
    // El primer thread inicializa el nodo raíz
    if (idx == 0)
    {
        node[0].start = 0;
        node[0].end = nBodies - 1;
    }
}
