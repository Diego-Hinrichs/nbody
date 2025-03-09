#include "../common/types.cuh"
#include "../common/constants.cuh"
#include "../sfc/body_sorter.cuh"
#include <cstdint>

__global__ void ComputeHilbertCodesKernel(Body *bodies, uint64_t *hilbertCodes, int *indices,
                                          int nBodies, Vector minBound, Vector maxBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBodies)
    {
        // Use node center for Hilbert code calculation
        Vector position = bodies[idx].position;
        hilbertCodes[idx] = sfc::positionToHilbert(position, minBound, maxBound);

        // Initialize indices sequentially
        indices[idx] = idx;
    }
}

__global__ void ComputeOctantHilbertCodesKernel(Node *nodes, uint64_t *hilbertCodes, int *indices,
                                                int nNodes, Vector minBound, Vector maxBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodes)
    {
        // Skip empty nodes
        if (nodes[idx].start == -1 && nodes[idx].end == -1)
        {
            hilbertCodes[idx] = 0; // Assign lowest priority to empty nodes
        }
        else
        {
            // Use node center for Hilbert code calculation
            Vector center = nodes[idx].getCenter();
            hilbertCodes[idx] = sfc::positionToHilbert(center, minBound, maxBound);
        }
        indices[idx] = idx; // Initialize with sequential indices
    }
}