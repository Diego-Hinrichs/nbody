#include "../../include/sfc/body_sorter.cuh"

__global__ void ComputeMortonCodesKernel(Body *bodies, uint64_t *mortonCodes, int *indices,
                                         int nBodies, Vector minBound, Vector maxBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBodies)
    {
        mortonCodes[idx] = sfc::positionToMorton(bodies[idx].position, minBound, maxBound);
        indices[idx] = idx;
    }
}

__global__ void ReorderBodiesKernel(Body *bodies, Body *sortedBodies, int *indices, int nBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBodies)
    {
        sortedBodies[idx] = bodies[indices[idx]];
    }
}

namespace sfc
{

    BodySorter::BodySorter(int numBodies) : nBodies(numBodies)
    {
        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_mortonCodes, nBodies * sizeof(uint64_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_indices, nBodies * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_tempBodies, nBodies * sizeof(Body)));
    }

    BodySorter::~BodySorter()
    {
        // Free device memory
        if (d_mortonCodes)
        {
            CHECK_CUDA_ERROR(cudaFree(d_mortonCodes));
            d_mortonCodes = nullptr;
        }

        if (d_indices)
        {
            CHECK_CUDA_ERROR(cudaFree(d_indices));
            d_indices = nullptr;
        }

        if (d_tempBodies)
        {
            CHECK_CUDA_ERROR(cudaFree(d_tempBodies));
            d_tempBodies = nullptr;
        }
    }

    void BodySorter::sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound)
    {
        // Calculate launch configuration
        int blockSize = BLOCK_SIZE;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        // Compute Morton codes for all bodies
        ComputeMortonCodesKernel<<<gridSize, blockSize>>>(
            d_bodies, d_mortonCodes, d_indices, nBodies, minBound, maxBound);
        CHECK_LAST_CUDA_ERROR();

        // Sort bodies by Morton code using Thrust
        thrust::device_ptr<uint64_t> thrust_codes(d_mortonCodes);
        thrust::device_ptr<int> thrust_indices(d_indices);
        thrust::sort_by_key(thrust::device, thrust_codes, thrust_codes + nBodies, thrust_indices);

        // Reorder bodies based on sorted indices
        ReorderBodiesKernel<<<gridSize, blockSize>>>(d_bodies, d_tempBodies, d_indices, nBodies);
        CHECK_LAST_CUDA_ERROR();

        // Copy back to original array
        CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, d_tempBodies, nBodies * sizeof(Body), cudaMemcpyDeviceToDevice));
    }

} // namespace sfc