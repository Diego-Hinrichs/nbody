#ifndef SFC_MORTON_CUH
#define SFC_MORTON_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "barnes_hut.cuh"

// Expand bits (for Morton code calculation)
__host__ __device__ inline uint64_t expandBits(uint64_t v)
{
    v = (v * 0x0000000100000001ULL) & 0xFFFF00000000FFFFULL;
    v = (v * 0x0000000000010001ULL) & 0x00FF0000FF0000FFULL;
    v = (v * 0x0000000000000101ULL) & 0xF00F00F00F00F00FULL;
    v = (v * 0x0000000000000011ULL) & 0x30C30C30C30C30C3ULL;
    v = (v * 0x0000000000000005ULL) & 0x9249249249249249ULL;
    return v;
}

// Compute Morton code for 3D coordinates
__host__ __device__ inline uint64_t getMortonCode(uint64_t x, uint64_t y, uint64_t z)
{
    x = expandBits(x);
    y = expandBits(y);
    z = expandBits(z);
    return x | (y << 1) | (z << 2);
}

// Convert normalized position to Morton code
__host__ __device__ inline uint64_t positionToMorton(Vector pos, Vector minBound, Vector maxBound, int bits = 21)
{
    // Normalize coordinates to [0,1]
    double normalizedX = (pos.x - minBound.x) / (maxBound.x - minBound.x);
    double normalizedY = (pos.y - minBound.y) / (maxBound.y - minBound.y);
    double normalizedZ = (pos.z - minBound.z) / (maxBound.z - minBound.z);

    // Scale to [0, 2^bits-1]
    uint64_t maxCoord = (1ULL << bits) - 1;
    uint64_t x = (normalizedX >= 1.0) ? maxCoord : ((normalizedX <= 0.0) ? 0 : (uint64_t)(normalizedX * maxCoord));
    uint64_t y = (normalizedY >= 1.0) ? maxCoord : ((normalizedY <= 0.0) ? 0 : (uint64_t)(normalizedY * maxCoord));
    uint64_t z = (normalizedZ >= 1.0) ? maxCoord : ((normalizedZ <= 0.0) ? 0 : (uint64_t)(normalizedZ * maxCoord));

    return getMortonCode(x, y, z);
}

// Kernel to compute Morton codes for all bodies
__global__ void computeMortonCodes(Body *bodies, uint64_t *codes, int *indices, int nBodies,
                                   Vector minBound, Vector maxBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBodies)
    {
        codes[idx] = positionToMorton(bodies[idx].position, minBound, maxBound);
        indices[idx] = idx;
    }
}

// Kernel to reorder bodies based on sorted indices
__global__ void reorderBodies(Body *bodies, Body *sortedBodies, int *indices, int nBodies)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBodies)
    {
        sortedBodies[idx] = bodies[indices[idx]];
    }
}

// Class for Barnes-Hut with SFC ordering
class SFCBarnesHutCuda : public BarnesHutCuda
{
private:
    // Additional device memory for SFC ordering
    uint64_t *d_codes;    // Morton codes
    int *d_indices;       // Indices for sorting
    Body *d_sortedBodies; // Temporary storage for sorted bodies
    Vector minBound;      // Domain bounds
    Vector maxBound;
    bool useSFC; // Flag to enable/disable SFC ordering

    void updateBoundingBox()
    {
        // Copy root node to get current bounding box
        Node rootNode;
        CHECK_CUDA_ERROR(cudaMemcpy(&rootNode, d_node, sizeof(Node), cudaMemcpyDeviceToHost));
        minBound = rootNode.topLeftFront;
        maxBound = rootNode.botRightBack;
    }

    void orderBodiesBySFC()
    {
        if (!useSFC)
            return;

        int blockSize = BLOCK_SIZE;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        // Update bounding box for SFC calculation
        updateBoundingBox();

        // Compute Morton codes
        computeMortonCodes<<<gridSize, blockSize>>>(
            d_b, d_codes, d_indices, nBodies, minBound, maxBound);

        // Sort particles by Morton code
        thrust::device_ptr<uint64_t> thrust_codes(d_codes);
        thrust::device_ptr<int> thrust_indices(d_indices);
        thrust::sort_by_key(thrust::device, thrust_codes, thrust_codes + nBodies, thrust_indices);

        // Reorder bodies based on sorted indices
        reorderBodies<<<gridSize, blockSize>>>(d_b, d_sortedBodies, d_indices, nBodies);

        // Copy back to original array
        cudaMemcpy(d_b, d_sortedBodies, nBodies * sizeof(Body), cudaMemcpyDeviceToDevice);
    }

public:
    SFCBarnesHutCuda(int n, bool useSFC = true) : BarnesHutCuda(n), useSFC(useSFC)
    {
        if (useSFC)
        {
            // Allocate additional memory for SFC ordering
            CHECK_CUDA_ERROR(cudaMalloc(&d_codes, nBodies * sizeof(uint64_t)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_indices, nBodies * sizeof(int)));
            CHECK_CUDA_ERROR(cudaMalloc(&d_sortedBodies, nBodies * sizeof(Body)));
        }
    }

    ~SFCBarnesHutCuda()
    {
        if (useSFC)
        {
            // Free SFC-specific memory
            CHECK_CUDA_ERROR(cudaFree(d_codes));
            CHECK_CUDA_ERROR(cudaFree(d_indices));
            CHECK_CUDA_ERROR(cudaFree(d_sortedBodies));
        }
    }

    void update()
    {
        resetCUDA();
        computeBoundingBoxCUDA();

        // Apply SFC ordering before constructing octree
        if (useSFC)
        {
            orderBodiesBySFC();
        }

        constructOctreeCUDA();
        computeForceCUDA();
        CHECK_LAST_CUDA_ERROR();
    }
};

#endif // SFC_MORTON_CUH