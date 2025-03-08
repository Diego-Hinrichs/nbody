#include "../../include/sfc/body_sorter.cuh"

/**
 * @brief Kernel function to compute Morton codes for a set of bodies.
 *
 * This kernel computes the Morton codes for a given set of bodies based on their positions
 * and the provided bounding box. The Morton codes are used for spatial sorting of the bodies.
 *
 * @param bodies Pointer to the array of Body structures representing the bodies.
 * @param mortonCodes Pointer to the array where the computed Morton codes will be stored.
 * @param indices Pointer to the array where the indices of the bodies will be stored.
 * @param nBodies The number of bodies in the array.
 * @param minBound The minimum bound of the bounding box.
 * @param maxBound The maximum bound of the bounding box.
 */
__global__ void ComputeMortonCodesKernel(Body *bodies, uint64_t *mortonCodes, int *indices,
                                         int nBodies, Vector minBound, Vector maxBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBodies)
    {
        mortonCodes[idx] = sfc::positionToMorton(bodies[idx].position, minBound, maxBound);
        indices[idx] = idx; // Inicializar con índices secuenciales
    }
}

namespace sfc
{

    BodySorter::BodySorter(int numBodies) : nBodies(numBodies)
    {
        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_mortonCodes, nBodies * sizeof(uint64_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_indices, nBodies * sizeof(int)));

        d_tempBodies = nullptr;
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

    // Modificar para retornar el puntero a los índices ordenados en lugar de reordenar los cuerpos
    int *BodySorter::sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound)
    {
        // Calculate launch configuration
        int blockSize = BLOCK_SIZE;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        // Compute Morton codes for all bodies
        ComputeMortonCodesKernel<<<gridSize, blockSize>>>(
            d_bodies, d_mortonCodes, d_indices, nBodies, minBound, maxBound);
        CHECK_LAST_CUDA_ERROR();

        // Sort bodies by Morton code using Thrust (solo ordenamos los índices)
        thrust::device_ptr<uint64_t> thrust_codes(d_mortonCodes);
        thrust::device_ptr<int> thrust_indices(d_indices);
        thrust::sort_by_key(thrust::device, thrust_codes, thrust_codes + nBodies, thrust_indices);

        // Retornar el puntero a los índices ordenados
        return d_indices;
    }

} // namespace sfc