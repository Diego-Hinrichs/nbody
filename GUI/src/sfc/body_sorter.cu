#include "../../include/sfc/body_sorter.cuh"

namespace sfc
{
    BodySorter::BodySorter(int numBodies) : nBodies(numBodies)
    {
        // Asignar memoria en dispositivo
        CHECK_CUDA_ERROR(cudaMalloc(&d_mortonCodes, nBodies * sizeof(uint64_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_indices, nBodies * sizeof(int)));

        d_tempBodies = nullptr;
    }

    BodySorter::~BodySorter()
    {
        // Liberar memoria
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

    // Retornar puntero a índices ordenados
    int *BodySorter::sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound)
    {
        // Configuración de lanzamiento
        int blockSize = BLOCK_SIZE;
        int gridSize = (nBodies + blockSize - 1) / blockSize;

        // Computar códigos Morton para todos los cuerpos
        ComputeMortonCodesKernel<<<gridSize, blockSize>>>(
            d_bodies, d_mortonCodes, d_indices, nBodies, minBound, maxBound);
        CHECK_LAST_CUDA_ERROR();

        // Ordenar cuerpos por código Morton usando Thrust
        thrust::device_ptr<uint64_t> thrust_codes(d_mortonCodes);
        thrust::device_ptr<int> thrust_indices(d_indices);
        thrust::sort_by_key(thrust::device, thrust_codes, thrust_codes + nBodies, thrust_indices);

        // Retornar puntero a índices ordenados
        return d_indices;
    }

} // namespace sfc