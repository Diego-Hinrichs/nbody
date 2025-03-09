#include "../../include/sfc/body_sorter.cuh"

/**
 * @brief Kernel para calcular códigos Morton para un conjunto de cuerpos
 */
__global__ void ComputeMortonCodesKernel(Body *bodies, uint64_t *mortonCodes, int *indices,
                                         int nBodies, Vector minBound, Vector maxBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nBodies)
    {
        // Normalizar coordenadas a [0, 2^21-1]
        double maxCoord = (1ULL << MORTON_BITS) - 1;

        // Calcular coordenadas normalizadas
        double normalizedX = (bodies[idx].position.x - minBound.x) / (maxBound.x - minBound.x);
        double normalizedY = (bodies[idx].position.y - minBound.y) / (maxBound.y - minBound.y);
        double normalizedZ = (bodies[idx].position.z - minBound.z) / (maxBound.z - minBound.z);

        // Asegurar que estén en rango [0,1]
        normalizedX = fmin(fmax(normalizedX, 0.0), 1.0);
        normalizedY = fmin(fmax(normalizedY, 0.0), 1.0);
        normalizedZ = fmin(fmax(normalizedZ, 0.0), 1.0);

        // Escalar a [0, 2^21-1]
        uint32_t x = uint32_t(normalizedX * maxCoord);
        uint32_t y = uint32_t(normalizedY * maxCoord);
        uint32_t z = uint32_t(normalizedZ * maxCoord);

        // Calcular código Morton usando la función libmorton (esto se hace en CPU)
        // En CUDA, seguimos usando nuestra implementación optimizada
        mortonCodes[idx] = (x & 0x1FFFFF) | ((y & 0x1FFFFF) << 21) | ((uint64_t)(z & 0x1FFFFF) << 42);

        // Inicializar índices secuenciales
        indices[idx] = idx;
    }
}

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