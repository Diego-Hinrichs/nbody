#include "../common/types.cuh"
#include "../common/constants.cuh"
#include "../sfc/body_sorter.cuh"
#include <cstdint>

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

__global__ void ComputeOctantMortonCodesKernel(Node *nodes, uint64_t *mortonCodes, int *indices,
                                               int nNodes, Vector minBound, Vector maxBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodes)
    {
        // Skip empty nodes
        if (nodes[idx].start == -1 && nodes[idx].end == -1)
        {
            mortonCodes[idx] = 0; // Assign lowest priority to empty nodes
        }
        else
        {
            // Use node center for Morton code calculation
            Vector center = nodes[idx].getCenter();
            mortonCodes[idx] = sfc::positionToMorton(center, minBound, maxBound);
        }
        indices[idx] = idx; // Initialize with sequential indices
    }
}