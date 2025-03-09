#ifndef MORTON_CUH
#define MORTON_CUH

#include "../common/types.cuh"
#include "../common/constants.cuh"
#include <cstdint>

__global__ void ComputeMortonCodesKernel(Body *bodies, uint64_t *mortonCodes, int *indices,
                                         int nBodies, Vector minBound, Vector maxBound);

__global__ void ComputeOctantMortonCodesKernel(Node *nodes, uint64_t *mortonCodes, int *indices,
                                               int nNodes, Vector minBound, Vector maxBound);

namespace sfc
{
    /**
     * @brief Expands a 21-bit integer to 63 bits by inserting 2 zeros after each bit
     * @param v Value to expand
     * @return Expanded value
     */
    __host__ __device__ inline uint64_t expandBits(uint64_t v)
    {
        v = (v | v << 32) & 0x1f00000000ffff;
        v = (v | v << 16) & 0x1f0000ff0000ff;
        v = (v | v << 8) & 0x100f00f00f00f00f;
        v = (v | v << 4) & 0x10c30c30c30c30c3;
        v = (v | v << 2) & 0x1249249249249249;
        return v;
    }

    /**
     * @brief Compute Morton code from 3D coordinates
     * @param x X coordinate (must be appropriately scaled, typically to 21 bits)
     * @param y Y coordinate (must be appropriately scaled, typically to 21 bits)
     * @param z Z coordinate (must be appropriately scaled, typically to 21 bits)
     * @return 63-bit Morton code
     */
    __host__ __device__ inline uint64_t getMortonCode(uint64_t x, uint64_t y, uint64_t z)
    {
        x = expandBits(x);
        y = expandBits(y);
        z = expandBits(z);
        return x | (y << 1) | (z << 2);
    }

    /**
     * @brief Convert position to Morton code
     * @param pos 3D position
     * @param minBound Minimum bounds of the domain
     * @param maxBound Maximum bounds of the domain
     * @param bits Number of bits per dimension (default 21)
     * @return Morton code for the position
     */
    __host__ __device__ inline uint64_t positionToMorton(
        const Vector &pos,
        const Vector &minBound,
        const Vector &maxBound,
        int bits = MORTON_BITS)
    {
        double maxCoord = (1ULL << bits) - 1;

        // Normalize coordinates to [0,1]
        double normalizedX = (pos.x - minBound.x) / (maxBound.x - minBound.x);
        double normalizedY = (pos.y - minBound.y) / (maxBound.y - minBound.y);
        double normalizedZ = (pos.z - minBound.z) / (maxBound.z - minBound.z);

        // Clamp to ensure values are in range [0,1]
        normalizedX = fmin(fmax(normalizedX, 0.0), 1.0);
        normalizedY = fmin(fmax(normalizedY, 0.0), 1.0);
        normalizedZ = fmin(fmax(normalizedZ, 0.0), 1.0);

        // Scale to [0, 2^bits-1]
        uint64_t x = uint64_t(normalizedX * maxCoord);
        uint64_t y = uint64_t(normalizedY * maxCoord);
        uint64_t z = uint64_t(normalizedZ * maxCoord);

        return getMortonCode(x, y, z);
    }

} // namespace sfc

#endif // MORTON_CUH