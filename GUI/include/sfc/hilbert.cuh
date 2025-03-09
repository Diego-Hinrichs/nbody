#ifndef HILBERT_CUH
#define HILBERT_CUH

#include "../common/types.cuh"
#include "../common/constants.cuh"
#include <cstdint>

__global__ void ComputeHilbertCodesKernel(Body *bodies, uint64_t *hilbertCodes, int *indices,
                                          int nBodies, Vector minBound, Vector maxBound);

__global__ void ComputeOctantHilbertCodesKernel(Node *nodes, uint64_t *hilbertCodes, int *indices,
                                                int nNodes, Vector minBound, Vector maxBound);

namespace sfc
{
    // Rotate/flip a quadrant appropriately
    __host__ __device__ inline void rotateAndFlip(
        uint32_t n, uint32_t *x, uint32_t *y, uint32_t *z,
        uint32_t rx, uint32_t ry, uint32_t rz)
    {
        if (ry == 0)
        {
            // Swap x and y if ry is 0
            if (rz == 1)
            {
                *x = n - 1 - *x;
                *y = n - 1 - *y;
            }
            // Swap x and z
            uint32_t t = *x;
            *x = *z;
            *z = t;
        }
        else
        {
            // Swap y and z if ry is 1
            if (rx == 1)
            {
                *y = n - 1 - *y;
                *z = n - 1 - *z;
            }
            // Swap y and z
            uint32_t t = *y;
            *y = *z;
            *z = t;
        }
    }

    // Convert (x,y,z) to Hilbert distance d
    __host__ __device__ inline uint64_t coordsToHilbert(uint32_t x, uint32_t y, uint32_t z, uint32_t bits)
    {
        uint64_t d = 0;
        uint32_t rx, ry, rz, s;
        uint32_t n = 1 << bits; // side length of the cube (2^bits)

        for (s = n / 2; s > 0; s >>= 1)
        {
            rx = (x & s) > 0 ? 1 : 0;
            ry = (y & s) > 0 ? 1 : 0;
            rz = (z & s) > 0 ? 1 : 0;

            // Add contribution from this level
            d += s * s * s * ((3 * rx) ^ (2 * ry) ^ rz);

            // Rotate/flip the quadrant to place the origin correctly for the next level
            rotateAndFlip(s * 2, &x, &y, &z, rx, ry, rz);
        }

        return d;
    }

    // Convert Hilbert distance d to (x,y,z) coordinates
    __host__ __device__ inline void hilbertToCoords(uint64_t d, uint32_t bits, uint32_t *x, uint32_t *y, uint32_t *z)
    {
        uint32_t rx, ry, rz, t = d;
        uint32_t n = 1 << bits; // side length of the cube (2^bits)

        *x = *y = *z = 0;
        for (uint32_t s = 1; s < n; s *= 2)
        {
            rx = 1 & (t / 3);
            ry = 1 & ((t / 3) ^ rx);
            rz = 1 & ((t ^ rx) ^ ry);

            rotateAndFlip(s * 2, x, y, z, rx, ry, rz);

            *x += s * rx;
            *y += s * ry;
            *z += s * rz;

            t /= 7; // Move to the next triplet of bits
        }
    }

    /**
     * @brief Compute Hilbert code from 3D coordinates
     * @param x X coordinate (must be appropriately scaled, typically to 'bits' bits)
     * @param y Y coordinate (must be appropriately scaled, typically to 'bits' bits)
     * @param z Z coordinate (must be appropriately scaled, typically to 'bits' bits)
     * @param bits Number of bits per dimension (default MORTON_BITS)
     * @return Hilbert index
     */
    __host__ __device__ inline uint64_t getHilbertCode(uint32_t x, uint32_t y, uint32_t z, uint32_t bits = MORTON_BITS)
    {
        return coordsToHilbert(x, y, z, bits);
    }

    /**
     * @brief Convert position to Hilbert code
     * @param pos 3D position
     * @param minBound Minimum bounds of the domain
     * @param maxBound Maximum bounds of the domain
     * @param bits Number of bits per dimension (default MORTON_BITS)
     * @return Hilbert code for the position
     */
    __host__ __device__ inline uint64_t positionToHilbert(
        const Vector &pos,
        const Vector &minBound,
        const Vector &maxBound,
        int bits = MORTON_BITS)
    {
        // Normalize coordinates to [0,1]
        double normalizedX = (pos.x - minBound.x) / (maxBound.x - minBound.x);
        double normalizedY = (pos.y - minBound.y) / (maxBound.y - minBound.y);
        double normalizedZ = (pos.z - minBound.z) / (maxBound.z - minBound.z);

        // Clamp to ensure values are in range [0,1]
        normalizedX = fmin(fmax(normalizedX, 0.0), 1.0);
        normalizedY = fmin(fmax(normalizedY, 0.0), 1.0);
        normalizedZ = fmin(fmax(normalizedZ, 0.0), 1.0);

        // Scale to [0, 2^bits-1]
        uint32_t maxCoord = (1 << bits) - 1;
        uint32_t x = uint32_t(normalizedX * maxCoord);
        uint32_t y = uint32_t(normalizedY * maxCoord);
        uint32_t z = uint32_t(normalizedZ * maxCoord);

        return getHilbertCode(x, y, z, bits);
    }

} // namespace sfc

#endif // HILBERT_CUH