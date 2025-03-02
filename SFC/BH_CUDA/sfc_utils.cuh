#ifndef SFC_UTILS_H
#define SFC_UTILS_H

#include <cstdint>
#include "barnesHutCuda.cuh"

// Morton encoding (Z-order curve)
__host__ __device__ inline uint64_t expandBits(uint32_t v) {
    uint64_t x = v & 0x1fffff; // 21 bits is enough for most N-body simulations
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

__host__ __device__ inline uint64_t mortonEncode(uint32_t x, uint32_t y, uint32_t z) {
    uint64_t answer = 0;
    answer |= expandBits(x) | (expandBits(y) << 1) | (expandBits(z) << 2);
    return answer;
}

// Scaling coordinates to the fixed range [0, 2^21-1]
__host__ __device__ inline uint32_t scaleCoordinate(double val, double min, double max) {
    // Scale and ensure the value is in the [0, 2^21-1] range
    double normalized = (val - min) / (max - min);
    return static_cast<uint32_t>(normalized * ((1 << 21) - 1));
}

// Morton code for a 3D position
__host__ __device__ inline uint64_t positionToMortonCode(Vector pos, Vector min, Vector max) {
    uint32_t x = scaleCoordinate(pos.x, min.x, max.x);
    uint32_t y = scaleCoordinate(pos.y, min.y, max.y);
    uint32_t z = scaleCoordinate(pos.z, min.z, max.z);
    return mortonEncode(x, y, z);
}

// Hilbert curve encoding (2D version converted to 3D)
__host__ __device__ inline void rotateAndFlip(uint32_t n, uint32_t *x, uint32_t *y, uint32_t rx, uint32_t ry) {
    if (ry == 0) {
        if (rx == 1) {
            *x = n - 1 - *x;
            *y = n - 1 - *y;
        }
        // Swap x and y
        uint32_t t = *x;
        *x = *y;
        *y = t;
    }
}

__host__ __device__ inline uint64_t xy2d(uint32_t n, uint32_t x, uint32_t y) {
    uint64_t d = 0;
    for (int s = n/2; s > 0; s /= 2) {
        uint32_t rx = (x & s) > 0;
        uint32_t ry = (y & s) > 0;
        d += s * s * ((3 * rx) ^ ry);
        rotateAndFlip(s, &x, &y, rx, ry);
    }
    return d;
}

// 3D Hilbert curve using 2D implementation and stacking
__host__ __device__ inline uint64_t hilbertEncode(uint32_t x, uint32_t y, uint32_t z, uint32_t bits) {
    uint32_t n = 1 << bits; // 2^bits
    // Combine z with xy using Hilbert for xy and linear for z
    uint64_t xy = xy2d(n, x, y);
    return (static_cast<uint64_t>(z) << (2 * bits)) | xy;
}

__host__ __device__ inline uint64_t positionToHilbertCode(Vector pos, Vector min, Vector max, uint32_t bits = 10) {
    uint32_t x = scaleCoordinate(pos.x, min.x, max.x) >> (21 - bits);
    uint32_t y = scaleCoordinate(pos.y, min.y, max.y) >> (21 - bits);
    uint32_t z = scaleCoordinate(pos.z, min.z, max.z) >> (21 - bits);
    return hilbertEncode(x, y, z, bits);
}

// Comparator functions for sorting
struct MortonComparator {
    Vector min, max;
    
    MortonComparator(Vector _min, Vector _max) : min(_min), max(_max) {}
    
    __host__ __device__ inline bool operator()(const Body& a, const Body& b) const {
        uint64_t codeA = positionToMortonCode(a.position, min, max);
        uint64_t codeB = positionToMortonCode(b.position, min, max);
        return codeA < codeB;
    }
};

struct HilbertComparator {
    Vector min, max;
    uint32_t bits;
    
    HilbertComparator(Vector _min, Vector _max, uint32_t _bits = 10) 
        : min(_min), max(_max), bits(_bits) {}
    
    __host__ __device__ inline bool operator()(const Body& a, const Body& b) const {
        uint64_t codeA = positionToHilbertCode(a.position, min, max, bits);
        uint64_t codeB = positionToHilbertCode(b.position, min, max, bits);
        return codeA < codeB;
    }
};

#endif // SFC_UTILS_H