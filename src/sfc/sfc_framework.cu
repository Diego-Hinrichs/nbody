#include "../../include/sfc/sfc_framework.cuh"
#include "../../include/common/error_handling.cuh"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

namespace sfc
{
    //-------------------------------------------------------------------------
    // MortonCurve Implementation
    //-------------------------------------------------------------------------

    __host__ __device__ uint64_t MortonCurve::expandBits(uint64_t v) const
    {
        v = (v | v << 32) & 0x1f00000000ffff;
        v = (v | v << 16) & 0x1f0000ff0000ff;
        v = (v | v << 8) & 0x100f00f00f00f00f;
        v = (v | v << 4) & 0x10c30c30c30c30c3;
        v = (v | v << 2) & 0x1249249249249249;
        return v;
    }

    __host__ __device__ uint64_t MortonCurve::getMortonCode(uint64_t x, uint64_t y, uint64_t z) const
    {
        x = expandBits(x);
        y = expandBits(y);
        z = expandBits(z);
        return x | (y << 1) | (z << 2);
    }

    uint64_t MortonCurve::positionToCode(const Vector &pos, const Vector &minBound,
                                         const Vector &maxBound, int bits) const
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

    //-------------------------------------------------------------------------
    // HilbertCurve Implementation
    //-------------------------------------------------------------------------

    __host__ __device__ void HilbertCurve::rotateAndFlip(
        uint32_t n, uint32_t *x, uint32_t *y, uint32_t *z,
        uint32_t rx, uint32_t ry, uint32_t rz) const
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

    __host__ __device__ uint64_t HilbertCurve::coordsToHilbert(
        uint32_t x, uint32_t y, uint32_t z, uint32_t bits) const
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

    uint64_t HilbertCurve::positionToCode(const Vector &pos, const Vector &minBound,
                                          const Vector &maxBound, int bits) const
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

        return coordsToHilbert(x, y, z, bits);
    }

    //-------------------------------------------------------------------------
    // Factory Function
    //-------------------------------------------------------------------------

    std::unique_ptr<SpaceFillingCurve> createCurve(CurveType type)
    {
        switch (type)
        {
        case CurveType::HILBERT:
            return std::make_unique<HilbertCurve>();
        case CurveType::MORTON:
        default:
            return std::make_unique<MortonCurve>();
        }
    }

    //-------------------------------------------------------------------------
    // BodySorter Implementation
    //-------------------------------------------------------------------------

    BodySorter::BodySorter(int numBodies, CurveType type)
        : numBodies(numBodies), curveType(type), d_mortonCodes(nullptr), d_indices(nullptr)
    {
        // Create the appropriate curve
        curve = createCurve(type);

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_mortonCodes, numBodies * sizeof(uint64_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_indices, numBodies * sizeof(int)));
    }

    BodySorter::~BodySorter()
    {
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
    }

    void BodySorter::setCurveType(CurveType type)
    {
        if (type != curveType)
        {
            curveType = type;
            curve = createCurve(type);
        }
    }

    int *BodySorter::sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound)
    {
        // Determine block and grid size
        int blockSize = BLOCK_SIZE;
        int gridSize = (numBodies + blockSize - 1) / blockSize;

        // Determine whether we're using Hilbert curve
        bool isHilbert = (curveType == CurveType::HILBERT);

        // Launch kernel to compute SFC codes
        ComputeSFCCodesKernelForBodies<<<gridSize, blockSize>>>(
            d_bodies, d_mortonCodes, d_indices, numBodies, minBound, maxBound, isHilbert);
        CHECK_LAST_CUDA_ERROR();

        // Sort indices by SFC codes
        thrust::device_ptr<uint64_t> thrust_codes(d_mortonCodes);
        thrust::device_ptr<int> thrust_indices(d_indices);
        thrust::sort_by_key(thrust::device, thrust_codes, thrust_codes + numBodies, thrust_indices);

        return d_indices;
    }

    //-------------------------------------------------------------------------
    // OctantSorter Implementation
    //-------------------------------------------------------------------------

    OctantSorter::OctantSorter(int numNodes, CurveType type)
        : numNodes(numNodes), curveType(type), d_mortonCodes(nullptr), d_indices(nullptr)
    {
        // Create the appropriate curve
        curve = createCurve(type);

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_mortonCodes, numNodes * sizeof(uint64_t)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_indices, numNodes * sizeof(int)));
    }

    OctantSorter::~OctantSorter()
    {
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
    }

    void OctantSorter::setCurveType(CurveType type)
    {
        if (type != curveType)
        {
            curveType = type;
            curve = createCurve(type);
        }
    }

    int *OctantSorter::sortOctants(Node *d_nodes, const Vector &minBound, const Vector &maxBound)
    {
        // Determine block and grid size
        int blockSize = BLOCK_SIZE;
        int gridSize = (numNodes + blockSize - 1) / blockSize;

        // Determine whether we're using Hilbert curve
        bool isHilbert = (curveType == CurveType::HILBERT);

        // Launch kernel to compute SFC codes
        ComputeSFCCodesKernelForNodes<<<gridSize, blockSize>>>(
            d_nodes, d_mortonCodes, d_indices, numNodes, minBound, maxBound, isHilbert);
        CHECK_LAST_CUDA_ERROR();

        // Sort indices by SFC codes
        thrust::device_ptr<uint64_t> thrust_codes(d_mortonCodes);
        thrust::device_ptr<int> thrust_indices(d_indices);
        thrust::sort_by_key(thrust::device, thrust_codes, thrust_codes + numNodes, thrust_indices);

        return d_indices;
    }

} // namespace sfc

__global__ void ComputeSFCCodesKernelForBodies(
    Body *bodies, uint64_t *codes, int *indices,
    int numBodies, Vector minBound, Vector maxBound,
    bool isHilbert)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numBodies)
    {
        // Get body position
        Vector position = bodies[idx].position;

        uint64_t code;
        if (isHilbert)
        {
            // Use Hilbert curve
            // Normalize coordinates to [0,1]
            double normalizedX = (position.x - minBound.x) / (maxBound.x - minBound.x);
            double normalizedY = (position.y - minBound.y) / (maxBound.y - minBound.y);
            double normalizedZ = (position.z - minBound.z) / (maxBound.z - minBound.z);

            // Clamp to ensure values are in range [0,1]
            normalizedX = fmin(fmax(normalizedX, 0.0), 1.0);
            normalizedY = fmin(fmax(normalizedY, 0.0), 1.0);
            normalizedZ = fmin(fmax(normalizedZ, 0.0), 1.0);

            // Scale to [0, 2^bits-1]
            uint32_t maxCoord = (1 << MORTON_BITS) - 1;
            uint32_t x = uint32_t(normalizedX * maxCoord);
            uint32_t y = uint32_t(normalizedY * maxCoord);
            uint32_t z = uint32_t(normalizedZ * maxCoord);

            // Calculate Hilbert code
            code = sfc::HilbertCurve().coordsToHilbert(x, y, z, MORTON_BITS);
        }
        else
        {
            // Use Morton curve
            // Normalize coordinates to [0,1]
            double normalizedX = (position.x - minBound.x) / (maxBound.x - minBound.x);
            double normalizedY = (position.y - minBound.y) / (maxBound.y - minBound.y);
            double normalizedZ = (position.z - minBound.z) / (maxBound.z - minBound.z);

            // Clamp to ensure values are in range [0,1]
            normalizedX = fmin(fmax(normalizedX, 0.0), 1.0);
            normalizedY = fmin(fmax(normalizedY, 0.0), 1.0);
            normalizedZ = fmin(fmax(normalizedZ, 0.0), 1.0);

            // Scale to [0, 2^bits-1]
            uint64_t maxCoord = (1ULL << MORTON_BITS) - 1;
            uint64_t x = uint64_t(normalizedX * maxCoord);
            uint64_t y = uint64_t(normalizedY * maxCoord);
            uint64_t z = uint64_t(normalizedZ * maxCoord);

            // Calculate Morton code
            code = sfc::MortonCurve().getMortonCode(x, y, z);
        }

        // Store results
        codes[idx] = code;
        indices[idx] = idx;
    }
}

__global__ void ComputeSFCCodesKernelForNodes(
    Node *nodes, uint64_t *codes, int *indices,
    int numNodes, Vector minBound, Vector maxBound,
    bool isHilbert)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numNodes)
    {
        // Skip empty nodes
        if (nodes[idx].start == -1 && nodes[idx].end == -1)
        {
            codes[idx] = 0; // Assign lowest priority to empty nodes
        }
        else
        {
            // Get node center
            Vector center = nodes[idx].getCenter();

            // Calculate code
            uint64_t code;
            if (isHilbert)
            {
                // Use Hilbert curve
                // Normalize coordinates to [0,1]
                double normalizedX = (center.x - minBound.x) / (maxBound.x - minBound.x);
                double normalizedY = (center.y - minBound.y) / (maxBound.y - minBound.y);
                double normalizedZ = (center.z - minBound.z) / (maxBound.z - minBound.z);

                // Clamp to ensure values are in range [0,1]
                normalizedX = fmin(fmax(normalizedX, 0.0), 1.0);
                normalizedY = fmin(fmax(normalizedY, 0.0), 1.0);
                normalizedZ = fmin(fmax(normalizedZ, 0.0), 1.0);

                // Scale to [0, 2^bits-1]
                uint32_t maxCoord = (1 << MORTON_BITS) - 1;
                uint32_t x = uint32_t(normalizedX * maxCoord);
                uint32_t y = uint32_t(normalizedY * maxCoord);
                uint32_t z = uint32_t(normalizedZ * maxCoord);

                // Calculate Hilbert code
                code = sfc::HilbertCurve().coordsToHilbert(x, y, z, MORTON_BITS);
            }
            else
            {
                // Use Morton curve
                // Normalize coordinates to [0,1]
                double normalizedX = (center.x - minBound.x) / (maxBound.x - minBound.x);
                double normalizedY = (center.y - minBound.y) / (maxBound.y - minBound.y);
                double normalizedZ = (center.z - minBound.z) / (maxBound.z - minBound.z);

                // Clamp to ensure values are in range [0,1]
                normalizedX = fmin(fmax(normalizedX, 0.0), 1.0);
                normalizedY = fmin(fmax(normalizedY, 0.0), 1.0);
                normalizedZ = fmin(fmax(normalizedZ, 0.0), 1.0);

                // Scale to [0, 2^bits-1]
                uint64_t maxCoord = (1ULL << MORTON_BITS) - 1;
                uint64_t x = uint64_t(normalizedX * maxCoord);
                uint64_t y = uint64_t(normalizedY * maxCoord);
                uint64_t z = uint64_t(normalizedZ * maxCoord);

                // Calculate Morton code
                code = sfc::MortonCurve().getMortonCode(x, y, z);
            }

            codes[idx] = code;
        }

        indices[idx] = idx; // Initialize with sequential indices
    }
}
