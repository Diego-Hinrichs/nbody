#ifndef SFC_FRAMEWORK_CUH
#define SFC_FRAMEWORK_CUH

#include <cstdint>
#include <memory>
#include "../common/types.cuh"
#include "../common/constants.cuh"

// Forward declare CUDA kernels
__global__ void ComputeSFCCodesKernelForBodies(
    Body *bodies, uint64_t *codes, int *indices,
    int numBodies, Vector minBound, Vector maxBound,
    bool isHilbert);

__global__ void ComputeSFCCodesKernelForNodes(
    Node *nodes, uint64_t *codes, int *indices,
    int numNodes, Vector minBound, Vector maxBound,
    bool isHilbert);

namespace sfc
{
    enum class CurveType
    {
        MORTON, // Z-order curve
        HILBERT // Hilbert curve
    };

    class SpaceFillingCurve
    {
    public:
        virtual ~SpaceFillingCurve() = default;
        virtual uint64_t positionToCode(const Vector &pos, const Vector &minBound, const Vector &maxBound, int bits = MORTON_BITS) const = 0;
    };

    class MortonCurve : public SpaceFillingCurve
    {
    public:
        uint64_t positionToCode(const Vector &pos, const Vector &minBound, const Vector &maxBound, int bits = MORTON_BITS) const override;
        __host__ __device__ uint64_t expandBits(uint64_t v) const;
        __host__ __device__ uint64_t getMortonCode(uint64_t x, uint64_t y, uint64_t z) const;
    };

    class HilbertCurve : public SpaceFillingCurve
    {
    public:
        uint64_t positionToCode(const Vector &pos, const Vector &minBound, const Vector &maxBound, int bits = MORTON_BITS) const override;
        __host__ __device__ void rotateAndFlip(uint32_t n, uint32_t *x, uint32_t *y, uint32_t *z, uint32_t rx, uint32_t ry, uint32_t rz) const;
        __host__ __device__ uint64_t coordsToHilbert(uint32_t x, uint32_t y, uint32_t z, uint32_t bits) const;
    };

    std::unique_ptr<SpaceFillingCurve> createCurve(CurveType type);

    class BodySorter
    {
    private:
        int numBodies;
        CurveType curveType;
        std::unique_ptr<SpaceFillingCurve> curve;
        uint64_t *d_mortonCodes;
        int *d_indices;

    public:

        BodySorter(int numBodies, CurveType type = CurveType::MORTON);
        ~BodySorter();
        CurveType getCurveType() const { return curveType; }
        void setCurveType(CurveType type);
        int *sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound);
    };

    class OctantSorter
    {
    private:
        int numNodes;
        CurveType curveType;
        std::unique_ptr<SpaceFillingCurve> curve;

        uint64_t *d_mortonCodes;
        int *d_indices;

    public:
        OctantSorter(int numNodes, CurveType type = CurveType::MORTON);
        ~OctantSorter();
        CurveType getCurveType() const { return curveType; }
        void setCurveType(CurveType type);
        int *sortOctants(Node *d_nodes, const Vector &minBound, const Vector &maxBound);
    };

}

#endif