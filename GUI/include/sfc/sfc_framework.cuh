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
    /**
     * @brief Enumeration for available SFC types
     */
    enum class CurveType
    {
        MORTON, // Z-order curve
        HILBERT // Hilbert curve
    };

    /**
     * @brief Abstract base class for Space-Filling Curves
     *
     * This class provides a common interface for all SFC implementations.
     */
    class SpaceFillingCurve
    {
    public:
        /**
         * @brief Virtual destructor
         */
        virtual ~SpaceFillingCurve() = default;

        /**
         * @brief Compute a SFC code from 3D position
         * @param pos 3D position
         * @param minBound Minimum domain bounds
         * @param maxBound Maximum domain bounds
         * @param bits Number of bits per dimension
         * @return SFC code
         */
        virtual uint64_t positionToCode(
            const Vector &pos,
            const Vector &minBound,
            const Vector &maxBound,
            int bits = MORTON_BITS) const = 0;
    };

    /**
     * @brief Morton (Z-order) curve implementation
     */
    class MortonCurve : public SpaceFillingCurve
    {
    public:
        /**
         * @brief Compute Morton code from 3D position
         * @param pos 3D position
         * @param minBound Minimum bounds of the domain
         * @param maxBound Maximum bounds of the domain
         * @param bits Number of bits per dimension
         * @return Morton code for the position
         */
        uint64_t positionToCode(
            const Vector &pos,
            const Vector &minBound,
            const Vector &maxBound,
            int bits = MORTON_BITS) const override;

        /**
         * @brief Expands a 21-bit integer to 63 bits by inserting 2 zeros after each bit
         * @param v Value to expand
         * @return Expanded value
         */
        __host__ __device__ uint64_t expandBits(uint64_t v) const;

        /**
         * @brief Compute Morton code from 3D coordinates
         * @param x X coordinate (must be appropriately scaled, typically to 21 bits)
         * @param y Y coordinate (must be appropriately scaled, typically to 21 bits)
         * @param z Z coordinate (must be appropriately scaled, typically to 21 bits)
         * @return 63-bit Morton code
         */
        __host__ __device__ uint64_t getMortonCode(uint64_t x, uint64_t y, uint64_t z) const;
    };

    /**
     * @brief Hilbert curve implementation
     */
    class HilbertCurve : public SpaceFillingCurve
    {
    public:
        /**
         * @brief Compute Hilbert code from 3D position
         * @param pos 3D position
         * @param minBound Minimum bounds of the domain
         * @param maxBound Maximum bounds of the domain
         * @param bits Number of bits per dimension
         * @return Hilbert code for the position
         */
        uint64_t positionToCode(
            const Vector &pos,
            const Vector &minBound,
            const Vector &maxBound,
            int bits = MORTON_BITS) const override;

        /**
         * @brief Rotate/flip a quadrant appropriately
         * @param n Side length
         * @param x X coordinate (in/out)
         * @param y Y coordinate (in/out)
         * @param z Z coordinate (in/out)
         * @param rx X rotation
         * @param ry Y rotation
         * @param rz Z rotation
         */
        __host__ __device__ void rotateAndFlip(
            uint32_t n, uint32_t *x, uint32_t *y, uint32_t *z,
            uint32_t rx, uint32_t ry, uint32_t rz) const;

        /**
         * @brief Convert (x,y,z) to Hilbert distance d
         * @param x X coordinate
         * @param y Y coordinate
         * @param z Z coordinate
         * @param bits Number of bits per dimension
         * @return Hilbert distance
         */
        __host__ __device__ uint64_t coordsToHilbert(
            uint32_t x, uint32_t y, uint32_t z, uint32_t bits) const;
    };

    /**
     * @brief Creates an appropriate SFC implementation
     * @param type Type of curve to create
     * @return Unique pointer to the created curve
     */
    std::unique_ptr<SpaceFillingCurve> createCurve(CurveType type);

    /**
     * @brief Body sorter using SFC
     *
     * This class provides functionality to sort bodies based on their position
     * along a space-filling curve.
     */
    class BodySorter
    {
    private:
        int numBodies;
        CurveType curveType;
        std::unique_ptr<SpaceFillingCurve> curve;

        uint64_t *d_mortonCodes;
        int *d_indices;

    public:
        /**
         * @brief Constructor
         * @param numBodies Number of bodies to sort
         * @param type Type of curve to use (default: MORTON)
         */
        BodySorter(int numBodies, CurveType type = CurveType::MORTON);

        /**
         * @brief Destructor
         */
        ~BodySorter();

        /**
         * @brief Get the curve type
         * @return Current curve type
         */
        CurveType getCurveType() const { return curveType; }

        /**
         * @brief Set the curve type
         * @param type New curve type
         */
        void setCurveType(CurveType type);

        /**
         * @brief Sort bodies based on their position along the SFC
         * @param d_bodies Device array of bodies
         * @param minBound Minimum domain bounds
         * @param maxBound Maximum domain bounds
         * @return Pointer to sorted indices
         */
        int *sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound);
    };

    /**
     * @brief Octant sorter using SFC
     *
     * This class provides functionality to sort octree nodes based on their
     * center position along a space-filling curve.
     */
    class OctantSorter
    {
    private:
        int numNodes;
        CurveType curveType;
        std::unique_ptr<SpaceFillingCurve> curve;

        uint64_t *d_mortonCodes;
        int *d_indices;

    public:
        /**
         * @brief Constructor
         * @param numNodes Number of nodes to sort
         * @param type Type of curve to use (default: MORTON)
         */
        OctantSorter(int numNodes, CurveType type = CurveType::MORTON);

        /**
         * @brief Destructor
         */
        ~OctantSorter();

        /**
         * @brief Get the curve type
         * @return Current curve type
         */
        CurveType getCurveType() const { return curveType; }

        /**
         * @brief Set the curve type
         * @param type New curve type
         */
        void setCurveType(CurveType type);

        /**
         * @brief Sort octants based on their center position along the SFC
         * @param d_nodes Device array of nodes
         * @param minBound Minimum domain bounds
         * @param maxBound Maximum domain bounds
         * @return Pointer to sorted indices
         */
        int *sortOctants(Node *d_nodes, const Vector &minBound, const Vector &maxBound);
    };

} // namespace sfc

#endif // SFC_FRAMEWORK_CUH