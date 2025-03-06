#ifndef BODY_SORTER_CUH
#define BODY_SORTER_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "../common/types.cuh"
#include "../common/constants.cuh"
#include "../common/error_handling.cuh"
#include "morton.cuh"

// Forward declarations of kernel functions
__global__ void ComputeMortonCodesKernel(Body *bodies, uint64_t *mortonCodes, int *indices,
                                         int nBodies, Vector minBound, Vector maxBound);
__global__ void ReorderBodiesKernel(Body *bodies, Body *sortedBodies, int *indices, int nBodies);

namespace sfc
{

    /**
     * @brief Class for sorting bodies by their Morton codes
     *
     * This class handles the sorting of bodies according to their Morton codes,
     * which improves memory coherence when traversing the octree.
     */
    class BodySorter
    {
    private:
        int nBodies;             // Number of bodies
        uint64_t *d_mortonCodes; // Device array for Morton codes
        int *d_indices;          // Device array for indices
        Body *d_tempBodies;      // Temporary buffer for sorted bodies

    public:
        /**
         * @brief Constructor
         * @param numBodies Number of bodies to sort
         */
        BodySorter(int numBodies);

        /**
         * @brief Destructor
         */
        ~BodySorter();

        /**
         * @brief Sort bodies by their Morton codes
         * @param d_bodies Device array of bodies to sort
         * @param minBound Minimum bounds of the domain
         * @param maxBound Maximum bounds of the domain
         */
        void sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound);
    };

} // namespace sfc

#endif // BODY_SORTER_CUH