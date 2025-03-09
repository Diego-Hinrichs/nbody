#ifndef BODY_SORTER_CUH
#define BODY_SORTER_CUH

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "../common/types.cuh"
#include "../common/constants.cuh"
#include "../common/error_handling.cuh"
// Incluir libmorton en lugar de nuestro morton.cuh
#include "../../external/libmorton/include/libmorton/morton.h"

// Forward declarations de kernels
__global__ void ComputeMortonCodesKernel(Body *bodies, uint64_t *mortonCodes, int *indices,
                                         int nBodies, Vector minBound, Vector maxBound);

                                         namespace sfc
{
    /**
     * @brief Clase para ordenar cuerpos según sus códigos Morton
     */
    class BodySorter
    {
    private:
        int nBodies;
        uint64_t *d_mortonCodes;
        int *d_indices;
        Body *d_tempBodies;

    public:
        /**
         * @brief Constructor
         * @param numBodies Número de cuerpos a ordenar
         */
        BodySorter(int numBodies);

        /**
         * @brief Destructor
         */
        ~BodySorter();

        /**
         * @brief Ordenar cuerpos según sus códigos Morton
         * @param d_bodies Array de cuerpos en dispositivo
         * @param minBound Límite mínimo del dominio
         * @param maxBound Límite máximo del dominio
         * @return Puntero al array de índices ordenados
         */
        int *sortBodies(Body *d_bodies, const Vector &minBound, const Vector &maxBound);
    };

} // namespace sfc

#endif // BODY_SORTER_CUH