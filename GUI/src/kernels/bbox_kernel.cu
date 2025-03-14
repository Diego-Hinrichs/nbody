#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

/**
 * @brief CUDA kernel to compute the bounding box for a set of bodies.
 *
 * This kernel calculates the bounding box for a given set of bodies, which
 * are represented by nodes and bodies arrays. The bounding box is used to
 * determine the spatial extent of the bodies in the simulation.
 *
 * @param nodes Pointer to the array of nodes representing the bounding box.
 * @param bodies Pointer to the array of bodies in the simulation.
 * @param orderedIndices Pointer to the array of ordered indices for the bodies.
 * @param useSFC Boolean flag indicating whether to use Space-Filling Curve (SFC) for ordering.
 * @param mutex Pointer to the mutex used for synchronization.
 * @param nBodies The number of bodies in the simulation.
 */
__global__ void ComputeBoundingBoxKernel(Node *node, Body *bodies, int *mutex, int nBodies)
{
    // Memoria compartida para cada dimensión
    __shared__ double topLeftFrontX[BLOCK_SIZE];
    __shared__ double topLeftFrontY[BLOCK_SIZE];
    __shared__ double topLeftFrontZ[BLOCK_SIZE];
    __shared__ double botRightBackX[BLOCK_SIZE];
    __shared__ double botRightBackY[BLOCK_SIZE];
    __shared__ double botRightBackZ[BLOCK_SIZE];

    int tx = threadIdx.x;
    int b = blockIdx.x * blockDim.x + tx;

    // Inicialización correcta para encontrar mínimos y máximos
    topLeftFrontX[tx] = INFINITY; // Para encontrar mínimo en X
    topLeftFrontY[tx] = INFINITY; // Para encontrar mínimo en Y
    topLeftFrontZ[tx] = INFINITY; // Para encontrar mínimo en Z

    botRightBackX[tx] = -INFINITY; // Para encontrar máximo en X
    botRightBackY[tx] = -INFINITY; // Para encontrar máximo en Y
    botRightBackZ[tx] = -INFINITY; // Para encontrar máximo en Z

    __syncthreads();

    // Cargar datos del cuerpo si está dentro del rango
    if (b < nBodies)
    {
        Body body = bodies[b];
        topLeftFrontX[tx] = body.position.x;
        topLeftFrontY[tx] = body.position.y;
        topLeftFrontZ[tx] = body.position.z;

        botRightBackX[tx] = body.position.x;
        botRightBackY[tx] = body.position.y;
        botRightBackZ[tx] = body.position.z;
    }

    // Reducción para encontrar mínimos y máximos
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            // Encontrar mínimos para topLeftFront
            topLeftFrontX[tx] = fmin(topLeftFrontX[tx], topLeftFrontX[tx + s]);
            topLeftFrontY[tx] = fmin(topLeftFrontY[tx], topLeftFrontY[tx + s]);
            topLeftFrontZ[tx] = fmin(topLeftFrontZ[tx], topLeftFrontZ[tx + s]);

            // Encontrar máximos para botRightBack
            botRightBackX[tx] = fmax(botRightBackX[tx], botRightBackX[tx + s]);
            botRightBackY[tx] = fmax(botRightBackY[tx], botRightBackY[tx + s]);
            botRightBackZ[tx] = fmax(botRightBackZ[tx], botRightBackZ[tx + s]);
        }
    }

    // Actualización del nodo raíz con mutex
    if (tx == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0)
            ;
        // Actualizar mínimos y máximos con margen
        node[0].topLeftFront.x = fmin(node[0].topLeftFront.x, topLeftFrontX[0] - 1.0e10);
        node[0].topLeftFront.y = fmin(node[0].topLeftFront.y, topLeftFrontY[0] - 1.0e10);
        node[0].topLeftFront.z = fmin(node[0].topLeftFront.z, topLeftFrontZ[0] - 1.0e10);

        node[0].botRightBack.x = fmax(node[0].botRightBack.x, botRightBackX[0] + 1.0e10);
        node[0].botRightBack.y = fmax(node[0].botRightBack.y, botRightBackY[0] + 1.0e10);
        node[0].botRightBack.z = fmax(node[0].botRightBack.z, botRightBackZ[0] + 1.0e10);

        atomicExch(mutex, 0);
    }
}
