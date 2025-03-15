#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

/**
 * @brief Compute the bounding box for the entire simulation domain
 *
 * This kernel uses a parallel reduction approach to find the minimum and
 * maximum coordinates of all bodies, defining the simulation domain.
 *
 * @param nodes Array of octree nodes (only updates the root node)
 * @param bodies Array of bodies
 * @param mutex Mutex for synchronization
 * @param nBodies Number of bodies in the simulation
 */
__global__ void ComputeBoundingBoxKernel(Node *nodes, Body *bodies, int *mutex, int nBodies)
{
    // Shared memory for parallel reduction of each dimension
    __shared__ double topLeftFrontX[BLOCK_SIZE];
    __shared__ double topLeftFrontY[BLOCK_SIZE];
    __shared__ double topLeftFrontZ[BLOCK_SIZE];
    __shared__ double botRightBackX[BLOCK_SIZE];
    __shared__ double botRightBackY[BLOCK_SIZE];
    __shared__ double botRightBackZ[BLOCK_SIZE];

    int tx = threadIdx.x;
    int b = blockIdx.x * blockDim.x + tx;

    // Initialize with extreme values
    topLeftFrontX[tx] = INFINITY;  // Min X
    topLeftFrontY[tx] = INFINITY;  // Min Y
    topLeftFrontZ[tx] = INFINITY;  // Min Z
    botRightBackX[tx] = -INFINITY; // Max X
    botRightBackY[tx] = -INFINITY; // Max Y
    botRightBackZ[tx] = -INFINITY; // Max Z

    __syncthreads();

    // Load body data if within range
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

    // Parallel reduction to find min/max values
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            // Min reduction for top-left-front
            topLeftFrontX[tx] = fmin(topLeftFrontX[tx], topLeftFrontX[tx + s]);
            topLeftFrontY[tx] = fmin(topLeftFrontY[tx], topLeftFrontY[tx + s]);
            topLeftFrontZ[tx] = fmin(topLeftFrontZ[tx], topLeftFrontZ[tx + s]);

            // Max reduction for bottom-right-back
            botRightBackX[tx] = fmax(botRightBackX[tx], botRightBackX[tx + s]);
            botRightBackY[tx] = fmax(botRightBackY[tx], botRightBackY[tx + s]);
            botRightBackZ[tx] = fmax(botRightBackZ[tx], botRightBackZ[tx + s]);
        }
    }

    // Update root node with mutex to avoid race conditions
    if (tx == 0)
    {
        // Wait until mutex is available
        while (atomicCAS(mutex, 0, 1) != 0)
        {
        }

        // Update bounds with a margin for numerical stability
        // Update minimum bounds (top-left-front corner)
        nodes[0].topLeftFront.x = fmin(nodes[0].topLeftFront.x, topLeftFrontX[0] - 1.0e10);
        nodes[0].topLeftFront.y = fmin(nodes[0].topLeftFront.y, topLeftFrontY[0] - 1.0e10);
        nodes[0].topLeftFront.z = fmin(nodes[0].topLeftFront.z, topLeftFrontZ[0] - 1.0e10);

        // Update maximum bounds (bottom-right-back corner)
        nodes[0].botRightBack.x = fmax(nodes[0].botRightBack.x, botRightBackX[0] + 1.0e10);
        nodes[0].botRightBack.y = fmax(nodes[0].botRightBack.y, botRightBackY[0] + 1.0e10);
        nodes[0].botRightBack.z = fmax(nodes[0].botRightBack.z, botRightBackZ[0] + 1.0e10);

        // Release mutex
        atomicExch(mutex, 0);
    }
}