#ifndef BARNES_HUT_KERNEL_H_
#define BARNES_HUT_KERNEL_H_
#include <stdio.h>
#include <stdlib.h>
#include "constants.h"
#include "err.h"

typedef struct
{
    double x;
    double y;
    double z;
} Vector;

typedef struct
{
    bool isDynamic;
    double mass;
    double radius;
    Vector position;
    Vector velocity;
    Vector acceleration;
} Body;

typedef struct
{
    Vector topLeftFront;
    Vector botRightBack;
    Vector centerMass;
    double totalMass;
    bool isLeaf;
    int start;
    int end;
} Node;

struct UpdateTimes
{
    float resetTimeMs;
    float bboxTimeMs;
    float octreeTimeMs;
    float forceTimeMs;
};

// Kernel declarations
__global__ void ResetKernel(Node *node, int *mutex, int nNodes, int nBodies);
__global__ void ComputeBoundingBoxKernel(Node *node, Body *bodies, int *mutex, int nBodies);
__global__ void ComputeForceKernel(Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit);
__global__ void ConstructOctTreeKernel(Node *node, Body *bodies, Body *bodiesBuffer, int depth, int nNodes, int nBodies, int leafLimit);

class BarnesHutCuda
{
protected:
    int nBodies;
    int nNodes;
    int leafLimit;

    Body *h_b;
    Node *h_node;

    Body *d_b;
    Body *d_b_buffer;
    Node *d_node;
    int *d_mutex;

    void initRandomBodies()
    {
        // Seed random number generator
        srand(time(NULL));

        double maxDistance = MAX_DIST;
        double minDistance = MIN_DIST;
        Vector centerPos = {CENTERX, CENTERY, CENTERZ};

        // Generate random bodies in a spherical distribution
        for (int i = 0; i < nBodies; ++i)
        {
            // Generate random spherical coordinates
            double u = rand() / (double)RAND_MAX; // For theta
            double v = rand() / (double)RAND_MAX; // For phi
            double theta = 2.0 * M_PI * u;
            double phi = acos(2.0 * v - 1.0);

            // Random radius between min and max distance
            double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

            // Convert to Cartesian coordinates
            double x = centerPos.x + radius * sin(phi) * cos(theta);
            double y = centerPos.y + radius * sin(phi) * sin(theta);
            double z = centerPos.z + radius * cos(phi);

            // Setup body properties
            h_b[i].isDynamic = true;
            h_b[i].mass = SUN_MASS;
            h_b[i].radius = SUN_DIA;
            h_b[i].position = {x, y, z};
            h_b[i].velocity = {0.0, 0.0, 0.0};
            h_b[i].acceleration = {0.0, 0.0, 0.0};
        }
    }

public:
    BarnesHutCuda(int n) : nBodies(n)
    {
        nNodes = MAX_NODES;
        leafLimit = MAX_NODES - N_LEAF;

        // Allocate host memory
        h_b = new Body[nBodies];
        h_node = new Node[nNodes];

        // Allocate device memory
        CHECK_CUDA_ERROR(cudaMalloc(&d_b, nBodies * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_node, nNodes * sizeof(Node)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b_buffer, nBodies * sizeof(Body)));
    }

    ~BarnesHutCuda()
    {
        // Free host memory
        delete[] h_b;
        delete[] h_node;

        // Free device memory
        CHECK_CUDA_ERROR(cudaFree(d_b));
        CHECK_CUDA_ERROR(cudaFree(d_node));
        CHECK_CUDA_ERROR(cudaFree(d_mutex));
        CHECK_CUDA_ERROR(cudaFree(d_b_buffer));
    }

    void resetCUDA()
    {
        int blockSize = BLOCK_SIZE;
        int gridSize = (nNodes + blockSize - 1) / blockSize;
        ResetKernel<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);
    }

    void computeBoundingBoxCUDA()
    {
        int blockSize = BLOCK_SIZE;
        int gridSize = (nBodies + blockSize - 1) / blockSize;
        ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);
    }

    void constructOctreeCUDA()
    {
        int blockSize = BLOCK_SIZE;
        ConstructOctTreeKernel<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);
    }

    void computeForceCUDA()
    {
        int blockSize = 256;
        int gridSize = (nBodies + blockSize - 1) / blockSize;
        ComputeForceKernel<<<gridSize, blockSize>>>(d_node, d_b, nNodes, nBodies, leafLimit);
    }

    void update()
    {
        resetCUDA();
        computeBoundingBoxCUDA();
        constructOctreeCUDA();
        computeForceCUDA();
        CHECK_LAST_CUDA_ERROR();
    }

    void setup()
    {
        initRandomBodies();
        CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_node, h_node, sizeof(Node) * nNodes, cudaMemcpyHostToDevice));
    }

    void readDeviceBodies()
    {
        CHECK_CUDA_ERROR(cudaMemcpy(h_b, d_b, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));
    }

    Body *getBodies()
    {
        return h_b;
    }
};

#endif // BARNES_HUT_KERNEL_H_