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
    float resetTimeMs;  // tiempo en ms de resetCUDA()
    float bboxTimeMs;   // tiempo en ms de computeBoundingBoxCUDA()
    float octreeTimeMs; // tiempo en ms de constructOctreeCUDA()
    float forceTimeMs;  // tiempo en ms de computeForceCUDA()
};

__global__ void ResetKernel(Node *node, int *mutex, int nNodes, int nBodies);
__global__ void ComputeBoundingBoxKernel(Node *node, Body *bodies, int *mutex, int nBodies);
__global__ void ConstructOctTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit);
__global__ void ComputeForceKernel(Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit);

class BarnesHutCuda
{
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
        srand(time(NULL));
        double maxDistance = MAX_DIST;
        double minDistance = MIN_DIST;
        Vector centerPos = {CENTERX, CENTERY, CENTERZ};
        for (int i = 0; i < nBodies; ++i)
        {
            double u = rand() / (double)RAND_MAX; // Para theta
            double v = rand() / (double)RAND_MAX; // Para phi
            double theta = 2.0 * M_PI * u;
            double phi = acos(2.0 * v - 1.0);
            double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;
            double x = centerPos.x + radius * sin(phi) * cos(theta);
            double y = centerPos.y + radius * sin(phi) * sin(theta);
            double z = centerPos.z + radius * cos(phi);
            Vector position = {x, y, z};
            h_b[i].isDynamic = true;
            h_b[i].mass = SUN_MASS;
            h_b[i].radius = SUN_DIA;
            h_b[i].position = position;
            h_b[i].velocity = {0.0, 0.0, 0.0};
            h_b[i].acceleration = {0.0, 0.0, 0.0};
        }
    }

    void setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration)
    {
        h_b[i].isDynamic = isDynamic;
        h_b[i].mass = mass;
        h_b[i].radius = radius;
        h_b[i].position = position;
        h_b[i].velocity = velocity;
        h_b[i].acceleration = acceleration;
    }

    void resetCUDA()
    {
        int blockSize = BLOCK_SIZE;
        dim3 gridSize = ceil((float)nNodes / blockSize);
        ResetKernel<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);
    }
    void computeBoundingBoxCUDA()
    {
        int blockSize = BLOCK_SIZE;
        dim3 gridSize = ceil((float)nBodies / blockSize);
        ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);
    }
    void constructOctreeCUDA()
    {
        int blockSize = BLOCK_SIZE;
        ConstructOctTreeKernel<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);
    }
    void computeForceCUDA()
    {
        int blockSize = 32;
        dim3 gridSize = ceil((float)nBodies / blockSize);
        ComputeForceKernel<<<gridSize, blockSize>>>(d_node, d_b, nNodes, nBodies, leafLimit);
    }

public:
    BarnesHutCuda(int n) : nBodies(n)
    {
        nNodes = MAX_NODES;
        leafLimit = MAX_NODES - N_LEAF;
        h_b = new Body[nBodies];
        h_node = new Node[nNodes];

        CHECK_CUDA_ERROR(cudaMalloc(&d_b, n * sizeof(Body)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_node, nNodes * sizeof(Node)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));
        CHECK_CUDA_ERROR(cudaMalloc(&d_b_buffer, n * sizeof(Body)));
    };

    ~BarnesHutCuda()
    {
        delete[] h_b;
        delete[] h_node;
        CHECK_CUDA_ERROR(cudaFree(d_b));
        CHECK_CUDA_ERROR(cudaFree(d_node));
        CHECK_CUDA_ERROR(cudaFree(d_mutex));
        CHECK_CUDA_ERROR(cudaFree(d_b_buffer));
    };

    void update()
    {
        resetCUDA();
        computeBoundingBoxCUDA();
        constructOctreeCUDA();
        computeForceCUDA();
        CHECK_LAST_CUDA_ERROR();
    };

    void setup()
    {
        initRandomBodies();
        CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
        CHECK_CUDA_ERROR(cudaMemcpy(d_node, h_node, sizeof(Node) * nNodes, cudaMemcpyHostToDevice));
    };

    void readDeviceBodies()
    {
        CHECK_CUDA_ERROR(cudaMemcpy(h_b, d_b, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));
    };

    Body *getBodies()
    {
        return h_b;
    };
};

#endif