#include "../../include/simulation/barnes_hut.cuh"

// Updated external function declaration to match the new signature
extern "C" void BuildOptimizedOctTree(
    Node *d_nodes, Body *d_bodies, Body *d_tempBodies,
    int *orderedIndices, bool useSFC,
    int *octantIndices, bool useOctantOrder,
    int nNodes, int nBodies, int leafLimit);

BarnesHut::BarnesHut(int numBodies, BodyDistribution dist, unsigned int seed)
    : SimulationBase(numBodies, dist, seed)
{
    nNodes = MAX_NODES;
    leafLimit = MAX_NODES - N_LEAF;

    // Allocate host memory for nodes
    h_nodes = new Node[nNodes];

    // Allocate device memory
    CHECK_CUDA_ERROR(cudaMalloc(&d_nodes, nNodes * sizeof(Node)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bodiesBuffer, nBodies * sizeof(Body)));
}

BarnesHut::~BarnesHut()
{
    // Free host memory
    if (h_nodes)
    {
        delete[] h_nodes;
        h_nodes = nullptr;
    }

    // Free device memory
    if (d_nodes)
    {
        CHECK_CUDA_ERROR(cudaFree(d_nodes));
        d_nodes = nullptr;
    }

    if (d_mutex)
    {
        CHECK_CUDA_ERROR(cudaFree(d_mutex));
        d_mutex = nullptr;
    }

    if (d_tempBodies)
    {
        cudaFree(d_tempBodies);
        d_tempBodies = nullptr;
    }

    if (d_bodiesBuffer)
    {
        CHECK_CUDA_ERROR(cudaFree(d_bodiesBuffer));
        d_bodiesBuffer = nullptr;
    }
}

void BarnesHut::resetOctree()
{
    // Measure execution time
    CudaTimer timer(metrics.resetTimeMs);

    // Launch reset kernel
    int blockSize = BLOCK_SIZE;
    int gridSize = (nNodes + blockSize - 1) / blockSize;

    ResetKernel<<<gridSize, blockSize>>>(d_nodes, d_mutex, nNodes, nBodies);
    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::computeBoundingBox()
{
    // Measure execution time
    CudaTimer timer(metrics.bboxTimeMs);

    // Launch bounding box computation kernel with SFC support
    int blockSize = BLOCK_SIZE;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(
        d_nodes, d_bodies, getOrderedIndices(), isUsingSFC(), d_mutex, nBodies);
    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::constructOctree()
{
    // Measure execution time
    CudaTimer timer(metrics.octreeTimeMs);

    // Launch octree construction kernel with SFC support
    // Updated to use the new function signature with octant indices
    BuildOptimizedOctTree(d_nodes, d_bodies, d_tempBodies,
                          getOrderedIndices(), isUsingSFC(),
                          nullptr, false, // Base class doesn't use octant ordering
                          nNodes, nBodies, leafLimit);
    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::computeForces()
{
    // Measure execution time
    CudaTimer timer(metrics.forceTimeMs);

    // Launch force computation kernel with SFC support
    int blockSize = 256;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    ComputeForceKernel<<<gridSize, blockSize>>>(
        d_nodes, d_bodies, getOrderedIndices(), isUsingSFC(),
        nNodes, nBodies, leafLimit);
    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Execute the Barnes-Hut algorithm steps
    resetOctree();
    computeBoundingBox();
    constructOctree();
    computeForces();
}