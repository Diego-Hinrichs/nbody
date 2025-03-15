#include "../../include/simulation/implementations/gpu/barnes_hut.cuh"

BarnesHut::BarnesHut(int numBodies, BodyDistribution dist, unsigned int seed)
    : SimulationBase(numBodies, dist, seed)
{
    nNodes = MAX_NODES;
    leafLimit = MAX_NODES - N_LEAF;
    h_bodies = new Body[nBodies];
    h_nodes = new Node[nNodes];

    CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, numBodies * sizeof(Body)));
    CHECK_CUDA_ERROR(cudaMalloc(&h_nodes, nNodes * sizeof(Node)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_tempBodies, numBodies * sizeof(Body)));
}

BarnesHut::~BarnesHut()
{
    delete[] h_bodies;
    delete[] h_nodes;
    CHECK_CUDA_ERROR(cudaFree(d_bodies));
    CHECK_CUDA_ERROR(cudaFree(d_nodes));
    CHECK_CUDA_ERROR(cudaFree(d_mutex));
    CHECK_CUDA_ERROR(cudaFree(d_tempBodies));
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
    ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_nodes, d_bodies, d_mutex, nBodies);
    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::constructOctree()
{
    // Measure execution time
    CudaTimer timer(metrics.octreeTimeMs);
    int blockSize = BLOCK_SIZE;
    ConstructOctTreeKernel<<<1, blockSize>>>(d_nodes, d_bodies, d_tempBodies, 0, nNodes, nBodies, leafLimit);
    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::computeForces()
{
    // Measure execution time
    CudaTimer timer(metrics.forceTimeMs);

    int blockSize = 32;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    ComputeForceKernel<<<gridSize, blockSize>>>(d_nodes, d_bodies, nNodes, nBodies, leafLimit);
    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::update()
{
    // Ensure initialization
    checkInitialization();
    CudaTimer timer(metrics.totalTimeMs);
    resetOctree();
    computeBoundingBox();
    constructOctree();
    computeForces();
    CHECK_LAST_CUDA_ERROR();
}