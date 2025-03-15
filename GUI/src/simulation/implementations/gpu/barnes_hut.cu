#include "../../include/simulation/implementations/gpu/barnes_hut.cuh"

BarnesHut::BarnesHut(int numBodies, BodyDistribution dist, unsigned int seed)
    : SimulationBase(numBodies, dist, seed)
{
    nNodes = MAX_NODES;
    leafLimit = MAX_NODES - N_LEAF;
    h_nodes = new Node[nNodes];

    CHECK_CUDA_ERROR(cudaMalloc(&d_nodes, nNodes * sizeof(Node)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_mutex, nNodes * sizeof(int)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_bodiesBuffer, numBodies * sizeof(Body))); // for octree construction
}

BarnesHut::~BarnesHut()
{
    if (h_nodes)
    {
        delete[] h_nodes;
        h_nodes = nullptr;
    }

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
    dim3 gridSize = ceil((float)nNodes / blockSize);
    ResetKernel<<<gridSize, blockSize>>>(d_nodes, d_mutex, nNodes, nBodies);

    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::computeBoundingBox()
{
    // Measure execution time
    CudaTimer timer(metrics.bboxTimeMs);

    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_nodes, d_bodies, d_mutex, nBodies);

    CHECK_LAST_CUDA_ERROR();
}

void BarnesHut::constructOctree()
{
    int blockSize = BLOCK_SIZE;
    ConstructOctTreeKernel<<<1, blockSize>>>(d_nodes, d_bodies, d_bodiesBuffer, 0, nNodes, nBodies, leafLimit);
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

void BarnesHut::swapBodyBuffers()
{
    Body *temp = d_bodies;
    d_bodies = d_tempBodies;
    d_tempBodies = temp;
}

void BarnesHut::update()
{
    checkInitialization();

    CudaTimer timer(metrics.totalTimeMs);

    // Reset the octree
    {
        resetOctree();
        cudaDeviceSynchronize();
    }

    // Compute bounding box
    {
        computeBoundingBox();
        cudaDeviceSynchronize();
    }

    // Construct octree
    {
        constructOctree();
        cudaDeviceSynchronize();

    }

    // Compute forces
    {
        computeForces();
        cudaDeviceSynchronize();
    }
}