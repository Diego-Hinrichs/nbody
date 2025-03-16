#include "../../include/simulation/implementations/gpu/barnes_hut.cuh"

BarnesHut::BarnesHut(int numBodies, BodyDistribution dist, unsigned int seed)
    : SimulationBase(numBodies, dist, seed)
{
    // Make sure we're using reasonable values
    if (numBodies < 1) numBodies = 1;
    
    nNodes = MAX_NODES;
    leafLimit = MAX_NODES - N_LEAF;

    printf("BarnesHut::BarnesHut: Creating simulation with numBodies=%d, nNodes=%d\n", 
           numBodies, nNodes);

    // Allocate host memory for nodes
    h_nodes = new Node[nNodes];
    
    // Allocate device memory with error checking
    cudaError_t err;
    
    err = cudaMalloc(&d_nodes, nNodes * sizeof(Node));
    if (err != cudaSuccess) {
        printf("CUDA Error allocating d_nodes: %s\n", cudaGetErrorString(err));
        // Handle error - you might want to throw an exception here
    }
    
    err = cudaMalloc(&d_mutex, nNodes * sizeof(int));
    if (err != cudaSuccess) {
        printf("CUDA Error allocating d_mutex: %s\n", cudaGetErrorString(err));
        // Handle error - you might want to throw an exception here
    }
    
    err = cudaMalloc(&d_bodiesBuffer, nBodies * sizeof(Body));
    if (err != cudaSuccess) {
        printf("CUDA Error allocating d_bodiesBuffer: %s\n", cudaGetErrorString(err));
        // Handle error - you might want to throw an exception here
    }
    
    // Print the allocated addresses
    printf("BarnesHut::BarnesHut: d_nodes=%p, d_mutex=%p, d_bodiesBuffer=%p\n", 
           d_nodes, d_mutex, d_bodiesBuffer);
}

BarnesHut::~BarnesHut()
{
    delete[] h_nodes;
     // Free resources specific to BarnesHut
    if (d_nodes) {
        CHECK_CUDA_ERROR(cudaFree(d_nodes));
        d_nodes = nullptr;
    }
    
    if (d_mutex) {
        CHECK_CUDA_ERROR(cudaFree(d_mutex));
        d_mutex = nullptr;
    }
    
    if (d_bodiesBuffer) {
        CHECK_CUDA_ERROR(cudaFree(d_bodiesBuffer));
        d_bodiesBuffer = nullptr;
    }
}

void BarnesHut::resetOctree()
{
    CudaTimer timer(metrics.resetTimeMs);
    
    // Add debugging prints
    // printf("BarnesHut::resetOctree: nNodes=%d, nBodies=%d\n", nNodes, nBodies);
    // printf("BarnesHut::resetOctree: d_nodes=%p, d_mutex=%p\n", d_nodes, d_mutex);
    
    int blockSize = BLOCK_SIZE;
    int numBlocks = (nNodes + blockSize - 1) / blockSize;
    // printf("BarnesHut::resetOctree: blockSize=%d, numBlocks=%d\n", blockSize, numBlocks);
    
    // Launch kernel with corrected grid size
    dim3 gridSize(numBlocks);
    ResetKernel<<<gridSize, blockSize>>>(d_nodes, d_mutex, nNodes, nBodies);
    
    // Check for errors immediately after kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error during ResetKernel: %s\n", cudaGetErrorString(err));
    }
    
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
    int blockSize = BLOCK_SIZE;
    ConstructOctTreeKernel<<<1, blockSize>>>(d_nodes, d_bodies, d_bodiesBuffer, 0, nNodes, nBodies, leafLimit);
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