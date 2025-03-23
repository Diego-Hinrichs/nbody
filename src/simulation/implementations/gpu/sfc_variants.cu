#include "../../include/simulation/implementations/gpu/sfc_variants.cuh"
#include "../../include/sfc/sfc_framework.cuh"
#include <iostream>
#include <algorithm>

__global__ void SFCDirectSumForceKernel(Body *bodies, int *orderedIndices, bool useSFC, int nBodies)
{
    // Reduced size shared memory arrays
    __shared__ Vector sharedPos[256];  // Reduced from BLOCK_SIZE
    __shared__ double sharedMass[256]; // Reduced from BLOCK_SIZE

    // Get global thread ID
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    // Get the real body index when using SFC ordering
    int realBodyIndex = (useSFC && orderedIndices != nullptr) ? orderedIndices[i] : i;

    // Load data only if the index is valid
    Vector myPos = Vector(0, 0, 0);
    Vector myVel = Vector(0, 0, 0);
    Vector myAcc = Vector(0, 0, 0);
    double myMass = 0.0;
    bool isDynamic = false;

    if (i < nBodies)
    {
        myPos = bodies[realBodyIndex].position;
        myVel = bodies[realBodyIndex].velocity;
        myMass = bodies[realBodyIndex].mass;
        isDynamic = bodies[realBodyIndex].isDynamic;
    }

    // Reduce computation with tiling approach
    const int tileSize = 256; // Smaller tile size for better occupancy

    // Process all tiles
    for (int tile = 0; tile < (nBodies + tileSize - 1) / tileSize; ++tile)
    {
        // Load this tile to shared memory
        int idx = tile * tileSize + tx;

        // Only load valid data to shared memory
        if (tx < tileSize)
        { // Ensure we don't exceed array size
            if (idx < nBodies)
            {
                // When using SFC ordering, get the real body index
                int tileBodyIndex = (useSFC && orderedIndices != nullptr) ? orderedIndices[idx] : idx;
                sharedPos[tx] = bodies[tileBodyIndex].position;
                sharedMass[tx] = bodies[tileBodyIndex].mass;
            }
            else
            {
                sharedPos[tx] = Vector(0, 0, 0);
                sharedMass[tx] = 0.0;
            }
        }

        __syncthreads();

        // Calculate force only for valid and dynamic bodies
        if (i < nBodies && isDynamic)
        {
            // Limit the loop to the real tile size
            int tileLimit = min(tileSize, nBodies - tile * tileSize);

            for (int j = 0; j < tileLimit; ++j)
            {
                int jBody = tile * tileSize + j;

                // Avoid self-interaction
                if (jBody != i)
                {
                    // Distance vector
                    double rx = sharedPos[j].x - myPos.x;
                    double ry = sharedPos[j].y - myPos.y;
                    double rz = sharedPos[j].z - myPos.z;

                    // Distance squared with softening
                    double distSqr = rx * rx + ry * ry + rz * rz + E * E;
                    double dist = sqrt(distSqr);

                    // Apply force only if above collision threshold
                    if (dist >= COLLISION_TH)
                    {
                        double forceMag = (GRAVITY * myMass * sharedMass[j]) / (dist * distSqr);

                        // Accumulate acceleration
                        myAcc.x += rx * forceMag / myMass;
                        myAcc.y += ry * forceMag / myMass;
                        myAcc.z += rz * forceMag / myMass;
                    }
                }
            }
        }

        __syncthreads();
    }

    // Update the body only if valid and dynamic
    if (i < nBodies && isDynamic)
    {
        // Save acceleration
        bodies[realBodyIndex].acceleration = myAcc;

        // Update velocity
        myVel.x += myAcc.x * DT;
        myVel.y += myAcc.y * DT;
        myVel.z += myAcc.z * DT;
        bodies[realBodyIndex].velocity = myVel;

        // Update position
        myPos.x += myVel.x * DT;
        myPos.y += myVel.y * DT;
        myPos.z += myVel.z * DT;
        bodies[realBodyIndex].position = myPos;
    }
}

SFCGPUDirectSum::SFCGPUDirectSum(
    int numBodies,
    bool useSpaceFillingCurve,
    int initialReorderFreq,
    BodyDistribution dist,
    unsigned int seed,
    MassDistribution massDist)
    : GPUDirectSum(numBodies, dist, seed, massDist),
      useSFC(useSpaceFillingCurve),
      sorter(nullptr),
      d_orderedIndices(nullptr),
      curveType(sfc::CurveType::MORTON), // Default to Morton curve
      reorderFrequency(initialReorderFreq),
      iterationCounter(0)
{
    if (useSFC)
    {
        sorter = new sfc::BodySorter(numBodies, curveType);
    }

    // Initialize domain bounds to invalid values to force update
    minBound = Vector(INFINITY, INFINITY, INFINITY);
    maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

    std::cout << "SFC GPU Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
    if (useSFC)
    {
        std::cout << "Space-Filling Curve ordering enabled with reorder frequency "
                  << reorderFrequency << std::endl;
    }
}

SFCGPUDirectSum::~SFCGPUDirectSum()
{
    if (sorter)
    {
        delete sorter;
        sorter = nullptr;
    }

    // The d_orderedIndices is managed by the sorter, so we don't free it here
}

void SFCGPUDirectSum::setCurveType(sfc::CurveType type)
{
    if (type != curveType)
    {
        curveType = type;

        // Update sorter with new curve type
        if (sorter)
            sorter->setCurveType(type);

        // Force reordering on next update
        iterationCounter = reorderFrequency;
    }
}

void SFCGPUDirectSum::updateBoundingBox()
{
    // We need to compute bounding box for SFC ordering
    // Using a simple kernel launch or by copying data to host

    // Temporary solution: allocate host memory and copy bodies
    Body *tempBodies = new Body[nBodies];
    CHECK_CUDA_ERROR(cudaMemcpy(tempBodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost));

    // Find min and max bounds
    minBound = Vector(INFINITY, INFINITY, INFINITY);
    maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

    for (int i = 0; i < nBodies; i++)
    {
        Vector pos = tempBodies[i].position;

        // Update minimum bounds
        minBound.x = std::min(minBound.x, pos.x);
        minBound.y = std::min(minBound.y, pos.y);
        minBound.z = std::min(minBound.z, pos.z);

        // Update maximum bounds
        maxBound.x = std::max(maxBound.x, pos.x);
        maxBound.y = std::max(maxBound.y, pos.y);
        maxBound.z = std::max(maxBound.z, pos.z);
    }

    // Add padding to avoid edge issues
    double padding = std::max(1.0e10, (maxBound.x - minBound.x) * 0.01);
    minBound.x -= padding;
    minBound.y -= padding;
    minBound.z -= padding;
    maxBound.x += padding;
    maxBound.y += padding;
    maxBound.z += padding;

    // Cleanup
    delete[] tempBodies;
}

void SFCGPUDirectSum::orderBodiesBySFC()
{
    if (!useSFC || !sorter)
    {
        d_orderedIndices = nullptr;
        return;
    }

    // Update bounds for SFC calculation
    updateBoundingBox();

    // Get indices ordered by SFC
    d_orderedIndices = sorter->sortBodies(d_bodies, minBound, maxBound);
}

void SFCGPUDirectSum::computeForces()
{
    // Measure execution time
    CudaTimer timer(metrics.forceTimeMs);

    // Launch kernel with SFC support
    int blockSize = 256; // Reduced block size for better occupancy
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    SFCDirectSumForceKernel<<<gridSize, blockSize>>>(d_bodies, d_orderedIndices, useSFC, nBodies);
    CHECK_LAST_CUDA_ERROR();
}

void SFCGPUDirectSum::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Reset unused metrics
    metrics.resetTimeMs = 0.0f;  // Not used in Direct Sum
    metrics.bboxTimeMs = 0.0f;   // Used internally in SFC ordering
    metrics.octreeTimeMs = 0.0f; // Not used in Direct Sum

    // Apply SFC ordering if enabled
    if (useSFC)
    {
        // Increment the iteration counter
        iterationCounter++;

        // Only reorder if it's time based on the frequency or this is the first iteration
        if (iterationCounter >= reorderFrequency || iterationCounter == 1)
        {
            // Reset the counter
            iterationCounter = 0;

            // Perform the SFC ordering
            orderBodiesBySFC();
        }
    }
    else
    {
        d_orderedIndices = nullptr;
    }

    // Compute forces and update positions
    computeForces();
}

__global__ void ApplyBodyOrderingKernel(Body *bodies, Body *buffer, int *indices, int numBodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBodies)
    {
        buffer[i] = bodies[indices[i]];
    }
}

__global__ void ApplyNodeOrderingKernel(Node *nodes, Node *buffer, int *indices, int numNodes)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNodes)
    {
        buffer[i] = nodes[indices[i]];
    }
}

SFCBarnesHut::SFCBarnesHut(
    int numBodies,
    bool enableSFC,
    SFCOrderingMode ordMode,
    int reorderFreq,
    BodyDistribution dist,
    unsigned int seed,
    MassDistribution massDist)
    : BarnesHut(numBodies, dist, seed, massDist),
      useSFC(enableSFC),
      curveType(sfc::CurveType::MORTON),
      orderingMode(ordMode),
      bodySorter(nullptr),
      octantSorter(nullptr),
      d_orderedBodyIndices(nullptr),
      d_orderedNodeIndices(nullptr),
      reorderFrequency(reorderFreq),
      iterationCounter(0)
{
    if (useSFC)
    {
        bodySorter = new sfc::BodySorter(numBodies, curveType);
        octantSorter = new sfc::OctantSorter(MAX_NODES, curveType);
    }

    minBound = Vector(INFINITY, INFINITY, INFINITY);
    maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

    std::cout << "SFC Barnes-Hut Simulation created with " << numBodies << " bodies." << std::endl;
    std::cout << "SFC Ordering: " << (useSFC ? "enabled" : "disabled");

    if (useSFC)
    {
        std::cout << ", Mode: " << (orderingMode == SFCOrderingMode::PARTICLES ? "particles" : "octants");
        std::cout << ", Curve: " << (curveType == sfc::CurveType::MORTON ? "Morton" : "Hilbert");
        std::cout << ", Reorder Frequency: " << reorderFrequency;
    }
    std::cout << std::endl;
}

SFCBarnesHut::~SFCBarnesHut()
{
    if (bodySorter)
    {
        delete bodySorter;
        bodySorter = nullptr;
    }

    if (octantSorter)
    {
        delete octantSorter;
        octantSorter = nullptr;
    }
}

void SFCBarnesHut::setCurveType(sfc::CurveType type)
{
    if (type != curveType)
    {
        curveType = type;

        if (bodySorter)
            bodySorter->setCurveType(type);

        if (octantSorter)
            octantSorter->setCurveType(type);

        iterationCounter = reorderFrequency;
    }
}

void SFCBarnesHut::updateBoundingBox()
{
    Body *tempBodies = new Body[nBodies];
    CHECK_CUDA_ERROR(cudaMemcpy(tempBodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost));

    minBound = Vector(INFINITY, INFINITY, INFINITY);
    maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

    for (int i = 0; i < nBodies; i++)
    {
        Vector pos = tempBodies[i].position;

        minBound.x = std::min(minBound.x, pos.x);
        minBound.y = std::min(minBound.y, pos.y);
        minBound.z = std::min(minBound.z, pos.z);

        maxBound.x = std::max(maxBound.x, pos.x);
        maxBound.y = std::max(maxBound.y, pos.y);
        maxBound.z = std::max(maxBound.z, pos.z);
    }

    double padding = std::max(1.0e10, (maxBound.x - minBound.x) * 0.01);
    minBound.x -= padding;
    minBound.y -= padding;
    minBound.z -= padding;
    maxBound.x += padding;
    maxBound.y += padding;
    maxBound.z += padding;

    delete[] tempBodies;
}

void SFCBarnesHut::orderBodiesBySFC()
{
    if (!useSFC || !bodySorter)
        return;

    updateBoundingBox();

    d_orderedBodyIndices = bodySorter->sortBodies(d_bodies, minBound, maxBound);

    Body *d_tempBodies = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_tempBodies, nBodies * sizeof(Body)));

    int blockSize = 256;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    ApplyBodyOrderingKernel<<<gridSize, blockSize>>>(d_bodies, d_tempBodies, d_orderedBodyIndices, nBodies);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, d_tempBodies, nBodies * sizeof(Body), cudaMemcpyDeviceToDevice));

    CHECK_CUDA_ERROR(cudaFree(d_tempBodies));
}

void SFCBarnesHut::orderOctantsBySFC()
{
    if (!useSFC || !octantSorter || orderingMode != SFCOrderingMode::OCTANTS)
        return;

    d_orderedNodeIndices = octantSorter->sortOctants(d_nodes, minBound, maxBound);

    Node *d_tempNodes = nullptr;
    CHECK_CUDA_ERROR(cudaMalloc(&d_tempNodes, nNodes * sizeof(Node)));

    int blockSize = 256;
    int gridSize = (nNodes + blockSize - 1) / blockSize;

    ApplyNodeOrderingKernel<<<gridSize, blockSize>>>(d_nodes, d_tempNodes, d_orderedNodeIndices, nNodes);
    CHECK_LAST_CUDA_ERROR();

    CHECK_CUDA_ERROR(cudaMemcpy(d_nodes, d_tempNodes, nNodes * sizeof(Node), cudaMemcpyDeviceToDevice));

    CHECK_CUDA_ERROR(cudaFree(d_tempNodes));
}

void SFCBarnesHut::resetOctree()
{
    BarnesHut::resetOctree();
}

void SFCBarnesHut::computeBoundingBox()
{
    BarnesHut::computeBoundingBox();

    if (useSFC)
    {
        Node rootNode;
        CHECK_CUDA_ERROR(cudaMemcpy(&rootNode, d_nodes, sizeof(Node), cudaMemcpyDeviceToHost));

        minBound = rootNode.topLeftFront;
        maxBound = rootNode.botRightBack;
    }
}

void SFCBarnesHut::constructOctree()
{
    BarnesHut::constructOctree();

    if (useSFC && orderingMode == SFCOrderingMode::OCTANTS)
    {
        orderOctantsBySFC();
    }
}

void SFCBarnesHut::computeForces()
{
    BarnesHut::computeForces();
}

void SFCBarnesHut::update()
{
    checkInitialization();

    CudaTimer timer(metrics.totalTimeMs);

    if (useSFC)
    {
        iterationCounter++;

        if (iterationCounter >= reorderFrequency || iterationCounter == 1)
        {
            iterationCounter = 0;

            if (orderingMode == SFCOrderingMode::PARTICLES)
            {
                orderBodiesBySFC();
            }
        }
    }

    resetOctree();
    constructOctree();
    computeForces();

    CHECK_LAST_CUDA_ERROR();
}