#include "../../include/simulation/implementations/gpu/sfc_variants.cuh"
#include "../../include/sfc/sfc_framework.cuh"
#include <iostream>
#include <algorithm>

SFCBarnesHut::SFCBarnesHut(int numBodies, bool useSpaceFillingCurve,
                           SFCOrderingMode initialOrderingMode, int initialReorderFreq,
                           BodyDistribution dist, unsigned int seed)
    : BarnesHut(numBodies, dist, seed),
      useSFC(useSpaceFillingCurve),
      bodySorter(nullptr),
      octantSorter(nullptr),
      d_orderedIndices(nullptr),
      d_octantIndices(nullptr),
      orderingMode(initialOrderingMode),
      reorderFrequency(initialReorderFreq),
      iterationCounter(0),
      curveType(sfc::CurveType::MORTON) // Default to Morton curve
{
    if (useSFC)
    {
        // Create sorters with appropriate curve type
        bodySorter = new sfc::BodySorter(numBodies, curveType);
        octantSorter = new sfc::OctantSorter(MAX_NODES, curveType);
    }

    // Initialize domain bounds to invalid values to force update
    minBound = Vector(INFINITY, INFINITY, INFINITY);
    maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

    std::cout << "SFC Barnes-Hut Simulation created with " << numBodies << " bodies." << std::endl;
    if (useSFC)
    {
        std::cout << "Space-Filling Curve ordering enabled with "
                  << (orderingMode == SFCOrderingMode::PARTICLES ? "particle" : "octant")
                  << " ordering and reorder frequency " << reorderFrequency << std::endl;
    }
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

    // Note: d_orderedIndices and d_octantIndices are managed by the sorters
}

void SFCBarnesHut::setCurveType(sfc::CurveType type)
{
    if (type != curveType)
    {
        curveType = type;

        // Update sorters with new curve type
        if (bodySorter)
            bodySorter->setCurveType(type);

        if (octantSorter)
            octantSorter->setCurveType(type);

        // Force reordering on next update
        iterationCounter = reorderFrequency;
    }
}

void SFCBarnesHut::updateBoundingBox()
{
    // Copy root node to get current bounding box
    Node rootNode;
    CHECK_CUDA_ERROR(cudaMemcpy(&rootNode, d_nodes, sizeof(Node), cudaMemcpyDeviceToHost));

    // Update domain bounds
    minBound = rootNode.topLeftFront;
    maxBound = rootNode.botRightBack;

    // Add a small padding to avoid edge cases
    double padding = std::max(1.0e10, (maxBound.x - minBound.x) * 0.01);
    minBound.x -= padding;
    minBound.y -= padding;
    minBound.z -= padding;
    maxBound.x += padding;
    maxBound.y += padding;
    maxBound.z += padding;
}

void SFCBarnesHut::orderBodiesBySFC()
{
    if (!useSFC || !bodySorter)
    {
        d_orderedIndices = nullptr;
        return;
    }

    // Update bounds for SFC calculation
    updateBoundingBox();

    // Get indices ordered by SFC
    d_orderedIndices = bodySorter->sortBodies(d_bodies, minBound, maxBound);
}

void SFCBarnesHut::orderOctantsBySFC(Node *nodes, int nNodes)
{
    if (!useSFC || !octantSorter)
    {
        d_octantIndices = nullptr;
        return;
    }

    // Update domain bounds
    updateBoundingBox();

    // Get indices ordered by SFC
    d_octantIndices = octantSorter->sortOctants(nodes, minBound, maxBound);
}

// Override constructOctree from the base class
void SFCBarnesHut::constructOctree()
{
    // Measure execution time
    CudaTimer timer(metrics.octreeTimeMs);

    // Determine if we're using octant ordering mode
    bool useOctantOrder = (useSFC && orderingMode == SFCOrderingMode::OCTANTS && d_octantIndices != nullptr);

    // Launch octree construction kernel with appropriate ordering
    BuildOptimizedOctTree(d_nodes, d_bodies, d_tempBodies,
                          d_orderedIndices, useSFC && orderingMode == SFCOrderingMode::PARTICLES,
                          d_octantIndices, useOctantOrder,
                          nNodes, nBodies, leafLimit);
    CHECK_LAST_CUDA_ERROR();
}

void SFCBarnesHut::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Execute the Barnes-Hut algorithm steps with SFC enhancement
    resetOctree();
    computeBoundingBox();

    // Apply SFC ordering based on mode and reordering frequency
    if (useSFC)
    {
        // Increment the iteration counter
        iterationCounter++;

        // Only reorder if it's time based on the frequency or this is the first iteration
        if (iterationCounter >= reorderFrequency || iterationCounter == 1)
        {
            // Reset the counter
            iterationCounter = 0;

            // Apply appropriate ordering based on the selected mode
            if (orderingMode == SFCOrderingMode::PARTICLES)
            {
                orderBodiesBySFC();
                d_octantIndices = nullptr; // Not using octant ordering
            }
            else // OCTANTS mode
            {
                // For octant ordering, we need to:
                // 1. Build tree first (with no ordering)
                d_orderedIndices = nullptr; // Not using particle ordering

                // Use base class's constructOctree method to build initial tree
                BarnesHut::constructOctree();

                // 2. Compute SFC codes for octants
                orderOctantsBySFC(d_nodes, nNodes);

                // 3. Now build the octree again, but with octant ordering
                constructOctree();
            }
        }
    }
    else
    {
        d_orderedIndices = nullptr;
        d_octantIndices = nullptr;
    }

    // Construct tree if not already built
    if (orderingMode != SFCOrderingMode::OCTANTS || !useSFC || iterationCounter != 0)
    {
        constructOctree();
    }

    // Compute forces
    computeForces();
}

/**
 * @brief CUDA kernel for direct force calculation between all body pairs, with SFC ordering support
 *
 * This kernel computes the gravitational forces between all pairs of bodies
 * using the Direct Sum approach (O(n²) complexity), with support for SFC-ordered indices.
 *
 * @param bodies Array of body structures
 * @param orderedIndices Array of SFC-ordered indices (can be nullptr if SFC not used)
 * @param useSFC Flag indicating whether to use SFC ordering
 * @param nBodies Number of bodies in the simulation
 */
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

SFCGPUDirectSum::SFCGPUDirectSum(int numBodies, bool useSpaceFillingCurve,
                                 int initialReorderFreq, BodyDistribution dist, unsigned int seed)
    : GPUDirectSum(numBodies, dist, seed),
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