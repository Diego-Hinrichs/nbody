
#include "../../include/simulation/gpu_sfc_barnes_hut.cuh"

__global__ void ComputeOctantMortonCodesKernel(Node *nodes, uint64_t *mortonCodes, int *indices,
                                               int nNodes, Vector minBound, Vector maxBound)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nNodes)
    {
        // Skip empty nodes
        if (nodes[idx].start == -1 && nodes[idx].end == -1)
        {
            mortonCodes[idx] = 0; // Assign lowest priority to empty nodes
        }
        else
        {
            // Use node center for Morton code calculation
            Vector center = nodes[idx].getCenter();
            mortonCodes[idx] = sfc::positionToMorton(center, minBound, maxBound);
        }
        indices[idx] = idx; // Initialize with sequential indices
    }
}

SFCBarnesHut::SFCBarnesHut(int numBodies, bool useSpaceFillingCurve,
                           SFCOrderingMode initialOrderingMode, int initialReorderFreq,
                           BodyDistribution dist, unsigned int seed)
    : BarnesHut(numBodies, dist, seed),
      useSFC(useSpaceFillingCurve),
      sorter(nullptr),
      d_orderedIndices(nullptr),
      d_octantIndices(nullptr),
      orderingMode(initialOrderingMode),
      reorderFrequency(initialReorderFreq),
      iterationCounter(0)
{
    if (useSFC)
    {
        sorter = new sfc::BodySorter(numBodies);

        // Allocate device memory for octant indices
        CHECK_CUDA_ERROR(cudaMalloc(&d_octantIndices, MAX_NODES * sizeof(int)));
    }

    // Initialize domain bounds to invalid values to force update
    minBound = Vector(INFINITY, INFINITY, INFINITY);
    maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);
}

SFCBarnesHut::~SFCBarnesHut()
{
    if (sorter)
    {
        delete sorter;
        sorter = nullptr;
    }

    // Clean up octant indices
    if (d_octantIndices)
    {
        CHECK_CUDA_ERROR(cudaFree(d_octantIndices));
        d_octantIndices = nullptr;
    }

    // We don't need to free d_orderedIndices as it's managed by the sorter
}

void SFCBarnesHut::updateBoundingBox()
{
    // Copy root node to get current bounding box
    Node rootNode;
    CHECK_CUDA_ERROR(cudaMemcpy(&rootNode, d_nodes, sizeof(Node), cudaMemcpyDeviceToHost));

    // Update domain bounds
    minBound = rootNode.topLeftFront;
    maxBound = rootNode.botRightBack;
}

void SFCBarnesHut::orderBodiesBySFC()
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

void SFCBarnesHut::orderOctantsBySFC(Node *nodes, int nNodes)
{
    if (!useSFC || !d_octantIndices)
    {
        return;
    }

    // Update domain bounds
    updateBoundingBox();

    // Allocate temporary device memory for Morton codes
    uint64_t *d_mortonCodes;
    CHECK_CUDA_ERROR(cudaMalloc(&d_mortonCodes, nNodes * sizeof(uint64_t)));

    // Compute Morton codes for all octants
    int blockSize = BLOCK_SIZE;
    int gridSize = (nNodes + blockSize - 1) / blockSize;

    ComputeOctantMortonCodesKernel<<<gridSize, blockSize>>>(
        nodes, d_mortonCodes, d_octantIndices, nNodes, minBound, maxBound);
    CHECK_LAST_CUDA_ERROR();

    // Sort octants by Morton code using Thrust
    thrust::device_ptr<uint64_t> thrust_codes(d_mortonCodes);
    thrust::device_ptr<int> thrust_indices(d_octantIndices);
    thrust::sort_by_key(thrust::device, thrust_codes, thrust_codes + nNodes, thrust_indices);

    // Free temporary memory
    CHECK_CUDA_ERROR(cudaFree(d_mortonCodes));
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

                // 2. Compute Morton codes for octants
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
