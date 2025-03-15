#include "../../include/simulation/implementations/gpu/sfc_variants.cuh"
#include <iostream>

SFCBarnesHut::SFCBarnesHut(int numBodies, bool useSpaceFillingCurve, 
                         SFCOrderingMode initialOrderingMode, int initialReorderFreq,
                         BodyDistribution dist, unsigned int seed)
    : BarnesHut(numBodies, dist, seed),
      useSFC(useSpaceFillingCurve),
      minBound(),
      maxBound(),
      bodySorter(nullptr),
      octantSorter(nullptr),
      d_orderedIndices(nullptr),
      d_octantIndices(nullptr),
      orderingMode(initialOrderingMode),
      reorderFrequency(initialReorderFreq),
      iterationCounter(0),
      curveType(sfc::CurveType::MORTON)
{
    if (useSFC)
    {
        bodySorter = new sfc::BodySorter(numBodies, curveType);
        octantSorter = new sfc::OctantSorter(MAX_NODES, curveType);
    }

    // Initialize domain bounds to invalid values to force update
    minBound = Vector(INFINITY, INFINITY, INFINITY);
    maxBound = Vector(-INFINITY, -INFINITY, -INFINITY);

    std::cout << "SFC Barnes-Hut Simulation created with " << numBodies << " bodies." << std::endl;
    if (useSFC)
    {
        std::cout << "Space-Filling Curve ordering enabled with mode "
                  << (orderingMode == SFCOrderingMode::PARTICLES ? "PARTICLES" : "OCTANTS")
                  << " and reorder frequency " << reorderFrequency << std::endl;
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

    // The d_orderedIndices and d_octantIndices are managed by the sorters, so we don't free them here
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
    // We need to compute a bounding box for SFC ordering
    // We'll use the root node's bounding box from the octree
    
    // Copy the root node from device to host to get its bounds
    Node rootNode;
    CHECK_CUDA_ERROR(cudaMemcpy(&rootNode, d_nodes, sizeof(Node), cudaMemcpyDeviceToHost));
    
    // Update our cached bounds
    minBound = rootNode.topLeftFront;
    maxBound = rootNode.botRightBack;
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
    if (!useSFC || !octantSorter || orderingMode != SFCOrderingMode::OCTANTS)
    {
        d_octantIndices = nullptr;
        return;
    }

    // Update bounds for SFC calculation
    updateBoundingBox();

    // Get indices ordered by SFC
    d_octantIndices = octantSorter->sortOctants(nodes, minBound, maxBound);
}

void SFCBarnesHut::constructOctree()
{
    // Measure execution time
    CudaTimer timer(metrics.octreeTimeMs);

    // Apply SFC ordering to bodies if using PARTICLES mode
    if (useSFC && orderingMode == SFCOrderingMode::PARTICLES)
    {
        // Order bodies by SFC
        orderBodiesBySFC();
        
        // Use the ordered indices for construction
        int blockSize = BLOCK_SIZE;
        ConstructOctTreeKernel<<<1, blockSize>>>(d_nodes, d_bodies, d_tempBodies, 0,
                                               nNodes, nBodies, leafLimit);
        CHECK_LAST_CUDA_ERROR();
    }
    else
    {
        // Use base class implementation without ordering
        BarnesHut::constructOctree();
        
        // Apply SFC ordering to octants if using OCTANTS mode
        if (useSFC && orderingMode == SFCOrderingMode::OCTANTS)
        {
            orderOctantsBySFC(d_nodes, nNodes);
        }
    }
}

void SFCBarnesHut::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Apply SFC ordering when enabled and it's time to reorder
    if (useSFC)
    {
        // Increment the iteration counter
        iterationCounter++;

        // Only reorder if it's time based on the frequency or this is the first iteration
        if (iterationCounter >= reorderFrequency || iterationCounter == 1)
        {
            // Reset the counter
            iterationCounter = 0;
        }
        // Actual ordering is done in either constructOctree() or a dedicated step
    }

    // Execute the Barnes-Hut algorithm steps
    resetOctree();
    computeBoundingBox();
    constructOctree();
    
    // Swap bodies if needed for ordered data
    if (useSFC && orderingMode == SFCOrderingMode::PARTICLES)
    {
        swapBodyBuffers();
    }
    
    computeForces();
}