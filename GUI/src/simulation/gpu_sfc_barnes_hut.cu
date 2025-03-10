#include "../../include/simulation/gpu_sfc_barnes_hut.cuh"
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