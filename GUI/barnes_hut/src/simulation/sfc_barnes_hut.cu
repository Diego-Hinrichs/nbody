#include "../../include/simulation/sfc_barnes_hut.cuh"

SFCBarnesHut::SFCBarnesHut(int numBodies, bool useSpaceFillingCurve)
    : BarnesHut(numBodies), useSFC(useSpaceFillingCurve), sorter(nullptr)
{

    if (useSFC)
    {
        sorter = new sfc::BodySorter(numBodies);
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
        return;
    }

    // Update bounds for SFC calculation
    updateBoundingBox();

    // Sort bodies using the SFC sorter
    sorter->sortBodies(d_bodies, minBound, maxBound);
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

    // Apply SFC ordering before constructing octree
    if (useSFC)
    {
        orderBodiesBySFC();
    }

    constructOctree();
    computeForces();
}