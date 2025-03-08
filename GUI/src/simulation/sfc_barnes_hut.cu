
#include "../../include/simulation/sfc_barnes_hut.cuh"

SFCBarnesHut::SFCBarnesHut(int numBodies, bool useSpaceFillingCurve)
    : BarnesHut(numBodies), useSFC(useSpaceFillingCurve), sorter(nullptr), d_orderedIndices(nullptr)
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

    // No necesitamos liberar d_orderedIndices porque es propiedad del sorter
    // y se libera en su destructor
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

    // Obtener los índices ordenados por SFC (ya no reordena los cuerpos)
    d_orderedIndices = sorter->sortBodies(d_bodies, minBound, maxBound);
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
    else
    {
        d_orderedIndices = nullptr;
    }

    // El nodo raíz del octree debe conocer si se están usando índices ordenados
    // y cuáles son esos índices
    // Esto se hace pasando esta información a los kernels constructOctree y computeForces
    constructOctree();
    computeForces();
}