#include "../../include/simulation/implementations/cpu/sfc_variants.hpp"
#include "../../include/sfc/sfc_framework.cuh"
#include <iostream>
#include <limits>
#include <algorithm>

SFCCPUDirectSum::SFCCPUDirectSum(int numBodies, bool useParallelization, int threads,
                                 bool enableSFC, int reorderFreq,
                                 BodyDistribution dist, unsigned int seed)
    : CPUDirectSum(numBodies, useParallelization, threads, dist, seed),
      useSFC(enableSFC),
      reorderFrequency(reorderFreq),
      iterationCounter(0)
{
    // Set initial bounds to invalid values to force computation
    minBound = Vector(std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max());
    maxBound = Vector(std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest());

    // Initialize Morton code and ordering structures
    if (useSFC)
    {
        mortonCodes.resize(numBodies);
        orderedIndices.resize(numBodies);

        // Initialize indices with identity mapping
        for (int i = 0; i < numBodies; i++)
        {
            orderedIndices[i] = i;
        }
    }

    // Log configuration
    std::cout << "CPU SFC Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
    if (useParallelization)
    {
        std::cout << "OpenMP enabled with " << threads << " threads." << std::endl;
    }
    if (useSFC)
    {
        std::cout << "Space-Filling Curve ordering enabled with reorder frequency "
                  << reorderFrequency << std::endl;
    }
}

SFCCPUDirectSum::~SFCCPUDirectSum()
{
    // Vector data will be automatically cleaned up
}

void SFCCPUDirectSum::computeBoundingBox()
{
    // Make sure bodies are on host side
    copyBodiesFromDevice();

    // Reset bounds
    minBound = Vector(std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max(),
                      std::numeric_limits<double>::max());
    maxBound = Vector(std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest(),
                      std::numeric_limits<double>::lowest());

    // Compute bounds - use OpenMP if enabled
    bool useOpenMP = this->useOpenMP;
    int numThreads = this->numThreads;

    if (useOpenMP)
    {
        // Set the number of threads
        omp_set_num_threads(numThreads);

        // Shared variables for reduction
        Vector localMin[64]; // Using fixed size to avoid dynamic allocation inside OpenMP
        Vector localMax[64];

        int actualThreads = numThreads;
        if (actualThreads > 64)
            actualThreads = 64; // Limit to array size

        // Initialize local bounds
        for (int i = 0; i < actualThreads; i++)
        {
            localMin[i] = Vector(std::numeric_limits<double>::max(),
                                 std::numeric_limits<double>::max(),
                                 std::numeric_limits<double>::max());
            localMax[i] = Vector(std::numeric_limits<double>::lowest(),
                                 std::numeric_limits<double>::lowest(),
                                 std::numeric_limits<double>::lowest());
        }

#pragma omp parallel num_threads(actualThreads)
        {
            int tid = omp_get_thread_num();

#pragma omp for
            for (int i = 0; i < nBodies; i++)
            {
                // Update local min coords
                localMin[tid].x = std::min(localMin[tid].x, h_bodies[i].position.x);
                localMin[tid].y = std::min(localMin[tid].y, h_bodies[i].position.y);
                localMin[tid].z = std::min(localMin[tid].z, h_bodies[i].position.z);

                // Update local max coords
                localMax[tid].x = std::max(localMax[tid].x, h_bodies[i].position.x);
                localMax[tid].y = std::max(localMax[tid].y, h_bodies[i].position.y);
                localMax[tid].z = std::max(localMax[tid].z, h_bodies[i].position.z);
            }
        }

        // Combine results from all threads
        for (int i = 0; i < actualThreads; i++)
        {
            minBound.x = std::min(minBound.x, localMin[i].x);
            minBound.y = std::min(minBound.y, localMin[i].y);
            minBound.z = std::min(minBound.z, localMin[i].z);

            maxBound.x = std::max(maxBound.x, localMax[i].x);
            maxBound.y = std::max(maxBound.y, localMax[i].y);
            maxBound.z = std::max(maxBound.z, localMax[i].z);
        }
    }
    else
    {
        // Single-threaded computation
        for (int i = 0; i < nBodies; i++)
        {
            // Update minimum bounds
            minBound.x = std::min(minBound.x, h_bodies[i].position.x);
            minBound.y = std::min(minBound.y, h_bodies[i].position.y);
            minBound.z = std::min(minBound.z, h_bodies[i].position.z);

            // Update maximum bounds
            maxBound.x = std::max(maxBound.x, h_bodies[i].position.x);
            maxBound.y = std::max(maxBound.y, h_bodies[i].position.y);
            maxBound.z = std::max(maxBound.z, h_bodies[i].position.z);
        }
    }

    // Add some padding to avoid edge cases
    double padding = std::max(1.0e10, (maxBound.x - minBound.x) * 0.01);
    minBound.x -= padding;
    minBound.y -= padding;
    minBound.z -= padding;
    maxBound.x += padding;
    maxBound.y += padding;
    maxBound.z += padding;
}

void SFCCPUDirectSum::orderBodiesBySFC()
{
    if (!useSFC)
    {
        return;
    }

    // Make sure bodies are on host side
    copyBodiesFromDevice();

    // First, compute the domain's bounding box
    computeBoundingBox();

    // Compute Morton codes for each body
    bool useOpenMP = this->useOpenMP;
    int numThreads = this->numThreads;

    if (useOpenMP)
    {
        omp_set_num_threads(numThreads);

#pragma omp parallel for
        for (int i = 0; i < nBodies; i++)
        {
            mortonCodes[i] = sfc::MortonCurve().positionToCode(h_bodies[i].position, minBound, maxBound);
            orderedIndices[i] = i;
        }
    }
    else
    {
        for (int i = 0; i < nBodies; i++)
        {
            mortonCodes[i] = sfc::MortonCurve().positionToCode(h_bodies[i].position, minBound, maxBound);
            orderedIndices[i] = i;
        }
    }

    // Sort indices based on Morton codes
    // Use parallel sort if OpenMP is enabled and we have enough bodies
    if (useOpenMP && nBodies > 10000)
    {
        // Use a serial sort for simplicity and reliability
        // Parallel sorting would require more complex implementation
        std::sort(orderedIndices.begin(), orderedIndices.end(),
                  [this](int a, int b)
                  {
                      return mortonCodes[a] < mortonCodes[b];
                  });
    }
    else
    {
        // Sequential sort
        std::sort(orderedIndices.begin(), orderedIndices.end(),
                  [this](int a, int b)
                  {
                      return mortonCodes[a] < mortonCodes[b];
                  });
    }
}

void SFCCPUDirectSum::computeForces()
{
    // Measure execution time
    CudaTimer timer(metrics.forceTimeMs);

    // Ensure bodies are on the host
    copyBodiesFromDevice();

    // Save a reference to whether we're using SFC ordering
    const bool enableSFC = useSFC && !orderedIndices.empty();
    const auto &orderIndices = orderedIndices; // Reference to avoid copies

    // Compute forces using direct summation (O(n²) complexity) with SFC support
    if (useOpenMP)
    {
        // Set the number of threads
        omp_set_num_threads(numThreads);

#pragma omp parallel for
        for (int i = 0; i < nBodies; i++)
        {
            // Get the actual body index when using SFC
            int bodyIdx = enableSFC ? orderIndices[i] : i;

            // Skip non-dynamic bodies
            if (!h_bodies[bodyIdx].isDynamic)
                continue;

            // Reset acceleration
            h_bodies[bodyIdx].acceleration = Vector(0.0, 0.0, 0.0);

            // Compute force from all other bodies
            for (int j = 0; j < nBodies; j++)
            {
                // Get the actual other body index when using SFC
                int otherIdx = enableSFC ? orderIndices[j] : j;

                if (bodyIdx == otherIdx)
                    continue; // Skip self-interaction

                // Vector from body i to body j
                Vector r = h_bodies[otherIdx].position - h_bodies[bodyIdx].position;

                // Distance calculation with softening
                double distSqr = r.lengthSquared() + (E * E);
                double dist = sqrt(distSqr);

                // Skip if bodies are too close (collision)
                if (dist < COLLISION_TH)
                    continue;

                // Gravitational force: G * m1 * m2 / r^3 * r_vector
                double forceMag = GRAVITY * h_bodies[bodyIdx].mass * h_bodies[otherIdx].mass / (distSqr * dist);

                // Update acceleration (F = ma, so a = F/m)
                h_bodies[bodyIdx].acceleration.x += (r.x * forceMag) / h_bodies[bodyIdx].mass;
                h_bodies[bodyIdx].acceleration.y += (r.y * forceMag) / h_bodies[bodyIdx].mass;
                h_bodies[bodyIdx].acceleration.z += (r.z * forceMag) / h_bodies[bodyIdx].mass;
            }

            // Update velocity (Euler integration)
            h_bodies[bodyIdx].velocity.x += h_bodies[bodyIdx].acceleration.x * DT;
            h_bodies[bodyIdx].velocity.y += h_bodies[bodyIdx].acceleration.y * DT;
            h_bodies[bodyIdx].velocity.z += h_bodies[bodyIdx].acceleration.z * DT;

            // Update position
            h_bodies[bodyIdx].position.x += h_bodies[bodyIdx].velocity.x * DT;
            h_bodies[bodyIdx].position.y += h_bodies[bodyIdx].velocity.y * DT;
            h_bodies[bodyIdx].position.z += h_bodies[bodyIdx].velocity.z * DT;
        }
    }
    else
    {
        // Single-threaded computation
        for (int i = 0; i < nBodies; i++)
        {
            // Get the actual body index when using SFC
            int bodyIdx = enableSFC ? orderIndices[i] : i;

            // Skip non-dynamic bodies
            if (!h_bodies[bodyIdx].isDynamic)
                continue;

            // Reset acceleration
            h_bodies[bodyIdx].acceleration = Vector(0.0, 0.0, 0.0);

            // Compute force from all other bodies
            for (int j = 0; j < nBodies; j++)
            {
                // Get the actual other body index when using SFC
                int otherIdx = enableSFC ? orderIndices[j] : j;

                if (bodyIdx == otherIdx)
                    continue; // Skip self-interaction

                // Vector from body i to body j
                Vector r = h_bodies[otherIdx].position - h_bodies[bodyIdx].position;

                // Distance calculation with softening
                double distSqr = r.lengthSquared() + (E * E);
                double dist = sqrt(distSqr);

                // Skip if bodies are too close (collision)
                if (dist < COLLISION_TH)
                    continue;

                // Gravitational force: G * m1 * m2 / r^3 * r_vector
                double forceMag = GRAVITY * h_bodies[bodyIdx].mass * h_bodies[otherIdx].mass / (distSqr * dist);

                // Update acceleration (F = ma, so a = F/m)
                h_bodies[bodyIdx].acceleration.x += (r.x * forceMag) / h_bodies[bodyIdx].mass;
                h_bodies[bodyIdx].acceleration.y += (r.y * forceMag) / h_bodies[bodyIdx].mass;
                h_bodies[bodyIdx].acceleration.z += (r.z * forceMag) / h_bodies[bodyIdx].mass;
            }

            // Update velocity (Euler integration)
            h_bodies[bodyIdx].velocity.x += h_bodies[bodyIdx].acceleration.x * DT;
            h_bodies[bodyIdx].velocity.y += h_bodies[bodyIdx].acceleration.y * DT;
            h_bodies[bodyIdx].velocity.z += h_bodies[bodyIdx].acceleration.z * DT;

            // Update position
            h_bodies[bodyIdx].position.x += h_bodies[bodyIdx].velocity.x * DT;
            h_bodies[bodyIdx].position.y += h_bodies[bodyIdx].velocity.y * DT;
            h_bodies[bodyIdx].position.z += h_bodies[bodyIdx].velocity.z * DT;
        }
    }

    // Copy updated bodies back to device for rendering
    copyBodiesToDevice();
}

void SFCCPUDirectSum::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Reset unused metrics
    metrics.resetTimeMs = 0.0f;  // Not used in direct sum
    metrics.bboxTimeMs = 0.0f;   // Used internally in SFC ordering
    metrics.octreeTimeMs = 0.0f; // Not used in direct sum

    // Apply SFC ordering when enabled and it's time to reorder
    if (useSFC)
    {
        // Increment the iteration counter
        iterationCounter++;

        // Only reorder if it's time based on the frequency or first iteration
        if (iterationCounter >= reorderFrequency || iterationCounter == 1)
        {
            // Reset the counter
            iterationCounter = 0;

            // Compute new SFC ordering
            orderBodiesBySFC();
        }
    }

    // Compute forces (with SFC ordering if enabled)
    computeForces();
}

// Barnes-hut
SFCCPUBarnesHut::SFCCPUBarnesHut(int numBodies, bool useParallelization, int threads,
                                 BodyDistribution dist, unsigned int seed,
                                 bool enableSFC, SFCOrderingMode sfcOrderingMode, int reorderFreq)
    : CPUBarnesHut(numBodies, useParallelization, threads, dist, seed, false, SFCOrderingMode::PARTICLES, 0)
{
    // Override the SFC parameters from the base class
    this->useSFC = enableSFC;
    this->orderingMode = sfcOrderingMode;
    this->reorderFrequency = reorderFreq;
    this->iterationCounter = 0;

    // Initialize Morton code and ordering structures
    if (enableSFC)
    {
        mortonCodes.resize(numBodies);
        orderedIndices.resize(numBodies);

        // Initialize indices with identity mapping
        for (int i = 0; i < numBodies; i++)
        {
            orderedIndices[i] = i;
        }
    }

    // Log configuration
    std::cout << "CPU SFC Barnes-Hut Simulation created with " << numBodies << " bodies." << std::endl;
    if (useParallelization)
    {
        std::cout << "OpenMP enabled with " << threads << " threads." << std::endl;
    }
    if (enableSFC)
    {
        std::cout << "Space-Filling Curve ordering enabled with mode "
                  << (sfcOrderingMode == SFCOrderingMode::PARTICLES ? "PARTICLES" : "OCTANTS")
                  << " and reorder frequency " << reorderFreq << std::endl;
    }
}

SFCCPUBarnesHut::~SFCCPUBarnesHut()
{
    // Vector data will be automatically cleaned up
}

void SFCCPUBarnesHut::update()
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

        // Only reorder if it's time based on the frequency or first iteration
        if (iterationCounter >= reorderFrequency || iterationCounter == 1)
        {
            // Reset the counter
            iterationCounter = 0;

            // Compute new SFC ordering
            orderBodiesBySFC();
        }
    }

    // Compute the bounding box
    computeBoundingBox();

    // Build the octree (with SFC ordering if enabled)
    buildOctree();

    // Compute forces (with SFC ordering if enabled)
    computeForces();
}

void SFCCPUBarnesHut::orderBodiesBySFC()
{
    if (!useSFC)
    {
        return;
    }

    // Make sure bodies are on host side
    copyBodiesFromDevice();

    // First, compute the domain's bounding box if not already done
    computeBoundingBox();

    // Compute Morton codes for each body
    if (useOpenMP)
    {
        omp_set_num_threads(numThreads);

#pragma omp parallel for
        for (int i = 0; i < nBodies; i++)
        {
            mortonCodes[i] = sfc::MortonCurve().positionToCode(h_bodies[i].position, minBound, maxBound);
            orderedIndices[i] = i;
        }
    }
    else
    {
        for (int i = 0; i < nBodies; i++)
        {
            mortonCodes[i] = sfc::MortonCurve().positionToCode(h_bodies[i].position, minBound, maxBound);
            orderedIndices[i] = i;
        }
    }

    // Sort indices based on Morton codes
    if (useOpenMP && nBodies > 10000)
    {
        // Use a serial sort for simplicity and reliability
        std::sort(orderedIndices.begin(), orderedIndices.end(),
                  [this](int a, int b)
                  {
                      return mortonCodes[a] < mortonCodes[b];
                  });
    }
    else
    {
        // Sequential sort
        std::sort(orderedIndices.begin(), orderedIndices.end(),
                  [this](int a, int b)
                  {
                      return mortonCodes[a] < mortonCodes[b];
                  });
    }
}

void SFCCPUBarnesHut::buildOctree()
{
    // Measure execution time
    CudaTimer timer(metrics.octreeTimeMs);

    // Clear any existing tree
    root.reset(new CPUOctreeNode());

    // Set the root node properties
    Vector center = Vector(
        (minBound.x + maxBound.x) * 0.5,
        (minBound.y + maxBound.y) * 0.5,
        (minBound.z + maxBound.z) * 0.5);
    double halfWidth = std::max(
        std::max(maxBound.x - center.x, maxBound.y - center.y),
        maxBound.z - center.z);

    root->center = center;
    root->halfWidth = halfWidth;

    // Insert all bodies into the octree with SFC ordering if enabled
    const bool enableSFC = useSFC && !orderedIndices.empty();

    for (int i = 0; i < nBodies; i++)
    {
        // Get the actual body index when using SFC
        int bodyIdx = enableSFC ? orderedIndices[i] : i;

        // Start at the root node
        CPUOctreeNode *node = root.get();

        // Keep track of the current node's level for subdivision limits
        int level = 0;
        const int MAX_LEVEL = 20; // Prevent excessive subdivision

        while (!node->isLeaf && level < MAX_LEVEL)
        {
            // Non-leaf node: determine which child octant the body belongs to
            int octant = node->getOctant(h_bodies[bodyIdx].position);

            // Create the child if it doesn't exist
            if (!node->children[octant])
            {
                node->children[octant] = new CPUOctreeNode();
                node->children[octant]->center = node->getOctantCenter(octant);
                node->children[octant]->halfWidth = node->halfWidth * 0.5;
            }

            // Move to the child node
            node = node->children[octant];
            level++;
        }

        // We've reached a leaf node
        if (node->bodyIndex == -1)
        {
            // Empty leaf node, store the body
            node->bodyIndex = bodyIdx;
        }
        else
        {
            // Node already contains a body, subdivide further
            int existingIndex = node->bodyIndex;
            node->bodyIndex = -1;
            node->isLeaf = false;

            // Add the existing body to a child
            int octant1 = node->getOctant(h_bodies[existingIndex].position);
            if (!node->children[octant1])
            {
                node->children[octant1] = new CPUOctreeNode();
                node->children[octant1]->center = node->getOctantCenter(octant1);
                node->children[octant1]->halfWidth = node->halfWidth * 0.5;
            }
            node->children[octant1]->bodyIndex = existingIndex;

            // Add the new body to a child
            int octant2 = node->getOctant(h_bodies[bodyIdx].position);
            if (!node->children[octant2])
            {
                node->children[octant2] = new CPUOctreeNode();
                node->children[octant2]->center = node->getOctantCenter(octant2);
                node->children[octant2]->halfWidth = node->halfWidth * 0.5;
            }

            // Handle the case where both bodies go to the same octant
            if (octant1 == octant2)
            {
                // We need to recursively insert into the child
                // Reset bodyIndex and call buildOctree again
                node->children[octant1]->bodyIndex = -1;
                node->children[octant1]->isLeaf = false;

                // Create grandchild for the existing body
                int subOctant1 = node->children[octant1]->getOctant(h_bodies[existingIndex].position);
                if (!node->children[octant1]->children[subOctant1])
                {
                    node->children[octant1]->children[subOctant1] = new CPUOctreeNode();
                    node->children[octant1]->children[subOctant1]->center =
                        node->children[octant1]->getOctantCenter(subOctant1);
                    node->children[octant1]->children[subOctant1]->halfWidth =
                        node->children[octant1]->halfWidth * 0.5;
                }
                node->children[octant1]->children[subOctant1]->bodyIndex = existingIndex;

                // Create grandchild for the new body
                int subOctant2 = node->children[octant1]->getOctant(h_bodies[bodyIdx].position);
                if (!node->children[octant1]->children[subOctant2])
                {
                    node->children[octant1]->children[subOctant2] = new CPUOctreeNode();
                    node->children[octant1]->children[subOctant2]->center =
                        node->children[octant1]->getOctantCenter(subOctant2);
                    node->children[octant1]->children[subOctant2]->halfWidth =
                        node->children[octant1]->halfWidth * 0.5;
                }

                // If they still end up in the same octant, we'll handle it in the next iteration
                if (subOctant1 != subOctant2)
                {
                    node->children[octant1]->children[subOctant2]->bodyIndex = bodyIdx;
                }
                else
                {
                    // Here we could continue the recursion, but for simplicity,
                    // we'll just add to a list of bodies in this node
                    if (node->children[octant1]->children[subOctant1]->bodies.empty())
                    {
                        node->children[octant1]->children[subOctant1]->bodies.push_back(existingIndex);
                    }
                    node->children[octant1]->children[subOctant1]->bodies.push_back(bodyIdx);
                    node->children[octant1]->children[subOctant1]->isLeaf = true;
                    node->children[octant1]->children[subOctant1]->bodyIndex = -1;
                }
            }
            else
            {
                // Bodies go to different octants, simple case
                node->children[octant2]->bodyIndex = bodyIdx;
            }
        }
    }

    // Compute centers of mass recursively
    computeCenterOfMass(root.get());

    // If OCTANTS ordering mode is used, reorder the octants for space-filling curve traversal
    if (useSFC && orderingMode == SFCOrderingMode::OCTANTS)
    {
        // Reorder octants (this would require a recursive function to traverse the tree)
        reorderOctants(root.get());
    }
}

void SFCCPUBarnesHut::reorderOctants(CPUOctreeNode *node)
{
    if (!node || node->isLeaf)
    {
        return;
    }

    // For each non-leaf node, we'll compute the SFC order of its children
    std::vector<std::pair<uint64_t, int>> octantCodes;

    for (int i = 0; i < 8; i++)
    {
        if (node->children[i])
        {
            // Compute the Morton code for this child's center
            uint64_t code = sfc::MortonCurve().positionToCode(node->children[i]->center, minBound, maxBound);
            octantCodes.push_back({code, i});
        }
    }

    // Sort the octants by their Morton codes
    std::sort(octantCodes.begin(), octantCodes.end());

    // Recursively reorder octants in the children
    for (const auto &pair : octantCodes)
    {
        reorderOctants(node->children[pair.second]);
    }
}

void SFCCPUBarnesHut::computeForces()
{
    // Measure execution time
    CudaTimer timer(metrics.forceTimeMs);

    // Make sure the tree is built
    if (!root)
    {
        std::cerr << "Error: Octree not built before force computation" << std::endl;
        return;
    }

    // Save a reference to whether we're using SFC ordering
    const bool enableSFC = useSFC && !orderedIndices.empty();
    const auto &orderIndices = orderedIndices; // Reference to avoid copies

    // Compute forces using Barnes-Hut approximation with SFC support
    if (useOpenMP)
    {
        // Set the number of threads
        omp_set_num_threads(numThreads);

#pragma omp parallel for
        for (int i = 0; i < nBodies; i++)
        {
            // Get the actual body index when using SFC
            int bodyIdx = enableSFC ? orderIndices[i] : i;

            // Skip non-dynamic bodies
            if (!h_bodies[bodyIdx].isDynamic)
                continue;

            // Reset acceleration
            h_bodies[bodyIdx].acceleration = Vector(0.0, 0.0, 0.0);

            // Compute force from the octree
            computeForceFromNode(h_bodies[bodyIdx], root.get());

            // Update velocity (Euler integration)
            h_bodies[bodyIdx].velocity.x += h_bodies[bodyIdx].acceleration.x * DT;
            h_bodies[bodyIdx].velocity.y += h_bodies[bodyIdx].acceleration.y * DT;
            h_bodies[bodyIdx].velocity.z += h_bodies[bodyIdx].acceleration.z * DT;

            // Update position
            h_bodies[bodyIdx].position.x += h_bodies[bodyIdx].velocity.x * DT;
            h_bodies[bodyIdx].position.y += h_bodies[bodyIdx].velocity.y * DT;
            h_bodies[bodyIdx].position.z += h_bodies[bodyIdx].velocity.z * DT;
        }
    }
    else
    {
        // Single-threaded computation
        for (int i = 0; i < nBodies; i++)
        {
            // Get the actual body index when using SFC
            int bodyIdx = enableSFC ? orderIndices[i] : i;

            // Skip non-dynamic bodies
            if (!h_bodies[bodyIdx].isDynamic)
                continue;

            // Reset acceleration
            h_bodies[bodyIdx].acceleration = Vector(0.0, 0.0, 0.0);

            // Compute force from the octree
            computeForceFromNode(h_bodies[bodyIdx], root.get());

            // Update velocity (Euler integration)
            h_bodies[bodyIdx].velocity.x += h_bodies[bodyIdx].acceleration.x * DT;
            h_bodies[bodyIdx].velocity.y += h_bodies[bodyIdx].acceleration.y * DT;
            h_bodies[bodyIdx].velocity.z += h_bodies[bodyIdx].acceleration.z * DT;

            // Update position
            h_bodies[bodyIdx].position.x += h_bodies[bodyIdx].velocity.x * DT;
            h_bodies[bodyIdx].position.y += h_bodies[bodyIdx].velocity.y * DT;
            h_bodies[bodyIdx].position.z += h_bodies[bodyIdx].velocity.z * DT;
        }
    }

    // Copy updated bodies back to device for rendering
    copyBodiesToDevice();
}