#include "../../include/simulation/cpu_sfc_direct_sum.hpp"
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