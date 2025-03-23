#include "../../include/simulation/implementations/cpu/direct_sum.hpp"
#include <cmath>
#include <iostream>

CPUDirectSum::CPUDirectSum(
    int numBodies,
    bool useParallelization,
    int threads,
    BodyDistribution dist, 
    unsigned int seed,
    MassDistribution massDist)
    : SimulationBase(numBodies, dist, seed, massDist),
      useOpenMP(useParallelization)
{
    // Initialize thread count
    setThreadCount(threads);

    // Log configuration
    std::cout << "CPU Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
    if (useOpenMP)
    {
        std::cout << "OpenMP enabled with " << numThreads << " threads." << std::endl;
    }
    else
    {
        std::cout << "OpenMP disabled, using single-threaded mode." << std::endl;
    }
}

CPUDirectSum::~CPUDirectSum()
{
    // Nothing specific to clean up
}

void CPUDirectSum::computeForces()
{
    // Measure execution time
    CudaTimer timer(metrics.forceTimeMs);

    // Copy bodies to host for CPU computation if they're not already there
    copyBodiesFromDevice();

    // Compute forces using direct summation (O(n²) complexity)
    if (useOpenMP)
    {
        // Set the number of threads
        omp_set_num_threads(numThreads);

#pragma omp parallel for
        for (int i = 0; i < nBodies; i++)
        {
            // Skip non-dynamic bodies
            if (!h_bodies[i].isDynamic)
                continue;

            // Reset acceleration
            h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);

            // Compute force from all other bodies
            for (int j = 0; j < nBodies; j++)
            {
                if (i == j)
                    continue; // Skip self-interaction

                // Vector from body i to body j
                Vector r = h_bodies[j].position - h_bodies[i].position;

                // Distance calculation with softening
                double distSqr = r.lengthSquared() + (E * E);
                double dist = sqrt(distSqr);

                // Skip if bodies are too close (collision)
                if (dist < COLLISION_TH)
                    continue;

                // Gravitational force: G * m1 * m2 / r^3 * r_vector
                double forceMag = GRAVITY * h_bodies[i].mass * h_bodies[j].mass / (distSqr * dist);

                // Update acceleration (F = ma, so a = F/m)
                h_bodies[i].acceleration.x += (r.x * forceMag) / h_bodies[i].mass;
                h_bodies[i].acceleration.y += (r.y * forceMag) / h_bodies[i].mass;
                h_bodies[i].acceleration.z += (r.z * forceMag) / h_bodies[i].mass;
            }

            // Update velocity (Euler integration)
            h_bodies[i].velocity.x += h_bodies[i].acceleration.x * DT;
            h_bodies[i].velocity.y += h_bodies[i].acceleration.y * DT;
            h_bodies[i].velocity.z += h_bodies[i].acceleration.z * DT;

            // Update position
            h_bodies[i].position.x += h_bodies[i].velocity.x * DT;
            h_bodies[i].position.y += h_bodies[i].velocity.y * DT;
            h_bodies[i].position.z += h_bodies[i].velocity.z * DT;
        }
    }
    else
    {
        // Single-threaded computation
        for (int i = 0; i < nBodies; i++)
        {
            // Skip non-dynamic bodies
            if (!h_bodies[i].isDynamic)
                continue;

            // Reset acceleration
            h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);

            // Compute force from all other bodies
            for (int j = 0; j < nBodies; j++)
            {
                if (i == j)
                    continue; // Skip self-interaction

                // Vector from body i to body j
                Vector r = h_bodies[j].position - h_bodies[i].position;

                // Distance calculation with softening
                double distSqr = r.lengthSquared() + (E * E);
                double dist = sqrt(distSqr);

                // Skip if bodies are too close (collision)
                if (dist < COLLISION_TH)
                    continue;

                // Gravitational force: G * m1 * m2 / r^3 * r_vector
                double forceMag = GRAVITY * h_bodies[i].mass * h_bodies[j].mass / (distSqr * dist);

                // Update acceleration (F = ma, so a = F/m)
                h_bodies[i].acceleration.x += (r.x * forceMag) / h_bodies[i].mass;
                h_bodies[i].acceleration.y += (r.y * forceMag) / h_bodies[i].mass;
                h_bodies[i].acceleration.z += (r.z * forceMag) / h_bodies[i].mass;
            }

            // Update velocity (Euler integration)
            h_bodies[i].velocity.x += h_bodies[i].acceleration.x * DT;
            h_bodies[i].velocity.y += h_bodies[i].acceleration.y * DT;
            h_bodies[i].velocity.z += h_bodies[i].acceleration.z * DT;

            // Update position
            h_bodies[i].position.x += h_bodies[i].velocity.x * DT;
            h_bodies[i].position.y += h_bodies[i].velocity.y * DT;
            h_bodies[i].position.z += h_bodies[i].velocity.z * DT;
        }
    }

    // Copy updated bodies back to device for rendering
    copyBodiesToDevice();
}

void CPUDirectSum::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Simply compute forces and update positions
    // No need for octree or bounding box
    metrics.resetTimeMs = 0.0f;  // Not used
    metrics.bboxTimeMs = 0.0f;   // Not used
    metrics.octreeTimeMs = 0.0f; // Not used

    // Compute forces
    computeForces();
}