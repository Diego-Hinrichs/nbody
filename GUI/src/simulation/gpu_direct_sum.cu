#include "../../include/simulation/gpu_direct_sum.cuh"
#include <iostream>

/**
 * @brief CUDA kernel for direct force calculation between all body pairs
 *
 * This kernel computes the gravitational forces between all pairs of bodies
 * using the Direct Sum approach (O(n²) complexity).
 *
 * @param bodies Array of body structures
 * @param nBodies Number of bodies in the simulation
 */
__global__ void DirectSumForceKernel(Body *bodies, int nBodies)
{
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if this thread should process a body
    if (i >= nBodies || !bodies[i].isDynamic)
        return;

    // Reset acceleration
    bodies[i].acceleration = Vector(0.0, 0.0, 0.0);

    // Cache this body's data to registers for faster access
    Vector myPos = bodies[i].position;
    double myMass = bodies[i].mass;

    // Compute force from all other bodies
    for (int j = 0; j < nBodies; j++)
    {
        if (i == j)
            continue; // Skip self-interaction

        // Vector from body i to body j
        Vector r = bodies[j].position - myPos;

        // Distance calculation with softening
        double distSqr = r.lengthSquared() + (E * E);
        double dist = sqrt(distSqr);

        // Skip if bodies are too close (collision)
        if (dist < COLLISION_TH)
            continue;

        // Gravitational force: G * m1 * m2 / r^3 * r_vector
        double forceMag = GRAVITY * myMass * bodies[j].mass / (distSqr * dist);

        // Update acceleration (F = ma, so a = F/m)
        bodies[i].acceleration.x += (r.x * forceMag) / myMass;
        bodies[i].acceleration.y += (r.y * forceMag) / myMass;
        bodies[i].acceleration.z += (r.z * forceMag) / myMass;
    }

    // Update velocity (Euler integration)
    bodies[i].velocity.x += bodies[i].acceleration.x * DT;
    bodies[i].velocity.y += bodies[i].acceleration.y * DT;
    bodies[i].velocity.z += bodies[i].acceleration.z * DT;

    // Update position
    bodies[i].position.x += bodies[i].velocity.x * DT;
    bodies[i].position.y += bodies[i].velocity.y * DT;
    bodies[i].position.z += bodies[i].velocity.z * DT;
}

GPUDirectSum::GPUDirectSum(int numBodies, BodyDistribution dist, unsigned int seed)
    : SimulationBase(numBodies, dist, seed)
{
    std::cout << "GPU Direct Sum Simulation created with " << numBodies << " bodies." << std::endl;
}

GPUDirectSum::~GPUDirectSum()
{
    // Base class destructor handles most memory cleanup
}

void GPUDirectSum::computeForces()
{
    // Measure execution time
    CudaTimer timer(metrics.forceTimeMs);

    // Launch kernel for direct force calculation
    int blockSize = BLOCK_SIZE;
    int gridSize = (nBodies + blockSize - 1) / blockSize;

    // Launch kernel with error checking
    CUDA_KERNEL_CALL(DirectSumForceKernel, gridSize, blockSize, 0, 0,
                     d_bodies, nBodies);
}

void GPUDirectSum::update()
{
    // Ensure initialization
    checkInitialization();

    // Measure total execution time
    CudaTimer timer(metrics.totalTimeMs);

    // Reset unused metrics
    metrics.resetTimeMs = 0.0f;  // Not used
    metrics.bboxTimeMs = 0.0f;   // Not used
    metrics.octreeTimeMs = 0.0f; // Not used

    // Compute forces and update positions in one kernel
    computeForces();
}