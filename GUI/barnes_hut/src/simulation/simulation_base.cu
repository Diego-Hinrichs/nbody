#include "../../include/simulation/simulation_base.cuh"
#include <cmath>
#include <cstdlib>
#include <ctime>

SimulationBase::SimulationBase(int numBodies) : nBodies(numBodies),
                                                h_bodies(nullptr),
                                                d_bodies(nullptr),
                                                isInitialized(false)
{

    // Allocate host memory for bodies
    h_bodies = new Body[nBodies];

    // Allocate device memory for bodies
    CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, nBodies * sizeof(Body)));
}

SimulationBase::~SimulationBase()
{
    // Free host memory
    if (h_bodies)
    {
        delete[] h_bodies;
        h_bodies = nullptr;
    }

    // Free device memory
    if (d_bodies)
    {
        CHECK_CUDA_ERROR(cudaFree(d_bodies));
        d_bodies = nullptr;
    }
}

void SimulationBase::initRandomBodies()
{
    // Seed random number generator
    srand(time(NULL));

    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos(CENTERX, CENTERY, CENTERZ);

    // Generate random bodies in a spherical distribution
    for (int i = 0; i < nBodies; ++i)
    {
        // Generate random spherical coordinates
        double u = rand() / (double)RAND_MAX; // For theta
        double v = rand() / (double)RAND_MAX; // For phi
        double theta = 2.0 * M_PI * u;
        double phi = acos(2.0 * v - 1.0);

        // Random radius between min and max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Convert to Cartesian coordinates
        double x = centerPos.x + radius * sin(phi) * cos(theta);
        double y = centerPos.y + radius * sin(phi) * sin(theta);
        double z = centerPos.z + radius * cos(phi);

        // Setup body properties
        h_bodies[i].isDynamic = true;
        h_bodies[i].mass = (i == 0) ? SUN_MASS : EARTH_MASS * (0.8 + 0.4 * (rand() / (double)RAND_MAX));
        h_bodies[i].radius = (i == 0) ? SUN_DIA : EARTH_DIA * (0.5 + (rand() / (double)RAND_MAX));
        h_bodies[i].position = Vector(x, y, z);

        // Setup initial velocities for orbital motion
        if (i > 0)
        { // Skip the sun
            // Calculate orbital velocity for a stable orbit
            double dist = sqrt(x * x + y * y + z * z);
            double orbitalSpeed = sqrt(GRAVITY * SUN_MASS / dist);

            // Calculate a vector perpendicular to the radius
            double vx = -y * orbitalSpeed / dist;
            double vy = x * orbitalSpeed / dist;
            double vz = 0; // Simplified to planar orbits

            h_bodies[i].velocity = Vector(vx, vy, vz);
        }
        else
        {
            h_bodies[i].velocity = Vector(0.0, 0.0, 0.0); // Sun doesn't move initially
        }

        h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
    }
}

void SimulationBase::setup()
{
    // Initialize bodies with random positions and velocities
    initRandomBodies();

    // Transfer bodies to device
    copyBodiesToDevice();

    // Mark as initialized
    isInitialized = true;
}

void SimulationBase::copyBodiesToDevice()
{
    CHECK_CUDA_ERROR(cudaMemcpy(d_bodies, h_bodies, nBodies * sizeof(Body), cudaMemcpyHostToDevice));
}

void SimulationBase::copyBodiesFromDevice()
{
    checkInitialization();
    CHECK_CUDA_ERROR(cudaMemcpy(h_bodies, d_bodies, nBodies * sizeof(Body), cudaMemcpyDeviceToHost));
}