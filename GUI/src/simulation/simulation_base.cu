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
    for (int i = 1; i < nBodies; ++i)
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

        // Assign mass based on a narrower power law distribution
        double massFactor = pow(rand() / (double)RAND_MAX, 2.0);
        h_bodies[i].mass = EARTH_MASS;

        // Adjust radius based on mass
        h_bodies[i].radius = EARTH_DIA/2;

        h_bodies[i].position = Vector(x, y, z);

        // Setup initial velocities for orbital motion
        double orbitalSpeed = sqrt(GRAVITY * EARTH_MASS * (1.0 + massFactor) / radius);

        // Setup initial velocities
        double velocityFactor = rand() / (double)RAND_MAX;

        // Calculate a vector perpendicular to the radius with random orientation
        double vx = (-y + (rand() / (double)RAND_MAX - 0.5) * radius) * orbitalSpeed / radius;
        double vy = (x + (rand() / (double)RAND_MAX - 0.5) * radius) * orbitalSpeed / radius;
        double vz = (rand() / (double)RAND_MAX - 0.5) * orbitalSpeed;

        // Apply velocity factor to allow for different orbit shapes
        h_bodies[i].velocity = Vector(vx, vy, vz) * velocityFactor;

        h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
    }
    // Place a sun in the center
    h_bodies[0].isDynamic = false;
    h_bodies[0].mass = SUN_MASS;
    h_bodies[0].radius = SUN_DIA / 2;
    h_bodies[0].position = centerPos;
    h_bodies[0].velocity = Vector(0.0, 0.0, 0.0);
    h_bodies[0].acceleration = Vector(0.0, 0.0, 0.0);
}

void SimulationBase::setup()
{
    initRandomBodies();

    // Imprimir primeros 5 cuerpos para verificación
    // for (int i = 0; i < 5 && i < nBodies; ++i)
    // {
    //     std::cout << "Cuerpo " << i
    //               << ": x=" << h_bodies[i].position.x
    //               << ", y=" << h_bodies[i].position.y
    //               << ", z=" << h_bodies[i].position.z
    //               << ", masa=" << h_bodies[i].mass << std::endl;
    // }

    // Resto del método sin cambios
    copyBodiesToDevice();
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