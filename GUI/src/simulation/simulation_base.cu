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
    double minDistance = MIN_DIST * 1.2;
    Vector centerPos(CENTERX, CENTERY, CENTERZ);
    
    // Número de planetas grandes, medianos y pequeños para una distribución más equilibrada
    int largeCount = nBodies / 20;   // 5% planetas grandes
    int mediumCount = nBodies / 8;   // 12.5% planetas medianos
    
    // Generate random bodies in a spherical distribution
    for (int i = 1; i < nBodies; ++i)
    {
        // Generate random spherical coordinates
        double u = rand() / (double)RAND_MAX; // For theta
        double v = rand() / (double)RAND_MAX; // For phi
        double theta = 2.0 * M_PI * u;
        double phi = acos(2.0 * v - 1.0);
        
        // Random radius between min and max distance con mejor distribución
        double distFactor = pow(rand() / (double)RAND_MAX, 0.7); // Mejor distribución espacial
        double radius = (maxDistance - minDistance) * distFactor + minDistance;
        
        // Distribuir los planetas en "órbitas" para mayor realismo
        if (i <= largeCount) {
            // Planetas grandes en órbitas más separadas
            radius = minDistance + (i * (maxDistance - minDistance) / (largeCount * 1.5));
        } else if (i <= (largeCount + mediumCount)) {
            // Planetas medianos en órbitas intermedias
            int idx = i - largeCount;
            radius = minDistance + (idx * (maxDistance - minDistance) / (mediumCount * 1.2));
            // Añadir variación aleatoria para evitar simetría
            radius *= (0.9 + (rand() / (double)RAND_MAX) * 0.2);
        }
        
        // Convert to Cartesian coordinates
        double x = centerPos.x + radius * sin(phi) * cos(theta);
        double y = centerPos.y + radius * sin(phi) * sin(theta);
        double z = centerPos.z + radius * cos(phi);
        
        // Setup body properties
        h_bodies[i].isDynamic = true;
        
        // Adjust mass based on body type
        if (i <= largeCount) {
            // Planetas grandes (tipo Júpiter/Saturno)
            double massFactor = 50.0 + (rand() / (double)RAND_MAX) * 250.0;
            h_bodies[i].mass = EARTH_MASS * massFactor;
            h_bodies[i].radius = EARTH_DIA * 5.0;
        } else if (i <= (largeCount + mediumCount)) {
            // Planetas medianos (tipo Tierra/Neptuno)
            double massFactor = 5.0 + (rand() / (double)RAND_MAX) * 15.0;
            h_bodies[i].mass = EARTH_MASS * massFactor;
            h_bodies[i].radius = EARTH_DIA * 2.0;
        } else {
            // Pequeños cuerpos (menos dominantes visualmente)
            double massFactor = 0.1 + (rand() / (double)RAND_MAX) * 1.0;
            h_bodies[i].mass = EARTH_MASS * massFactor;
            h_bodies[i].radius = EARTH_DIA * 0.5;
        }
        
        h_bodies[i].position = Vector(x, y, z);
        
        // Velocidades orbitales estables
        double orbitalSpeed = sqrt(GRAVITY * SUN_MASS / radius) * (0.8 + (rand() / (double)RAND_MAX) * 0.4);
        
        // Crear vector perpendicular al radial para obtener órbita circular
        Vector radial = h_bodies[i].position - centerPos;
        Vector perpendicular;
        
        // Seleccionar un plano orbital adecuado
        if (fabs(radial.z) < fabs(radial.x) && fabs(radial.z) < fabs(radial.y)) {
            perpendicular = Vector(-radial.y, radial.x, 0);
        } else {
            perpendicular = Vector(0, -radial.z, radial.y);
        }
        
        // Normalizar y aplicar velocidad orbital
        perpendicular = perpendicular.normalize() * orbitalSpeed;
        
        // Aplicar velocidad orbital con ligera inclinación para mayor realismo
        h_bodies[i].velocity = perpendicular;
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