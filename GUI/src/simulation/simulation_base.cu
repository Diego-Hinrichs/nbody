#include "../../include/simulation/simulation_base.cuh"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <algorithm>

SimulationBase::SimulationBase(int numBodies, BodyDistribution dist, unsigned int seed)
    : nBodies(numBodies),
      h_bodies(nullptr),
      d_bodies(nullptr),
      d_tempBodies(nullptr),
      distribution(dist),
      randomSeed(seed),
      isInitialized(false)
{
    // Allocate host memory for bodies
    h_bodies = new Body[nBodies];

    // Allocate device memory for bodies
    CHECK_CUDA_ERROR(cudaMalloc(&d_bodies, nBodies * sizeof(Body)));

    // Allocate temporary device memory if needed
    CHECK_CUDA_ERROR(cudaMalloc(&d_tempBodies, nBodies * sizeof(Body)));
}

SimulationBase::~SimulationBase()
{
    // Free host memory
    if (h_bodies)
    {
        delete[] h_bodies;
        h_bodies = nullptr;
    }

    if (d_bodies)
    {
        CHECK_CUDA_ERROR(cudaFree(d_bodies));
        d_bodies = nullptr;
    }

    if (d_tempBodies)
    {
        CHECK_CUDA_ERROR(cudaFree(d_tempBodies));
        d_tempBodies = nullptr;
    }
}

void SimulationBase::initBodies(BodyDistribution dist, unsigned int seed)
{
    // Call appropriate initialization function based on distribution type
    switch (dist)
    {
    case BodyDistribution::SOLAR_SYSTEM:
        initSolarSystem(seed);
        break;
    case BodyDistribution::GALAXY:
        initGalaxy(seed);
        break;
    case BodyDistribution::BINARY_SYSTEM:
        initBinarySystem(seed);
        break;
    case BodyDistribution::UNIFORM_SPHERE:
        initUniformSphere(seed);
        break;
    case BodyDistribution::RANDOM_CLUSTERS:
        initRandomClusters(seed);
        break;
    default:
        // Default to solar system
        initSolarSystem(seed);
        break;
    }
}

void SimulationBase::setup()
{
    initBodies(distribution, randomSeed);
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

void SimulationBase::initBinarySystem(unsigned int seed)
{
    // Configuración para un sistema binario con planetas orbitando
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> distUni(0.0, 1.0);
    std::normal_distribution<double> distNorm(0.0, 1.0);

    Vector centerPos(CENTERX, CENTERY, CENTERZ);

    // Parámetros del sistema binario
    const double starSeparation = MAX_DIST * 0.1; // Separación entre las estrellas
    const double orbitRadius1 = MAX_DIST * 0.3;   // Radio orbital máximo alrededor de estrella 1
    const double orbitRadius2 = MAX_DIST * 0.3;   // Radio orbital máximo alrededor de estrella 2
    const double systemRadius = MAX_DIST;         // Radio total del sistema

    // Calcular las posiciones de las dos estrellas principales
    Vector star1Pos = Vector(centerPos.x - starSeparation / 2, centerPos.y, centerPos.z);
    Vector star2Pos = Vector(centerPos.x + starSeparation / 2, centerPos.y, centerPos.z);

    // Velocidad orbital de las estrellas entre sí
    double binaryOrbitalSpeed = sqrt(GRAVITY * SUN_MASS / starSeparation);

    // Estrella 1 (más masiva)
    h_bodies[0].isDynamic = true;
    h_bodies[0].mass = SUN_MASS * 1.5;
    h_bodies[0].radius = SUN_DIA * 1.2;
    h_bodies[0].position = star1Pos;
    h_bodies[0].velocity = Vector(0, binaryOrbitalSpeed * 0.4, 0);
    h_bodies[0].acceleration = Vector(0.0, 0.0, 0.0);

    // Estrella 2
    h_bodies[1].isDynamic = true;
    h_bodies[1].mass = SUN_MASS;
    h_bodies[1].radius = SUN_DIA;
    h_bodies[1].position = star2Pos;
    h_bodies[1].velocity = Vector(0, -binaryOrbitalSpeed * 0.6, 0);
    h_bodies[1].acceleration = Vector(0.0, 0.0, 0.0);

    // Fracción de cuerpos por sistema
    int bodiesAroundStar1 = (nBodies - 2) / 2;
    int bodiesAroundStar2 = (nBodies - 2) / 2;
    int bodiesInOuterSystem = nBodies - 2 - bodiesAroundStar1 - bodiesAroundStar2;

    // Planetas orbitando la estrella 1
    for (int i = 2; i < 2 + bodiesAroundStar1; i++)
    {
        // Generar posición orbital aleatoria
        double r = MIN_DIST + orbitRadius1 * pow(distUni(rng), 0.5);
        double theta = 2.0 * M_PI * distUni(rng);
        double phi = acos(2.0 * distUni(rng) - 1.0) * 0.3; // Mantener órbitas más planas

        // Posición relativa a la estrella 1
        double x = r * sin(phi) * cos(theta);
        double y = r * sin(phi) * sin(theta);
        double z = r * cos(phi);

        h_bodies[i].isDynamic = true;

        // Masa basada en la distancia (planetas más pequeños más lejos)
        double normalizedDist = r / orbitRadius1;
        if (normalizedDist < 0.3)
        {
            // Planetas rocosos internos
            h_bodies[i].mass = EARTH_MASS * (0.2 + distUni(rng) * 2.0);
            h_bodies[i].radius = EARTH_DIA * (0.4 + 0.6 * distUni(rng));
        }
        else if (normalizedDist < 0.7)
        {
            // Planetas gaseosos medianos
            h_bodies[i].mass = EARTH_MASS * (5.0 + distUni(rng) * 20.0);
            h_bodies[i].radius = EARTH_DIA * (2.0 + 3.0 * distUni(rng));
        }
        else
        {
            // Planetas helados/enanos lejanos
            h_bodies[i].mass = EARTH_MASS * (0.05 + distUni(rng) * 0.5);
            h_bodies[i].radius = EARTH_DIA * (0.2 + 0.3 * distUni(rng));
        }

        // Posición absoluta
        h_bodies[i].position = Vector(star1Pos.x + x, star1Pos.y + y, star1Pos.z + z);

        // Velocidad orbital
        double orbitalSpeed = sqrt(GRAVITY * h_bodies[0].mass / r);

        // Vector radial y perpendicular
        Vector radial(x, y, z);
        Vector perpendicular;

        if (fabs(radial.z) < fabs(radial.x) && fabs(radial.z) < fabs(radial.y))
        {
            perpendicular = Vector(-radial.y, radial.x, 0);
        }
        else
        {
            perpendicular = Vector(0, -radial.z, radial.y);
        }

        // Normalizar y aplicar velocidad orbital
        perpendicular = perpendicular.normalize() * orbitalSpeed;

        // Añadir la velocidad de la estrella 1
        h_bodies[i].velocity = Vector(
            perpendicular.x + h_bodies[0].velocity.x,
            perpendicular.y + h_bodies[0].velocity.y,
            perpendicular.z + h_bodies[0].velocity.z);

        h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
    }

    // Planetas orbitando la estrella 2
    for (int i = 2 + bodiesAroundStar1; i < 2 + bodiesAroundStar1 + bodiesAroundStar2; i++)
    {
        // Similar a la estrella 1, pero con diferentes parámetros
        double r = MIN_DIST + orbitRadius2 * pow(distUni(rng), 0.5);
        double theta = 2.0 * M_PI * distUni(rng);
        double phi = acos(2.0 * distUni(rng) - 1.0) * 0.3;

        double x = r * sin(phi) * cos(theta);
        double y = r * sin(phi) * sin(theta);
        double z = r * cos(phi);

        h_bodies[i].isDynamic = true;

        // Masa basada en la distancia
        double normalizedDist = r / orbitRadius2;
        if (normalizedDist < 0.3)
        {
            h_bodies[i].mass = EARTH_MASS * (0.1 + distUni(rng) * 1.5);
            h_bodies[i].radius = EARTH_DIA * (0.3 + 0.5 * distUni(rng));
        }
        else if (normalizedDist < 0.7)
        {
            h_bodies[i].mass = EARTH_MASS * (3.0 + distUni(rng) * 15.0);
            h_bodies[i].radius = EARTH_DIA * (1.5 + 2.5 * distUni(rng));
        }
        else
        {
            h_bodies[i].mass = EARTH_MASS * (0.05 + distUni(rng) * 0.3);
            h_bodies[i].radius = EARTH_DIA * (0.1 + 0.2 * distUni(rng));
        }

        h_bodies[i].position = Vector(star2Pos.x + x, star2Pos.y + y, star2Pos.z + z);

        double orbitalSpeed = sqrt(GRAVITY * h_bodies[1].mass / r);

        Vector radial(x, y, z);
        Vector perpendicular;

        if (fabs(radial.z) < fabs(radial.x) && fabs(radial.z) < fabs(radial.y))
        {
            perpendicular = Vector(-radial.y, radial.x, 0);
        }
        else
        {
            perpendicular = Vector(0, -radial.z, radial.y);
        }

        perpendicular = perpendicular.normalize() * orbitalSpeed;

        // Añadir la velocidad de la estrella 2
        h_bodies[i].velocity = Vector(
            perpendicular.x + h_bodies[1].velocity.x,
            perpendicular.y + h_bodies[1].velocity.y,
            perpendicular.z + h_bodies[1].velocity.z);

        h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
    }

    // Cuerpos en órbita exterior del sistema binario
    int startIdx = 2 + bodiesAroundStar1 + bodiesAroundStar2;
    for (int i = startIdx; i < nBodies; i++)
    {
        // Órbitas distantes alrededor del centro de masa del sistema
        double r = (starSeparation * 2) + (systemRadius - starSeparation * 2) * pow(distUni(rng), 0.5);
        double theta = 2.0 * M_PI * distUni(rng);
        double phi = acos(2.0 * distUni(rng) - 1.0) * 0.2; // Más plano

        double x = r * sin(phi) * cos(theta);
        double y = r * sin(phi) * sin(theta);
        double z = r * cos(phi);

        h_bodies[i].isDynamic = true;

        // Objetos más pequeños en el exterior
        h_bodies[i].mass = EARTH_MASS * (0.001 + distUni(rng) * 0.1);
        h_bodies[i].radius = EARTH_DIA * (0.05 + 0.15 * distUni(rng));

        h_bodies[i].position = Vector(centerPos.x + x, centerPos.y + y, centerPos.z + z);

        // Masa total del sistema
        double totalMass = h_bodies[0].mass + h_bodies[1].mass;
        double orbitalSpeed = sqrt(GRAVITY * totalMass / r);

        Vector radial(x, y, z);
        Vector perpendicular;

        if (fabs(radial.z) < fabs(radial.x) && fabs(radial.z) < fabs(radial.y))
        {
            perpendicular = Vector(-radial.y, radial.x, 0);
        }
        else
        {
            perpendicular = Vector(0, -radial.z, radial.y);
        }

        perpendicular = perpendicular.normalize() * orbitalSpeed;

        h_bodies[i].velocity = perpendicular;
        h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
    }
}

void SimulationBase::initUniformSphere(unsigned int seed)
{
    // Distribución uniforme en una esfera
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> distUni(0.0, 1.0);
    std::normal_distribution<double> distNorm(0.0, 1.0);

    Vector centerPos(CENTERX, CENTERY, CENTERZ);
    double maxRadius = MAX_DIST * 0.9;

    // Configurar un objeto masivo en el centro
    h_bodies[0].isDynamic = true;
    h_bodies[0].mass = SUN_MASS * 10;
    h_bodies[0].radius = SUN_DIA;
    h_bodies[0].position = centerPos;
    h_bodies[0].velocity = Vector(0.0, 0.0, 0.0);
    h_bodies[0].acceleration = Vector(0.0, 0.0, 0.0);

    // Distribuir cuerpos uniformemente en la esfera
    for (int i = 1; i < nBodies; i++)
    {
        // Distribución uniforme en una esfera
        // Usamos el método de rechazo para generar puntos uniformes en una esfera
        Vector point;
        do
        {
            point.x = 2.0 * distUni(rng) - 1.0;
            point.y = 2.0 * distUni(rng) - 1.0;
            point.z = 2.0 * distUni(rng) - 1.0;
        } while (point.lengthSquared() > 1.0);

        // Escalar al radio deseado
        double radius = maxRadius * pow(distUni(rng), 1.0 / 3.0); // Corrección para distribución uniforme
        point = point.normalize() * radius;

        // Propiedades del cuerpo
        h_bodies[i].isDynamic = true;
        h_bodies[i].mass = EARTH_MASS * (0.1 + distUni(rng) * 2.0);
        h_bodies[i].radius = EARTH_DIA * (0.2 + 0.8 * distUni(rng));

        // Posición absoluta
        h_bodies[i].position = Vector(centerPos.x + point.x, centerPos.y + point.y, centerPos.z + point.z);

        // Velocidades aleatorias (distribución normal)
        Vector velocity;
        velocity.x = distNorm(rng) * 1.0e4;
        velocity.y = distNorm(rng) * 1.0e4;
        velocity.z = distNorm(rng) * 1.0e4;

        h_bodies[i].velocity = velocity;
        h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
    }
}

void SimulationBase::initRandomClusters(unsigned int seed)
{
    // Distribución con múltiples clústeres aleatorios
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> distUni(0.0, 1.0);
    std::normal_distribution<double> distNorm(0.0, 1.0);

    Vector centerPos(CENTERX, CENTERY, CENTERZ);
    double domainRadius = MAX_DIST * 0.9;

    // Parámetros de los clústeres
    const int numClusters = 5 + static_cast<int>(distUni(rng) * 5); // Entre 5 y 10 clústeres
    std::vector<Vector> clusterCenters(numClusters);
    std::vector<double> clusterRadii(numClusters);
    std::vector<double> clusterMass(numClusters);

    // Objeto masivo en el centro global
    h_bodies[0].isDynamic = false;
    h_bodies[0].mass = SUN_MASS * 50;
    h_bodies[0].radius = SUN_DIA * 2;
    h_bodies[0].position = centerPos;
    h_bodies[0].velocity = Vector(0.0, 0.0, 0.0);
    h_bodies[0].acceleration = Vector(0.0, 0.0, 0.0);

    // Generar posiciones de los clústeres
    for (int c = 0; c < numClusters; c++)
    {
        // Distribución uniforme en el dominio
        double r = domainRadius * pow(distUni(rng), 1.0 / 3.0);
        double theta = 2.0 * M_PI * distUni(rng);
        double phi = acos(2.0 * distUni(rng) - 1.0);

        double x = r * sin(phi) * cos(theta);
        double y = r * sin(phi) * sin(theta);
        double z = r * cos(phi);

        clusterCenters[c] = Vector(centerPos.x + x, centerPos.y + y, centerPos.z + z);

        // Radio del clúster (proporcional a la distancia al centro)
        clusterRadii[c] = MAX_DIST * (0.05 + 0.15 * distUni(rng)) * (0.5 + 0.5 * r / domainRadius);

        // Masa del clúster
        clusterMass[c] = SUN_MASS * (1.0 + 9.0 * distUni(rng));
    }

    // Centros de los clústeres (objetos masivos)
    for (int c = 0; c < numClusters && c + 1 < nBodies; c++)
    {
        h_bodies[c + 1].isDynamic = true;
        h_bodies[c + 1].mass = clusterMass[c];
        h_bodies[c + 1].radius = SUN_DIA * (0.2 + 0.8 * distUni(rng));
        h_bodies[c + 1].position = clusterCenters[c];

        // Velocidad orbital alrededor del centro global
        Vector radial = clusterCenters[c] - centerPos;
        double radialDist = radial.length();
        double orbitalSpeed = sqrt(GRAVITY * h_bodies[0].mass / radialDist) * (0.8 + 0.4 * distUni(rng));

        // Dirección perpendicular al radio
        Vector perpendicular;
        if (fabs(radial.z) < fabs(radial.x) && fabs(radial.z) < fabs(radial.y))
        {
            perpendicular = Vector(-radial.y, radial.x, 0);
        }
        else
        {
            perpendicular = Vector(0, -radial.z, radial.y);
        }

        perpendicular = perpendicular.normalize() * orbitalSpeed;

        h_bodies[c + 1].velocity = perpendicular;
        h_bodies[c + 1].acceleration = Vector(0.0, 0.0, 0.0);
    }

    // Distribuir el resto de los cuerpos en los clústeres
    int currentIdx = numClusters + 1;
    int bodiesPerCluster = (nBodies - currentIdx) / numClusters;

    for (int c = 0; c < numClusters; c++)
    {
        for (int i = 0; i < bodiesPerCluster && currentIdx < nBodies; i++, currentIdx++)
        {
            // Distribución gaussiana alrededor del centro del clúster
            double sigma = clusterRadii[c] / 3.0; // Para que la mayoría esté dentro del radio
            double x = distNorm(rng) * sigma;
            double y = distNorm(rng) * sigma;
            double z = distNorm(rng) * sigma;

            h_bodies[currentIdx].isDynamic = true;

            // Masa proporcional a la distancia al centro del clúster
            double distToCenter = sqrt(x * x + y * y + z * z);
            double distFactor = exp(-distToCenter / (clusterRadii[c] * 0.5));

            h_bodies[currentIdx].mass = EARTH_MASS * (0.1 + 2.0 * distUni(rng) * distFactor);
            h_bodies[currentIdx].radius = EARTH_DIA * (0.2 + 0.8 * distUni(rng) * distFactor);

            // Posición absoluta
            h_bodies[currentIdx].position = Vector(
                clusterCenters[c].x + x,
                clusterCenters[c].y + y,
                clusterCenters[c].z + z);

            // Velocidad orbital alrededor del centro del clúster
            double orbitalSpeed = sqrt(GRAVITY * h_bodies[c + 1].mass / distToCenter) * (0.5 + 0.5 * distUni(rng));

            Vector radial(x, y, z);
            Vector perpendicular;

            if (fabs(radial.z) < fabs(radial.x) && fabs(radial.z) < fabs(radial.y))
            {
                perpendicular = Vector(-radial.y, radial.x, 0);
            }
            else
            {
                perpendicular = Vector(0, -radial.z, radial.y);
            }

            // Añadir la velocidad del clúster
            Vector clusterVelocity = h_bodies[c + 1].velocity;
            perpendicular = perpendicular.normalize() * orbitalSpeed;

            h_bodies[currentIdx].velocity = Vector(
                perpendicular.x + clusterVelocity.x,
                perpendicular.y + clusterVelocity.y,
                perpendicular.z + clusterVelocity.z);

            h_bodies[currentIdx].acceleration = Vector(0.0, 0.0, 0.0);
        }
    }

    // Cualquier cuerpo restante se distribuye aleatoriamente
    while (currentIdx < nBodies)
    {
        double r = domainRadius * pow(distUni(rng), 1.0 / 3.0);
        double theta = 2.0 * M_PI * distUni(rng);
        double phi = acos(2.0 * distUni(rng) - 1.0);

        double x = r * sin(phi) * cos(theta);
        double y = r * sin(phi) * sin(theta);
        double z = r * cos(phi);

        h_bodies[currentIdx].isDynamic = true;
        h_bodies[currentIdx].mass = EARTH_MASS * (0.01 + 0.1 * distUni(rng));
        h_bodies[currentIdx].radius = EARTH_DIA * (0.1 + 0.2 * distUni(rng));
        h_bodies[currentIdx].position = Vector(centerPos.x + x, centerPos.y + y, centerPos.z + z);

        // Velocidad orbital respecto al centro
        double orbitalSpeed = sqrt(GRAVITY * h_bodies[0].mass / r) * (0.8 + 0.4 * distUni(rng));

        Vector radial(x, y, z);
        Vector perpendicular;

        if (fabs(radial.z) < fabs(radial.x) && fabs(radial.z) < fabs(radial.y))
        {
            perpendicular = Vector(-radial.y, radial.x, 0);
        }
        else
        {
            perpendicular = Vector(0, -radial.z, radial.y);
        }

        h_bodies[currentIdx].velocity = perpendicular.normalize() * orbitalSpeed;
        h_bodies[currentIdx].acceleration = Vector(0.0, 0.0, 0.0);

        currentIdx++;
    }
}

void SimulationBase::initSolarSystem(unsigned int seed)
{
    // Seed random number generator
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> distUni(0.0, 1.0);

    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST * 1.2;
    Vector centerPos(CENTERX, CENTERY, CENTERZ);

    // Número de planetas grandes, medianos y pequeños para una distribución más equilibrada
    int largeCount = nBodies / 20; // 5% planetas grandes
    int mediumCount = nBodies / 8; // 12.5% planetas medianos

    // Generate random bodies in a spherical distribution
    for (int i = 1; i < nBodies; ++i)
    {
        // Generate random spherical coordinates
        double u = distUni(rng); // For theta
        double v = distUni(rng); // For phi
        double theta = 2.0 * M_PI * u;
        double phi = acos(2.0 * v - 1.0);

        // Random radius between min and max distance con mejor distribución
        double distFactor = pow(distUni(rng), 0.7); // Mejor distribución espacial
        double radius = (maxDistance - minDistance) * distFactor + minDistance;

        // Distribuir los planetas en "órbitas" para mayor realismo
        if (i <= largeCount)
        {
            // Planetas grandes en órbitas más separadas
            radius = minDistance + (i * (maxDistance - minDistance) / (largeCount * 1.5));
        }
        else if (i <= (largeCount + mediumCount))
        {
            // Planetas medianos en órbitas intermedias
            int idx = i - largeCount;
            radius = minDistance + (idx * (maxDistance - minDistance) / (mediumCount * 1.2));
            // Añadir variación aleatoria para evitar simetría
            radius *= (0.9 + (distUni(rng)) * 0.2);
        }

        // Convert to Cartesian coordinates
        double x = centerPos.x + radius * sin(phi) * cos(theta);
        double y = centerPos.y + radius * sin(phi) * sin(theta);
        double z = centerPos.z + radius * cos(phi);

        // Setup body properties
        h_bodies[i].isDynamic = true;

        // Adjust mass based on body type
        if (i <= largeCount)
        {
            // Planetas grandes (tipo Júpiter/Saturno)
            double massFactor = 50.0 + (distUni(rng)) * 250.0;
            h_bodies[i].mass = EARTH_MASS * massFactor;
            h_bodies[i].radius = EARTH_DIA * 5.0;
        }
        else if (i <= (largeCount + mediumCount))
        {
            // Planetas medianos (tipo Tierra/Neptuno)
            double massFactor = 5.0 + (distUni(rng)) * 15.0;
            h_bodies[i].mass = EARTH_MASS * massFactor;
            h_bodies[i].radius = EARTH_DIA * 2.0;
        }
        else
        {
            // Pequeños cuerpos (menos dominantes visualmente)
            double massFactor = 0.1 + (distUni(rng)) * 1.0;
            h_bodies[i].mass = EARTH_MASS * massFactor;
            h_bodies[i].radius = EARTH_DIA * 0.5;
        }

        h_bodies[i].position = Vector(x, y, z);

        // Velocidades orbitales estables
        double orbitalSpeed = sqrt(GRAVITY * SUN_MASS / radius) * (0.8 + (distUni(rng)) * 0.4);

        // Crear vector perpendicular al radial para obtener órbita circular
        Vector radial = h_bodies[i].position - centerPos;
        Vector perpendicular;

        // Seleccionar un plano orbital adecuado
        if (fabs(radial.z) < fabs(radial.x) && fabs(radial.z) < fabs(radial.y))
        {
            perpendicular = Vector(-radial.y, radial.x, 0);
        }
        else
        {
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

void SimulationBase::initGalaxy(unsigned int seed)
{
    // Configuración para una distribución de galaxia espiral
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> distUni(0.0, 1.0);
    std::normal_distribution<double> distNorm(0.0, 1.0);

    Vector centerPos(CENTERX, CENTERY, CENTERZ);

    // Parámetros de la galaxia
    const double coreRadius = MAX_DIST * 0.1;     // Radio del núcleo central
    const double diskRadius = MAX_DIST * 0.8;     // Radio del disco galáctico
    const double diskThickness = MAX_DIST * 0.05; // Grosor del disco galáctico
    const double spiralTightness = 0.5;           // Qué tan apretados son los brazos espirales
    const int numSpiralArms = 2;                  // Número de brazos espirales

    // Agujero negro central
    h_bodies[0].isDynamic = false;
    h_bodies[0].mass = SUN_MASS * 10000; // Agujero negro supermasivo
    h_bodies[0].radius = SUN_DIA * 2;
    h_bodies[0].position = centerPos;
    h_bodies[0].velocity = Vector(0.0, 0.0, 0.0);
    h_bodies[0].acceleration = Vector(0.0, 0.0, 0.0);

    // Estrellas masivas cerca del núcleo (5% del total)
    int coreBodies = nBodies * 0.05;
    for (int i = 1; i <= coreBodies; i++)
    {
        // Distribución radial del núcleo (concentración central)
        double r = coreRadius * pow(distUni(rng), 0.5);
        double theta = 2.0 * M_PI * distUni(rng);
        double phi = acos(2.0 * distUni(rng) - 1.0) * 0.2; // Más plano en el núcleo

        // Posición en coordenadas cartesianas
        double x = centerPos.x + r * sin(phi) * cos(theta);
        double y = centerPos.y + r * sin(phi) * sin(theta);
        double z = centerPos.z + r * cos(phi);

        // Propiedades de los cuerpos en el núcleo (estrellas masivas y calientes)
        h_bodies[i].isDynamic = true;
        h_bodies[i].mass = SUN_MASS * (1.0 + 5.0 * distUni(rng)); // Estrellas de 1-6 masas solares
        h_bodies[i].radius = SUN_DIA * (0.5 + 0.5 * distUni(rng));
        h_bodies[i].position = Vector(x, y, z);

        // Velocidad orbital basada en el agujero negro central
        double orbitalSpeed = sqrt(GRAVITY * h_bodies[0].mass / r) * (0.9 + 0.2 * distUni(rng));

        // Dirección de la velocidad perpendicular al radio (órbita circular)
        Vector radial(x - centerPos.x, y - centerPos.y, z - centerPos.z);
        Vector perpendicular;

        // Velocidad principalmente en el plano XY para el disco
        if (fabs(radial.z) < 0.01)
        {
            perpendicular = Vector(-radial.y, radial.x, 0);
        }
        else
        {
            perpendicular = Vector(-radial.y, radial.x, 0);
            perpendicular = perpendicular.normalize();
        }

        h_bodies[i].velocity = perpendicular * orbitalSpeed;
        h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
    }

    // Estrellas del disco galáctico con estructura espiral
    for (int i = coreBodies + 1; i < nBodies; i++)
    {
        // Distribución radial (más densa entre coreRadius y diskRadius)
        double radialPosition = distUni(rng);
        double r = coreRadius + (diskRadius - coreRadius) * radialPosition;

        // Estructura espiral
        double spiralOffset = 0.0;
        if (distUni(rng) < 0.7)
        { // 70% de las estrellas siguen los brazos espirales
            // Ángulo base
            double baseAngle = 2.0 * M_PI * distUni(rng);

            // Elegir un brazo espiral al azar
            int arm = static_cast<int>(distUni(rng) * numSpiralArms);

            // Ajustar ángulo según el brazo (distribuir uniformemente)
            baseAngle = baseAngle + (2.0 * M_PI * arm) / numSpiralArms;

            // Ecuación espiral logarítmica: r = a * e^(b*theta)
            spiralOffset = spiralTightness * log(radialPosition * 9.0 + 1.0);

            // Agregar dispersión alrededor del brazo espiral
            spiralOffset += distNorm(rng) * 0.3;
        }

        // Ángulo azimutal con estructura espiral
        double theta = 2.0 * M_PI * distUni(rng) + spiralOffset;

        // Desviación del plano del disco (distribución normal)
        double zOffset = distNorm(rng) * diskThickness * (0.1 + 0.9 * radialPosition);

        // Posición final
        double x = centerPos.x + r * cos(theta);
        double y = centerPos.y + r * sin(theta);
        double z = centerPos.z + zOffset;

        // Propiedades según distancia al centro
        h_bodies[i].isDynamic = true;

        // Masa y tamaño según posición
        if (r < diskRadius * 0.3)
        {
            // Estrellas más masivas en las regiones interiores de los brazos
            h_bodies[i].mass = SUN_MASS * (0.5 + distUni(rng) * 2.0);
            h_bodies[i].radius = SUN_DIA * (0.5 + 0.5 * distUni(rng));
        }
        else
        {
            // Estrellas de masa media y baja en las regiones exteriores
            h_bodies[i].mass = SUN_MASS * (0.1 + distUni(rng) * 0.9);
            h_bodies[i].radius = SUN_DIA * (0.2 + 0.3 * distUni(rng));
        }

        h_bodies[i].position = Vector(x, y, z);

        // Velocidad orbital (ajustada por la masa total interior)
        double enclosedMass = h_bodies[0].mass * (0.1 + 0.9 * (r / diskRadius));
        double orbitalSpeed = sqrt(GRAVITY * enclosedMass / r) * (0.9 + 0.2 * distUni(rng));

        // Dirección de la velocidad (principalmente tangencial al centro en el plano XY)
        Vector direction(-sin(theta), cos(theta), 0);

        h_bodies[i].velocity = direction * orbitalSpeed;
        h_bodies[i].acceleration = Vector(0.0, 0.0, 0.0);
    }
}