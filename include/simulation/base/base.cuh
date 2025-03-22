#ifndef BASE_CUH
#define BASE_CUH

#include "../../common/types.cuh"
#include "../../common/constants.cuh"
#include "../../common/error_handling.cuh"
#include "../../ui/simulation_state.hpp"
#include <functional>
#include <vector>
#include <memory>

__global__ void ResetKernel(Node *node, int *mutex, int nNodes, int nBodies);
__global__ void ComputeBoundingBoxKernel(Node *node, Body *bodies, int *mutex, int nBodies);
__global__ void ConstructOctTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit);
__global__ void ComputeForceKernel(Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit);

class SimulationBase
{
protected:
    int nBodies;    // Number of bodies in the simulation
    Body *h_bodies; // Host bodies array
    Body *d_bodies; // Device bodies array

    SimulationMetrics metrics; // Performance metrics

    BodyDistribution distribution;
    unsigned int randomSeed;

    bool isInitialized;

    using InitFunction = std::function<void(Body *, int, Vector, unsigned int)>;

    virtual void initBodies(BodyDistribution dist, unsigned int seed);
    void distributeWithFunction(InitFunction initFunc);
    static void initSolarSystem(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);
    static void initGalaxy(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);
    static void initBinarySystem(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);
    static void initUniformSphere(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);
    static void initRandomClusters(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);
    void checkInitialization() const
    {
        if (!isInitialized)
        {
            std::cerr << "Simulation not initialized! Call setup() first." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

public:

    SimulationBase(int numBodies,
                   BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                   unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    virtual ~SimulationBase();
    virtual void setup();
    void setDistributionAndSeed(BodyDistribution dist, unsigned int seed)
    {
        distribution = dist;
        randomSeed = seed;
    }

    virtual void update() = 0;
    void copyBodiesToDevice();
    void copyBodiesFromDevice();

    Body *getBodies() const
    {
        return h_bodies;
    }


    Body *getDeviceBodies() const
    {
        return d_bodies;
    }

    int getNumBodies() const
    {
        return nBodies;
    }

    const SimulationMetrics &getMetrics() const
    {
        return metrics;
    }
};

#endif // BASE_CUH