#ifndef BASE_CUH
#define BASE_CUH

#include "../../common/types.cuh"
#include "../../common/constants.cuh"
#include "../../common/error_handling.cuh"
#include "../../ui/simulation_state.hpp"
#include <functional>
#include <vector>
#include <memory>

extern "C" void BuildOptimizedOctTree(
    Node *d_nodes, Body *d_bodies, Body *d_tempBodies,
    int *orderedIndices, bool useSFC,
    int *octantIndices, bool useOctantOrder,
    int nNodes, int nBodies, int leafLimit);

__global__ void ResetKernel(
    Node *nodes, int *mutex, int nNodes, int nBodies);

__global__ void ComputeBoundingBoxKernel(
    Node *nodes, Body *bodies, int *orderedIndices, bool useSFC,
    int *mutex, int nBodies);

__global__ void OptimizedConstructOctTreeKernelWithOctantOrder(
    Node *nodes, Body *bodies, Body *buffer,
    int *orderedIndices, bool useSFC,
    int *octantIndices, bool useOctantOrder,
    int *jobQueue, int *jobCount, int initJobCount,
    int nNodes, int nBodies, int leafLimit);

__global__ void ComputeForceKernel(
    Node *nodes, Body *bodies, int *orderedIndices, bool useSFC,
    int nNodes, int nBodies, int leafLimit);

/**
 * @brief Base class for N-body simulations
 *
 * This abstract class provides the foundation for various N-body simulation
 * implementations by managing the basic allocation, transfer, and lifecycle
 * of body data.
 */
class SimulationBase
{
protected:
    int nBodies;    // Number of bodies in the simulation
    Body *h_bodies; // Host bodies array
    Body *d_bodies; // Device bodies array
    Body *d_tempBodies;

    SimulationMetrics metrics; // Performance metrics

    // Configuration parameters
    BodyDistribution distribution;
    unsigned int randomSeed;

    // Flag to indicate whether the simulation is initialized
    bool isInitialized;

    // Typedef for body initialization functions
    using InitFunction = std::function<void(Body *, int, Vector, unsigned int)>;

    /**
     * @brief Initialize bodies with specified distribution and seed
     * @param dist Distribution type to use
     * @param seed Random seed for reproducibility
     */
    virtual void initBodies(BodyDistribution dist, unsigned int seed);

    /**
     * @brief Helper function to distribute bodies using a specific initialization function
     * @param initFunc Function to initialize a specific body distribution
     */
    void distributeWithFunction(InitFunction initFunc);

    /**
     * @brief Initialize bodies with solar system like distribution
     * @param bodies Array of bodies to initialize
     * @param numBodies Number of bodies in the array
     * @param centerPos Center position of the system
     * @param seed Random seed for reproducibility
     */
    static void initSolarSystem(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);

    /**
     * @brief Initialize bodies with galaxy-like spiral distribution
     * @param bodies Array of bodies to initialize
     * @param numBodies Number of bodies in the array
     * @param centerPos Center position of the system
     * @param seed Random seed for reproducibility
     */
    static void initGalaxy(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);

    /**
     * @brief Initialize bodies with binary star system distribution
     * @param bodies Array of bodies to initialize
     * @param numBodies Number of bodies in the array
     * @param centerPos Center position of the system
     * @param seed Random seed for reproducibility
     */
    static void initBinarySystem(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);

    /**
     * @brief Initialize bodies with uniform sphere distribution
     * @param bodies Array of bodies to initialize
     * @param numBodies Number of bodies in the array
     * @param centerPos Center position of the system
     * @param seed Random seed for reproducibility
     */
    static void initUniformSphere(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);

    /**
     * @brief Initialize bodies with random clusters
     * @param bodies Array of bodies to initialize
     * @param numBodies Number of bodies in the array
     * @param centerPos Center position of the system
     * @param seed Random seed for reproducibility
     */
    static void initRandomClusters(Body *bodies, int numBodies, Vector centerPos, unsigned int seed);

    /**
     * @brief Ensure the simulation is initialized before operations
     */
    void checkInitialization() const
    {
        if (!isInitialized)
        {
            std::cerr << "Simulation not initialized! Call setup() first." << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

public:
    /**
     * @brief Constructor
     * @param numBodies Number of bodies in the simulation
     * @param dist Initial distribution of bodies
     * @param seed Random seed for reproducibility
     */
    SimulationBase(int numBodies,
                   BodyDistribution dist = BodyDistribution::SOLAR_SYSTEM,
                   unsigned int seed = static_cast<unsigned int>(time(nullptr)));

    /**
     * @brief Virtual destructor
     */
    virtual ~SimulationBase();

    /**
     * @brief Setup the simulation
     *
     * Initializes bodies with specified distribution and transfers them to the device.
     */
    virtual void setup();

    /**
     * @brief Set the distribution and seed for the simulation
     * @param dist Distribution type
     * @param seed Random seed
     */
    void setDistributionAndSeed(BodyDistribution dist, unsigned int seed)
    {
        distribution = dist;
        randomSeed = seed;
    }

    /**
     * @brief Update the simulation
     *
     * This method must be implemented by derived classes to advance
     * the simulation by one time step.
     */
    virtual void update() = 0;

    /**
     * @brief Copy body data from host to device
     */
    void copyBodiesToDevice();

    /**
     * @brief Copy body data from device to host
     */
    void copyBodiesFromDevice();

    /**
     * @brief Get the bodies array
     * @return Pointer to the host bodies array
     */
    Body *getBodies() const
    {
        return h_bodies;
    }

    /**
     * @brief Get the bodies array
     * @return Pointer to the device bodies array
     */
    Body *getDeviceBodies() const
    {
        return d_bodies;
    }

    /**
     * @brief Get the number of bodies
     * @return Number of bodies in the simulation
     */
    int getNumBodies() const
    {
        return nBodies;
    }

    /**
     * @brief Get the performance metrics
     * @return Reference to the simulation metrics
     */
    const SimulationMetrics &getMetrics() const
    {
        return metrics;
    }
};

#endif // BASE_CUH