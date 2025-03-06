#ifndef SIMULATION_BASE_CUH
#define SIMULATION_BASE_CUH

#include "../common/types.cuh"
#include "../common/constants.cuh"
#include "../common/error_handling.cuh"

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

    SimulationMetrics metrics; // Performance metrics

    // Flag to indicate whether the simulation is initialized
    bool isInitialized;

    /**
     * @brief Initialize bodies with random positions and velocities
     */
    virtual void initRandomBodies();

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
     */
    SimulationBase(int numBodies);

    /**
     * @brief Virtual destructor
     */
    virtual ~SimulationBase();

    /**
     * @brief Setup the simulation
     *
     * Initializes random bodies and transfers them to the device.
     */
    virtual void setup();

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

#endif // SIMULATION_BASE_CUH