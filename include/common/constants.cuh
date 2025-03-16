#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

namespace SimulationConstants
{
    // Display and visualization constants
    constexpr int WINDOW_WIDTH = 2048;
    constexpr int WINDOW_HEIGHT = 2048;
    constexpr double NBODY_WIDTH = 10.0e11;
    constexpr double NBODY_HEIGHT = 10.0e11;

    // Physical constants
    constexpr double GRAVITY = 6.67430e-11;        // Gravitational constant
    constexpr double SOFTENING_FACTOR = 0.5;       // Softening factor for avoiding div by 0
    constexpr double TIME_STEP = 25000.0;          // Time step in seconds
    constexpr double THETA = 0.0;                  // Multipole acceptance criterion
    constexpr double COLLISION_THRESHOLD = 1.0e10; // Collision threshold distance

    // Simulation constants
    constexpr int BLOCK_SIZE = 1024;  // CUDA block size
    constexpr int MAX_NODES = 349525; // Maximum number of nodes in the octree
    constexpr int N_LEAF = 262144;    // Leaf threshold (affects recursion depth)

    // Astronomical constants
    constexpr double MAX_DIST = 5.0e11;     // Maximum distance for initial distribution
    constexpr double MIN_DIST = 2.0e10;     // Minimum distance for initial distribution
    constexpr double EARTH_MASS = 5.974e24; // Mass of Earth in kg
    constexpr double EARTH_DIA = 12756.0;   // Diameter of Earth in km
    constexpr double SUN_MASS = 1.989e30;   // Mass of Sun in kg
    constexpr double SUN_DIA = 1.3927e6;    // Diameter of Sun in km
    constexpr double CENTERX = 0;           // Center of simulation X coordinate
    constexpr double CENTERY = 0;           // Center of simulation Y coordinate
    constexpr double CENTERZ = 0;           // Center of simulation Z coordinate

    // Morton code constants
    constexpr int MORTON_BITS = 21; // Number of bits per dimension for Morton codes

    // Implementation constants
    constexpr int MAX_REORDER_BODY_SIZE = 20000; // Maximum size for body array reordering
}

// Compatibility macros for existing code
#define WINDOW_WIDTH SimulationConstants::WINDOW_WIDTH
#define WINDOW_HEIGHT SimulationConstants::WINDOW_HEIGHT
#define NBODY_WIDTH SimulationConstants::NBODY_WIDTH
#define NBODY_HEIGHT SimulationConstants::NBODY_HEIGHT
#define GRAVITY SimulationConstants::GRAVITY
#define E SimulationConstants::SOFTENING_FACTOR
#define DT SimulationConstants::TIME_STEP
#define THETA SimulationConstants::THETA
#define COLLISION_TH SimulationConstants::COLLISION_THRESHOLD
#define BLOCK_SIZE SimulationConstants::BLOCK_SIZE
#define MAX_NODES SimulationConstants::MAX_NODES
#define N_LEAF SimulationConstants::N_LEAF
#define MAX_DIST SimulationConstants::MAX_DIST
#define MIN_DIST SimulationConstants::MIN_DIST
#define EARTH_MASS SimulationConstants::EARTH_MASS
#define EARTH_DIA SimulationConstants::EARTH_DIA
#define SUN_MASS SimulationConstants::SUN_MASS
#define SUN_DIA SimulationConstants::SUN_DIA
#define CENTERX SimulationConstants::CENTERX
#define CENTERY SimulationConstants::CENTERY
#define CENTERZ SimulationConstants::CENTERZ
#define MORTON_BITS SimulationConstants::MORTON_BITS
#define MAX_REORDER_BODY_SIZE SimulationConstants::MAX_REORDER_BODY_SIZE

#endif // CONSTANTS_CUH