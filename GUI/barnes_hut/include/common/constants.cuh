#ifndef CONSTANTS_CUH
#define CONSTANTS_CUH

// Display and visualization constants
#define WINDOW_WIDTH 2048
#define WINDOW_HEIGHT 2048
#define NBODY_WIDTH 10.0e11
#define NBODY_HEIGHT 10.0e11

// Physical constants
#define GRAVITY 6.67430e-11 // Gravitational constant
#define E 0.5               // Softening factor for avoiding div by 0
#define DT 25000.0          // Time step in seconds
#define THETA 0.5           // Multipole acceptance criterion
#define COLLISION_TH 1.0e10 // Collision threshold distance

// Simulation constants
#define BLOCK_SIZE 1024  // CUDA block size
#define MAX_NODES 349525 // Maximum number of nodes in the octree
#define N_LEAF 262144    // Leaf threshold (affects recursion depth)

// Astronomical constants
#define MAX_DIST 5.0e11     // Maximum distance for initial distribution
#define MIN_DIST 2.0e10     // Minimum distance for initial distribution
#define EARTH_MASS 5.974e24 // Mass of Earth in kg
#define EARTH_DIA 12756.0   // Diameter of Earth in km
#define SUN_MASS 1.989e30   // Mass of Sun in kg
#define SUN_DIA 1.3927e6    // Diameter of Sun in km
#define CENTERX 0           // Center of simulation X coordinate
#define CENTERY 0           // Center of simulation Y coordinate
#define CENTERZ 0           // Center of simulation Z coordinate

// Morton code constants
#define MORTON_BITS 21 // Number of bits per dimension for Morton codes

// Implementation constants
#define MAX_REORDER_BODY_SIZE 20000 // Maximum size for body array reordering

#endif // CONSTANTS_CUH