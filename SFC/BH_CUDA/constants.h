#ifndef CONSTANTS_H
#define CONSTANTS_H

#define HBL 1.6e29
#define WINDOW_WIDTH 2048
#define WINDOW_HEIGHT 2048
#define NBODY_WIDTH 10.0e11
#define NBODY_HEIGHT 10.0e11

// Simulation parameters
#define GRAVITY 6.67430e-11
#define E 0.5              // Softening factor for avoiding div by 0
#define DT 25000.0         // Time step
#define THETA 0.5          // Multipole acceptance criterion
#define COLLISION_TH 1.0e10 // Reduced collision threshold (was 1.0e10)

// Kernel/Blocks
#define BLOCK_SIZE 1024
#define MAX_NODES 349525
#define N_LEAF 262144

// Astronomical constants
#define MAX_DIST 5.0e11 // Maximum distance for initial distribution
#define MIN_DIST 2.0e10 // Minimum distance for initial distribution
#define EARTH_MASS 5.974e24
#define EARTH_DIA 12756.0
#define SUN_MASS 1.989e30
#define SUN_DIA 1.3927e6
#define CENTERX 0
#define CENTERY 0
#define CENTERZ 0

#endif // CONSTANTS_H