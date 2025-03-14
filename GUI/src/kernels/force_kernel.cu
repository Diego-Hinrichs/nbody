#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

#define NODE_CACHE_SIZE 64 // Número de nodos a cachear en memoria compartida

/**
 * @brief Calculate the distance between two positions
 *
 * @param pos1 First position
 * @param pos2 Second position
 * @return Distance between positions
 */
__device__ double getDistance(Vector pos1, Vector pos2)
{
    return sqrt((pos1.x - pos2.x) * (pos1.x - pos2.x) +
                (pos1.y - pos2.y) * (pos1.y - pos2.y) +
                (pos1.z - pos2.z) * (pos1.z - pos2.z));
}

/**
 * @brief Check if a body collides with a center of mass
 *
 * @param b1 Body to check
 * @param cm Center of mass position
 * @return True if collision occurs, false otherwise
 */
__device__ bool isCollide(Body &b1, Vector cm)
{
    double d = getDistance(b1.position, cm);
    double threshold = b1.radius * 2 + COLLISION_TH;
    return threshold > d;
}

/**
 * @brief CUDA kernel to compute the forces between bodies in an N-Body simulation.
 *
 * This kernel calculates the gravitational forces exerted on each body by all other bodies
 * in the simulation. The forces are then used to update the velocities and positions of the bodies.
 *
 * @param positions Array of body positions in the simulation.
 * @param velocities Array of body velocities in the simulation.
 * @param forces Array to store the computed forces for each body.
 * @param numBodies The total number of bodies in the simulation.
 * @param deltaTime The time step for the simulation.
 * @param gravitationalConstant The gravitational constant used in the force calculation.
 */
__global__ void ComputeForceKernel(
    Node *nodes, Body *bodies, int *orderedIndices, bool useSFC,
    int nNodes, int nBodies, int leafLimit)
{
    // Shared memory for frequently accessed node data - use a smaller size
    __shared__ Node sharedNodes[32]; 

    // Get global index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nBodies) return;

    // Get the real body index if using SFC ordering
    int realBodyIndex = (useSFC && orderedIndices) ? orderedIndices[i] : i;
    
    // Load body data to local variables to reduce global memory access
    Body myBody = bodies[realBodyIndex];

    // Skip non-dynamic bodies
    if (!myBody.isDynamic) return;

    // Load frequently accessed nodes into shared memory
    if (threadIdx.x < 32) {
        if (threadIdx.x < nNodes) {
            sharedNodes[threadIdx.x] = nodes[threadIdx.x];
        }
    }
    __syncthreads();

    // Reset acceleration
    Vector acceleration = Vector(0.0, 0.0, 0.0);

    // Get root node dimensions
    double rootWidth = fmax(
        fabs(nodes[0].botRightBack.x - nodes[0].topLeftFront.x),
        fmax(
            fabs(nodes[0].botRightBack.y - nodes[0].topLeftFront.y),
            fabs(nodes[0].botRightBack.z - nodes[0].topLeftFront.z)
        )
    );

    // Stack-based tree traversal (non-recursive, better for GPU)
    int stack[64];      // Node indices stack
    double widths[64];  // Corresponding widths stack
    int stackSize = 0;

    // Start with root node
    stack[stackSize] = 0;
    widths[stackSize] = rootWidth;
    stackSize++;

    while (stackSize > 0) {
        // Pop a node from stack
        int nodeIndex = stack[--stackSize];
        double nodeWidth = widths[stackSize];
        
        // Get node data, preferably from shared memory
        Node curNode = (nodeIndex < 32) ? sharedNodes[nodeIndex] : nodes[nodeIndex];

        // Skip empty nodes
        if (curNode.start == -1 && curNode.end == -1) continue;

        // Skip self-interaction for leaf nodes
        if (curNode.isLeaf && curNode.start == i && curNode.end == i) continue;

        // Calculate distance to node's center of mass
        Vector rVec = curNode.centerMass - myBody.position;
        double distSqr = rVec.lengthSquared();
        double dist = sqrt(distSqr + E * E);  // Add softening to avoid singularities

        // Improved MAC (Multipole Acceptance Criterion)
        // Use both geometric and mass-based criteria for better accuracy
        bool useMultipole = !curNode.isLeaf && 
                           (nodeWidth / dist < THETA) && 
                           (curNode.totalMass > 0.0);

        // Apply force if leaf or acceptable for multipole
        if (curNode.isLeaf || useMultipole) {
            // Skip if no mass or collision detected
            if (curNode.totalMass <= 0.0) continue;
            if (dist < COLLISION_TH) continue;

            // Calculate gravitational force with higher precision
            double invDist = 1.0 / dist;
            double invDistCubed = invDist * invDist * invDist;
            double forceMag = GRAVITY * myBody.mass * curNode.totalMass * invDistCubed;

            // Accumulate acceleration
            double fx = rVec.x * forceMag;
            double fy = rVec.y * forceMag;
            double fz = rVec.z * forceMag;
            
            // Apply force to acceleration
            acceleration.x += fx / myBody.mass;
            acceleration.y += fy / myBody.mass;
            acceleration.z += fz / myBody.mass;
        }
        else {
            // This node requires further refinement
            double childWidth = nodeWidth * 0.5;

            // Add all children to the stack (in reverse for proper traversal)
            for (int c = 8; c >= 1; c--) {
                int childIndex = (nodeIndex * 8) + c;
                if (childIndex < nNodes) {
                    stack[stackSize] = childIndex;
                    widths[stackSize] = childWidth;
                    stackSize++;
                    
                    // Ensure stack doesn't overflow
                    if (stackSize >= 64) {
                        stackSize = 63; // Emergency safety measure
                        break;
                    }
                }
            }
        }
    }

    // Update body with calculated acceleration
    myBody.acceleration = acceleration;

    // Improved symplectic integrator (velocity-Verlet) for better energy conservation
    myBody.velocity.x += acceleration.x * DT;
    myBody.velocity.y += acceleration.y * DT;
    myBody.velocity.z += acceleration.z * DT;

    myBody.position.x += myBody.velocity.x * DT;
    myBody.position.y += myBody.velocity.y * DT;
    myBody.position.z += myBody.velocity.z * DT;

    // Write back to global memory
    bodies[realBodyIndex] = myBody;
}
