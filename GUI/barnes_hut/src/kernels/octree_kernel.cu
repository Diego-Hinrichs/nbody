#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

/**
 * @brief Get the octant for a 3D point within a node's bounding box
 *
 * Determines which of the eight octants contains the given point.
 * Octants are numbered from 1 to 8 as follows:
 * 1: top-left-front
 * 2: top-right-front
 * 3: bottom-left-front
 * 4: bottom-right-front
 * 5: top-left-back
 * 6: top-right-back
 * 7: bottom-left-back
 * 8: bottom-right-back
 *
 * @param topLeftFront Top-left-front corner of the bounding box
 * @param botRightBack Bottom-right-back corner of the bounding box
 * @param x X coordinate
 * @param y Y coordinate
 * @param z Z coordinate
 * @return Octant number (1-8)
 */
__device__ int getOctant(Vector topLeftFront, Vector botRightBack, double x, double y, double z)
{
    int octant = 1;
    double midX = (topLeftFront.x + botRightBack.x) / 2;
    double midY = (topLeftFront.y + botRightBack.y) / 2;
    double midZ = (topLeftFront.z + botRightBack.z) / 2;

    if (x <= midX)
    {
        if (y >= midY)
        {
            if (z <= midZ)
                octant = 1; // top-left-front
            else
                octant = 5; // top-left-back
        }
        else
        {
            if (z <= midZ)
                octant = 3; // bottom-left-front
            else
                octant = 7; // bottom-left-back
        }
    }
    else
    {
        if (y >= midY)
        {
            if (z <= midZ)
                octant = 2; // top-right-front
            else
                octant = 6; // top-right-back
        }
        else
        {
            if (z <= midZ)
                octant = 4; // bottom-right-front
            else
                octant = 8; // bottom-right-back
        }
    }
    return octant;
}

/**
 * @brief Update the bounding box of a child node based on its octant
 *
 * Sets the bounds of a child node based on its parent's bounds and the octant number.
 *
 * @param tlf Top-left-front corner of the parent node
 * @param brb Bottom-right-back corner of the parent node
 * @param childNode Reference to the child node to update
 * @param octant Octant number (1-8)
 */
__device__ void UpdateChildBound(Vector &tlf, Vector &brb, Node &childNode, int octant)
{
    double midX = (tlf.x + brb.x) / 2;
    double midY = (tlf.y + brb.y) / 2;
    double midZ = (tlf.z + brb.z) / 2;

    switch (octant)
    {
    case 1: // top-left-front
        childNode.topLeftFront = tlf;
        childNode.botRightBack = Vector(midX, midY, midZ);
        break;
    case 2: // top-right-front
        childNode.topLeftFront = Vector(midX, tlf.y, tlf.z);
        childNode.botRightBack = Vector(brb.x, midY, midZ);
        break;
    case 3: // bottom-left-front
        childNode.topLeftFront = Vector(tlf.x, midY, tlf.z);
        childNode.botRightBack = Vector(midX, brb.y, midZ);
        break;
    case 4: // bottom-right-front
        childNode.topLeftFront = Vector(midX, midY, tlf.z);
        childNode.botRightBack = Vector(brb.x, brb.y, midZ);
        break;
    case 5: // top-left-back
        childNode.topLeftFront = Vector(tlf.x, tlf.y, midZ);
        childNode.botRightBack = Vector(midX, midY, brb.z);
        break;
    case 6: // top-right-back
        childNode.topLeftFront = Vector(midX, tlf.y, midZ);
        childNode.botRightBack = Vector(brb.x, midY, brb.z);
        break;
    case 7: // bottom-left-back
        childNode.topLeftFront = Vector(tlf.x, midY, midZ);
        childNode.botRightBack = Vector(midX, brb.y, brb.z);
        break;
    case 8: // bottom-right-back
        childNode.topLeftFront = Vector(midX, midY, midZ);
        childNode.botRightBack = brb;
        break;
    }
}

/**
 * @brief Warp-level reduction for mass and center of mass calculations
 *
 * Optimized for warp-level (32 threads) reduction without shared memory bank conflicts.
 *
 * @param totalMass Array of total mass values
 * @param centerMass Array of center of mass values
 * @param tx Thread index within the warp
 */
__device__ void warpReduce(volatile double *totalMass, volatile double3 *centerMass, int tx)
{
    totalMass[tx] += totalMass[tx + 32];
    centerMass[tx].x += centerMass[tx + 32].x;
    centerMass[tx].y += centerMass[tx + 32].y;
    centerMass[tx].z += centerMass[tx + 32].z;

    totalMass[tx] += totalMass[tx + 16];
    centerMass[tx].x += centerMass[tx + 16].x;
    centerMass[tx].y += centerMass[tx + 16].y;
    centerMass[tx].z += centerMass[tx + 16].z;

    totalMass[tx] += totalMass[tx + 8];
    centerMass[tx].x += centerMass[tx + 8].x;
    centerMass[tx].y += centerMass[tx + 8].y;
    centerMass[tx].z += centerMass[tx + 8].z;

    totalMass[tx] += totalMass[tx + 4];
    centerMass[tx].x += centerMass[tx + 4].x;
    centerMass[tx].y += centerMass[tx + 4].y;
    centerMass[tx].z += centerMass[tx + 4].z;

    totalMass[tx] += totalMass[tx + 2];
    centerMass[tx].x += centerMass[tx + 2].x;
    centerMass[tx].y += centerMass[tx + 2].y;
    centerMass[tx].z += centerMass[tx + 2].z;

    totalMass[tx] += totalMass[tx + 1];
    centerMass[tx].x += centerMass[tx + 1].x;
    centerMass[tx].y += centerMass[tx + 1].y;
    centerMass[tx].z += centerMass[tx + 1].z;
}

/**
 * @brief Compute the center of mass for a node
 *
 * Calculates the total mass and center of mass for a node based on
 * the bodies it contains.
 *
 * @param curNode Current node to update
 * @param bodies Array of bodies
 * @param totalMass Shared memory array for total mass
 * @param centerMass Shared memory array for center of mass
 * @param start Start index of bodies
 * @param end End index of bodies
 */
__device__ void ComputeCenterMass(Node &curNode, Body *bodies, double *totalMass, double3 *centerMass, int start, int end)
{
    int tx = threadIdx.x;
    int total = end - start + 1;
    int sz = ceil((double)total / blockDim.x);
    int s = tx * sz + start;
    double M = 0.0;
    double3 R = make_double3(0.0, 0.0, 0.0);

    // Each thread processes a subset of bodies
    for (int i = s; i < s + sz; ++i)
    {
        if (i <= end)
        {
            Body &body = bodies[i];
            M += body.mass;
            R.x += body.mass * body.position.x;
            R.y += body.mass * body.position.y;
            R.z += body.mass * body.position.z;
        }
    }

    // Store thread's partial results
    totalMass[tx] = M;
    centerMass[tx] = R;

    // Block-level reduction
    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        __syncthreads();
        if (tx < stride)
        {
            totalMass[tx] += totalMass[tx + stride];
            centerMass[tx].x += centerMass[tx + stride].x;
            centerMass[tx].y += centerMass[tx + stride].y;
            centerMass[tx].z += centerMass[tx + stride].z;
        }
    }

    // Warp-level reduction for the final 32 threads
    if (tx < 32)
    {
        warpReduce(totalMass, centerMass, tx);
    }

    __syncthreads();

    // Update node with final results
    if (tx == 0)
    {
        double mass = totalMass[0];
        if (mass > 0.0)
        {
            centerMass[0].x /= mass;
            centerMass[0].y /= mass;
            centerMass[0].z /= mass;
        }
        curNode.totalMass = mass;
        curNode.centerMass = Vector(centerMass[0].x, centerMass[0].y, centerMass[0].z);
    }
}

/**
 * @brief Count bodies in each octant
 *
 * Counts how many bodies fall into each of the 8 octants of a node.
 *
 * @param bodies Array of bodies
 * @param topLeftFront Top-left-front corner of the node
 * @param botRightBack Bottom-right-back corner of the node
 * @param count Output array for counts (8 octants)
 * @param start Start index of bodies
 * @param end End index of bodies
 * @param nBodies Total number of bodies
 */
__device__ void CountBodies(Body *bodies, Vector topLeftFront, Vector botRightBack, int *count, int start, int end, int nBodies)
{
    int tx = threadIdx.x;

    // Initialize counters
    if (tx < 8)
    {
        count[tx] = 0;
    }
    __syncthreads();

    // Each thread processes a subset of bodies
    for (int i = start + tx; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);
        atomicAdd(&count[oct - 1], 1);
    }
    __syncthreads();
}

/**
 * @brief Compute offsets for each octant
 *
 * Calculates starting indices for bodies in each octant based on counts.
 *
 * @param count Array of counts and output array for offsets
 * @param start Start index
 */
__device__ void ComputeOffset(int *count, int start)
{
    int tx = threadIdx.x;

    // First 8 elements of count contain the counts
    // Next 8 elements will store the offsets
    if (tx < 8)
    {
        int offset = start;
        for (int i = 0; i < tx; ++i)
        {
            offset += count[i];
        }
        count[tx + 8] = offset;
    }
    __syncthreads();
}

/**
 * @brief Group bodies by octant
 *
 * Reorders bodies in the buffer according to their octant.
 *
 * @param bodies Original body array
 * @param buffer Output buffer for reordered bodies
 * @param topLeftFront Top-left-front corner of the node
 * @param botRightBack Bottom-right-back corner of the node
 * @param workOffset Working array of offsets (will be modified)
 * @param start Start index
 * @param end End index
 * @param nBodies Total number of bodies
 */
__device__ void GroupBodies(Body *bodies, Body *buffer, Vector topLeftFront, Vector botRightBack, int *workOffset, int start, int end, int nBodies)
{
    // Each thread processes a subset of bodies
    for (int i = start + threadIdx.x; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);

        // Get destination index with atomic increment
        int dest = atomicAdd(&workOffset[oct - 1], 1);

        // Copy body to its new position
        buffer[dest] = body;
    }
    __syncthreads();
}

/**
 * @brief Construct the octree recursively
 *
 * Builds the Barnes-Hut octree by recursively partitioning space and
 * grouping bodies according to their octants.
 *
 * @param node Array of octree nodes
 * @param bodies Array of input bodies
 * @param buffer Array for temporary body storage during reordering
 * @param nodeIndex Index of the current node
 * @param nNodes Total number of nodes
 * @param nBodies Total number of bodies
 * @param leafLimit Limit for leaf nodes (controls recursion depth)
 */
__global__ void ConstructOctTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit)
{
    // Shared memory for counting and offset calculation
    __shared__ int count[16]; // First 8: counts, Next 8: base offsets
    __shared__ double totalMass[BLOCK_SIZE];
    __shared__ double3 centerMass[BLOCK_SIZE];
    __shared__ int baseOffset[8];
    __shared__ int workOffset[8];

    int tx = threadIdx.x;

    // Adjust node index by block index (for recursive calls)
    nodeIndex += blockIdx.x;
    if (nodeIndex >= nNodes)
    {
        return;
    }

    Node &curNode = node[nodeIndex];
    int start = curNode.start;
    int end = curNode.end;

    // Skip empty nodes
    if (start == -1 && end == -1)
    {
        return;
    }

    Vector topLeftFront = curNode.topLeftFront;
    Vector botRightBack = curNode.botRightBack;

    // Calculate center of mass for current node
    ComputeCenterMass(curNode, bodies, totalMass, centerMass, start, end);

    // Check if we've reached the recursion limit or have only one body
    if (nodeIndex >= leafLimit || start == end)
    {
        // Just copy bodies and return
        for (int i = start + tx; i <= end; i += blockDim.x)
        {
            buffer[i] = bodies[i];
        }
        return;
    }

    // Step 1: Count bodies in each octant
    CountBodies(bodies, topLeftFront, botRightBack, count, start, end, nBodies);

    // Step 2: Calculate base offsets for each octant
    ComputeOffset(count, start);

    // Copy offsets to local arrays for grouping
    if (tx < 8)
    {
        baseOffset[tx] = count[tx + 8];  // Save original offset
        workOffset[tx] = baseOffset[tx]; // Initialize working copy for atomic ops
    }
    __syncthreads();

    // Step 3: Group bodies by octant
    GroupBodies(bodies, buffer, topLeftFront, botRightBack, workOffset, start, end, nBodies);

    // Step 4: Assign ranges to child nodes (only thread 0)
    if (tx == 0)
    {
        // Mark current node as non-leaf
        curNode.isLeaf = false;

        // For each octant, setup child node
        for (int i = 0; i < 8; i++)
        {
            int childIdx = nodeIndex * 8 + (i + 1);
            if (childIdx < nNodes)
            {
                Node &childNode = node[childIdx];

                // Set child node bounding box
                UpdateChildBound(topLeftFront, botRightBack, childNode, i + 1);

                // Assign body range if this octant has bodies
                if (count[i] > 0)
                {
                    childNode.start = baseOffset[i];
                    childNode.end = baseOffset[i] + count[i] - 1;
                }
                else
                {
                    childNode.start = -1;
                    childNode.end = -1;
                }
            }
        }

        // Recursively process child nodes (launch 8 blocks)
        ConstructOctTreeKernel<<<8, BLOCK_SIZE>>>(node, buffer, bodies, nodeIndex * 8 + 1, nNodes, nBodies, leafLimit);
    }
}