// #include "../../include/common/types.cuh"
// #include "../../include/common/constants.cuh"

// /**
//  * @brief Get the octant for a 3D point within a node's bounding box
//  *
//  * Determines which of the eight octants contains the given point.
//  * Octants are numbered from 1 to 8 as follows:
//  * 1: top-left-front
//  * 2: top-right-front
//  * 3: bottom-left-front
//  * 4: bottom-right-front
//  * 5: top-left-back
//  * 6: top-right-back
//  * 7: bottom-left-back
//  * 8: bottom-right-back
//  *
//  * @param topLeftFront Top-left-front corner of the bounding box
//  * @param botRightBack Bottom-right-back corner of the bounding box
//  * @param x X coordinate
//  * @param y Y coordinate
//  * @param z Z coordinate
//  * @return Octant number (1-8)
//  */
// __device__ int getOctant(Vector topLeftFront, Vector botRightBack, double x, double y, double z)
// {
//     int octant = 1;
//     double midX = (topLeftFront.x + botRightBack.x) / 2;
//     double midY = (topLeftFront.y + botRightBack.y) / 2;
//     double midZ = (topLeftFront.z + botRightBack.z) / 2;

//     if (x <= midX)
//     {
//         if (y >= midY)
//         {
//             if (z <= midZ)
//                 octant = 1; // top-left-front
//             else
//                 octant = 5; // top-left-back
//         }
//         else
//         {
//             if (z <= midZ)
//                 octant = 3; // bottom-left-front
//             else
//                 octant = 7; // bottom-left-back
//         }
//     }
//     else
//     {
//         if (y >= midY)
//         {
//             if (z <= midZ)
//                 octant = 2; // top-right-front
//             else
//                 octant = 6; // top-right-back
//         }
//         else
//         {
//             if (z <= midZ)
//                 octant = 4; // bottom-right-front
//             else
//                 octant = 8; // bottom-right-back
//         }
//     }
//     return octant;
// }

// /**
//  * @brief Update the bounding box of a child node based on its octant
//  *
//  * Sets the bounds of a child node based on its parent's bounds and the octant number.
//  *
//  * @param tlf Top-left-front corner of the parent node
//  * @param brb Bottom-right-back corner of the parent node
//  * @param childNode Reference to the child node to update
//  * @param octant Octant number (1-8)
//  */
// __device__ void UpdateChildBound(Vector &tlf, Vector &brb, Node &childNode, int octant)
// {
//     double midX = (tlf.x + brb.x) / 2;
//     double midY = (tlf.y + brb.y) / 2;
//     double midZ = (tlf.z + brb.z) / 2;

//     switch (octant)
//     {
//     case 1: // top-left-front
//         childNode.topLeftFront = tlf;
//         childNode.botRightBack = Vector(midX, midY, midZ);
//         break;
//     case 2: // top-right-front
//         childNode.topLeftFront = Vector(midX, tlf.y, tlf.z);
//         childNode.botRightBack = Vector(brb.x, midY, midZ);
//         break;
//     case 3: // bottom-left-front
//         childNode.topLeftFront = Vector(tlf.x, midY, tlf.z);
//         childNode.botRightBack = Vector(midX, brb.y, midZ);
//         break;
//     case 4: // bottom-right-front
//         childNode.topLeftFront = Vector(midX, midY, tlf.z);
//         childNode.botRightBack = Vector(brb.x, brb.y, midZ);
//         break;
//     case 5: // top-left-back
//         childNode.topLeftFront = Vector(tlf.x, tlf.y, midZ);
//         childNode.botRightBack = Vector(midX, midY, brb.z);
//         break;
//     case 6: // top-right-back
//         childNode.topLeftFront = Vector(midX, tlf.y, midZ);
//         childNode.botRightBack = Vector(brb.x, midY, brb.z);
//         break;
//     case 7: // bottom-left-back
//         childNode.topLeftFront = Vector(tlf.x, midY, midZ);
//         childNode.botRightBack = Vector(midX, brb.y, brb.z);
//         break;
//     case 8: // bottom-right-back
//         childNode.topLeftFront = Vector(midX, midY, midZ);
//         childNode.botRightBack = brb;
//         break;
//     }
// }

// /**
//  * @brief Warp-level reduction for mass and center of mass calculations
//  *
//  * Optimized for warp-level (32 threads) reduction without shared memory bank conflicts.
//  *
//  * @param totalMass Array of total mass values
//  * @param centerMass Array of center of mass values
//  * @param tx Thread index within the warp
//  */
// __device__ void warpReduce(volatile double *totalMass, volatile double3 *centerMass, int tx)
// {
//     totalMass[tx] += totalMass[tx + 32];
//     centerMass[tx].x += centerMass[tx + 32].x;
//     centerMass[tx].y += centerMass[tx + 32].y;
//     centerMass[tx].z += centerMass[tx + 32].z;

//     totalMass[tx] += totalMass[tx + 16];
//     centerMass[tx].x += centerMass[tx + 16].x;
//     centerMass[tx].y += centerMass[tx + 16].y;
//     centerMass[tx].z += centerMass[tx + 16].z;

//     totalMass[tx] += totalMass[tx + 8];
//     centerMass[tx].x += centerMass[tx + 8].x;
//     centerMass[tx].y += centerMass[tx + 8].y;
//     centerMass[tx].z += centerMass[tx + 8].z;

//     totalMass[tx] += totalMass[tx + 4];
//     centerMass[tx].x += centerMass[tx + 4].x;
//     centerMass[tx].y += centerMass[tx + 4].y;
//     centerMass[tx].z += centerMass[tx + 4].z;

//     totalMass[tx] += totalMass[tx + 2];
//     centerMass[tx].x += centerMass[tx + 2].x;
//     centerMass[tx].y += centerMass[tx + 2].y;
//     centerMass[tx].z += centerMass[tx + 2].z;

//     totalMass[tx] += totalMass[tx + 1];
//     centerMass[tx].x += centerMass[tx + 1].x;
//     centerMass[tx].y += centerMass[tx + 1].y;
//     centerMass[tx].z += centerMass[tx + 1].z;
// }

// /**
//  * @brief Compute the center of mass for a node
//  *
//  * Calculates the total mass and center of mass for a node based on
//  * the bodies it contains.
//  *
//  * @param curNode Current node to update
//  * @param bodies Array of bodies
//  * @param totalMass Shared memory array for total mass
//  * @param centerMass Shared memory array for center of mass
//  * @param start Start index of bodies
//  * @param end End index of bodies
//  */
// __device__ void ComputeCenterMass(Node &curNode, Body *bodies, double *totalMass, double3 *centerMass, int start, int end)
// {
//     int tx = threadIdx.x;
//     int total = end - start + 1;
//     int sz = ceil((double)total / blockDim.x);
//     int s = tx * sz + start;
//     double M = 0.0;
//     double3 R = make_double3(0.0, 0.0, 0.0);

//     // Each thread processes a subset of bodies
//     for (int i = s; i < s + sz; ++i)
//     {
//         if (i <= end)
//         {
//             Body &body = bodies[i];
//             M += body.mass;
//             R.x += body.mass * body.position.x;
//             R.y += body.mass * body.position.y;
//             R.z += body.mass * body.position.z;
//         }
//     }

//     // Store thread's partial results
//     totalMass[tx] = M;
//     centerMass[tx] = R;

//     // Block-level reduction
//     for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
//     {
//         __syncthreads();
//         if (tx < stride)
//         {
//             totalMass[tx] += totalMass[tx + stride];
//             centerMass[tx].x += centerMass[tx + stride].x;
//             centerMass[tx].y += centerMass[tx + stride].y;
//             centerMass[tx].z += centerMass[tx + stride].z;
//         }
//     }

//     // Warp-level reduction for the final 32 threads
//     if (tx < 32)
//     {
//         warpReduce(totalMass, centerMass, tx);
//     }

//     __syncthreads();

//     // Update node with final results
//     if (tx == 0)
//     {
//         double mass = totalMass[0];
//         if (mass > 0.0)
//         {
//             centerMass[0].x /= mass;
//             centerMass[0].y /= mass;
//             centerMass[0].z /= mass;
//         }
//         curNode.totalMass = mass;
//         curNode.centerMass = Vector(centerMass[0].x, centerMass[0].y, centerMass[0].z);
//     }
// }

// /**
//  * @brief Count bodies in each octant
//  *
//  * Counts how many bodies fall into each of the 8 octants of a node.
//  *
//  * @param bodies Array of bodies
//  * @param topLeftFront Top-left-front corner of the node
//  * @param botRightBack Bottom-right-back corner of the node
//  * @param count Output array for counts (8 octants)
//  * @param start Start index of bodies
//  * @param end End index of bodies
//  * @param nBodies Total number of bodies
//  */
// __device__ void CountBodies(Body *bodies, Vector topLeftFront, Vector botRightBack, int *count, int start, int end, int nBodies)
// {
//     int tx = threadIdx.x;

//     // Initialize counters
//     if (tx < 8)
//     {
//         count[tx] = 0;
//     }
//     __syncthreads();

//     // Each thread processes a subset of bodies
//     for (int i = start + tx; i <= end; i += blockDim.x)
//     {
//         Body body = bodies[i];
//         int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);
//         atomicAdd(&count[oct - 1], 1);
//     }
//     __syncthreads();
// }

// /**
//  * @brief Compute offsets for each octant
//  *
//  * Calculates starting indices for bodies in each octant based on counts.
//  *
//  * @param count Array of counts and output array for offsets
//  * @param start Start index
//  */
// __device__ void ComputeOffset(int *count, int start)
// {
//     int tx = threadIdx.x;

//     // First 8 elements of count contain the counts
//     // Next 8 elements will store the offsets
//     if (tx < 8)
//     {
//         int offset = start;
//         for (int i = 0; i < tx; ++i)
//         {
//             offset += count[i];
//         }
//         count[tx + 8] = offset;
//     }
//     __syncthreads();
// }

// /**
//  * @brief Group bodies by octant
//  *
//  * Reorders bodies in the buffer according to their octant.
//  *
//  * @param bodies Original body array
//  * @param buffer Output buffer for reordered bodies
//  * @param topLeftFront Top-left-front corner of the node
//  * @param botRightBack Bottom-right-back corner of the node
//  * @param workOffset Working array of offsets (will be modified)
//  * @param start Start index
//  * @param end End index
//  * @param nBodies Total number of bodies
//  */
// __device__ void GroupBodies(Body *bodies, Body *buffer, Vector topLeftFront, Vector botRightBack, int *workOffset, int start, int end, int nBodies)
// {
//     // Each thread processes a subset of bodies
//     for (int i = start + threadIdx.x; i <= end; i += blockDim.x)
//     {
//         Body body = bodies[i];
//         int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);

//         // Get destination index with atomic increment
//         int dest = atomicAdd(&workOffset[oct - 1], 1);

//         // Copy body to its new position
//         buffer[dest] = body;
//     }
//     __syncthreads();
// }

// /**
//  * @brief Construct the octree recursively
//  *
//  * Builds the Barnes-Hut octree by recursively partitioning space and
//  * grouping bodies according to their octants.
//  *
//  * @param node Array of octree nodes
//  * @param bodies Array of input bodies
//  * @param buffer Array for temporary body storage during reordering
//  * @param nodeIndex Index of the current node
//  * @param nNodes Total number of nodes
//  * @param nBodies Total number of bodies
//  * @param leafLimit Limit for leaf nodes (controls recursion depth)
//  */
// __global__ void ConstructOctTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit)
// {
//     // Shared memory for counting and offset calculation
//     __shared__ int count[16]; // First 8: counts, Next 8: base offsets
//     __shared__ double totalMass[BLOCK_SIZE];
//     __shared__ double3 centerMass[BLOCK_SIZE];
//     __shared__ int baseOffset[8];
//     __shared__ int workOffset[8];

//     int tx = threadIdx.x;

//     // Adjust node index by block index (for recursive calls)
//     nodeIndex += blockIdx.x;
//     if (nodeIndex >= nNodes)
//     {
//         return;
//     }

//     Node &curNode = node[nodeIndex];
//     int start = curNode.start;
//     int end = curNode.end;

//     // Skip empty nodes
//     if (start == -1 && end == -1)
//     {
//         return;
//     }

//     Vector topLeftFront = curNode.topLeftFront;
//     Vector botRightBack = curNode.botRightBack;

//     // Calculate center of mass for current node
//     ComputeCenterMass(curNode, bodies, totalMass, centerMass, start, end);

//     // Check if we've reached the recursion limit or have only one body
//     if (nodeIndex >= leafLimit || start == end)
//     {
//         // Just copy bodies and return
//         for (int i = start + tx; i <= end; i += blockDim.x)
//         {
//             buffer[i] = bodies[i];
//         }
//         return;
//     }

//     // Step 1: Count bodies in each octant
//     CountBodies(bodies, topLeftFront, botRightBack, count, start, end, nBodies);

//     // Step 2: Calculate base offsets for each octant
//     ComputeOffset(count, start);

//     // Copy offsets to local arrays for grouping
//     if (tx < 8)
//     {
//         baseOffset[tx] = count[tx + 8];  // Save original offset
//         workOffset[tx] = baseOffset[tx]; // Initialize working copy for atomic ops
//     }
//     __syncthreads();

//     // Step 3: Group bodies by octant
//     GroupBodies(bodies, buffer, topLeftFront, botRightBack, workOffset, start, end, nBodies);

//     // Step 4: Assign ranges to child nodes (only thread 0)
//     if (tx == 0)
//     {
//         // Mark current node as non-leaf
//         curNode.isLeaf = false;

//         // For each octant, setup child node
//         for (int i = 0; i < 8; i++)
//         {
//             int childIdx = nodeIndex * 8 + (i + 1);
//             if (childIdx < nNodes)
//             {
//                 Node &childNode = node[childIdx];

//                 // Set child node bounding box
//                 UpdateChildBound(topLeftFront, botRightBack, childNode, i + 1);

//                 // Assign body range if this octant has bodies
//                 if (count[i] > 0)
//                 {
//                     childNode.start = baseOffset[i];
//                     childNode.end = baseOffset[i] + count[i] - 1;
//                 }
//                 else
//                 {
//                     childNode.start = -1;
//                     childNode.end = -1;
//                 }
//             }
//         }

//         // Recursively process child nodes (launch 8 blocks)
//         ConstructOctTreeKernel<<<8, BLOCK_SIZE>>>(node, buffer, bodies, nodeIndex * 8 + 1, nNodes, nBodies, leafLimit);
//     }
// }

#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"

// Constantes del dispositivo para la cola de trabajo
constexpr int MAX_JOBS = 100000; // Ajustar según la profundidad máxima esperada del árbol
constexpr int QUEUE_EMPTY = -1;

/**
 * @brief Obtiene el octante para un punto 3D dentro del cuadro delimitador de un nodo
 *
 * Determina cuál de los ocho octantes contiene el punto dado.
 * Los octantes están numerados del 1 al 8 como sigue:
 * 1: arriba-izquierda-frente
 * 2: arriba-derecha-frente
 * 3: abajo-izquierda-frente
 * 4: abajo-derecha-frente
 * 5: arriba-izquierda-atrás
 * 6: arriba-derecha-atrás
 * 7: abajo-izquierda-atrás
 * 8: abajo-derecha-atrás
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
                octant = 1; // arriba-izquierda-frente
            else
                octant = 5; // arriba-izquierda-atrás
        }
        else
        {
            if (z <= midZ)
                octant = 3; // abajo-izquierda-frente
            else
                octant = 7; // abajo-izquierda-atrás
        }
    }
    else
    {
        if (y >= midY)
        {
            if (z <= midZ)
                octant = 2; // arriba-derecha-frente
            else
                octant = 6; // arriba-derecha-atrás
        }
        else
        {
            if (z <= midZ)
                octant = 4; // abajo-derecha-frente
            else
                octant = 8; // abajo-derecha-atrás
        }
    }
    return octant;
}

/**
 * @brief Actualiza el cuadro delimitador de un nodo hijo basado en su octante
 *
 * Establece los límites de un nodo hijo basado en los límites del nodo padre y el número de octante.
 */
__device__ void UpdateChildBound(Vector &tlf, Vector &brb, Node &childNode, int octant)
{
    double midX = (tlf.x + brb.x) / 2;
    double midY = (tlf.y + brb.y) / 2;
    double midZ = (tlf.z + brb.z) / 2;

    switch (octant)
    {
    case 1: // arriba-izquierda-frente
        childNode.topLeftFront = tlf;
        childNode.botRightBack = Vector(midX, midY, midZ);
        break;
    case 2: // arriba-derecha-frente
        childNode.topLeftFront = Vector(midX, tlf.y, tlf.z);
        childNode.botRightBack = Vector(brb.x, midY, midZ);
        break;
    case 3: // abajo-izquierda-frente
        childNode.topLeftFront = Vector(tlf.x, midY, tlf.z);
        childNode.botRightBack = Vector(midX, brb.y, midZ);
        break;
    case 4: // abajo-derecha-frente
        childNode.topLeftFront = Vector(midX, midY, tlf.z);
        childNode.botRightBack = Vector(brb.x, brb.y, midZ);
        break;
    case 5: // arriba-izquierda-atrás
        childNode.topLeftFront = Vector(tlf.x, tlf.y, midZ);
        childNode.botRightBack = Vector(midX, midY, brb.z);
        break;
    case 6: // arriba-derecha-atrás
        childNode.topLeftFront = Vector(midX, tlf.y, midZ);
        childNode.botRightBack = Vector(brb.x, midY, brb.z);
        break;
    case 7: // abajo-izquierda-atrás
        childNode.topLeftFront = Vector(tlf.x, midY, midZ);
        childNode.botRightBack = Vector(midX, brb.y, brb.z);
        break;
    case 8: // abajo-derecha-atrás
        childNode.topLeftFront = Vector(midX, midY, midZ);
        childNode.botRightBack = brb;
        break;
    }
}

/**
 * @brief Reducción a nivel de warp para cálculos de masa y centro de masa
 *
 * Optimizado para reducción a nivel de warp (32 hilos) sin conflictos de bancos de memoria compartida.
 */
__device__ void warpReduce(volatile double *totalMass, volatile double3 *centerMass, int tx)
{
    if (tx < 32)
    { // Solo el primer warp
        // Reducción de 32 a 16
        totalMass[tx] += totalMass[tx + 32];
        centerMass[tx].x += centerMass[tx + 32].x;
        centerMass[tx].y += centerMass[tx + 32].y;
        centerMass[tx].z += centerMass[tx + 32].z;

        // Reducción de 16 a 8
        totalMass[tx] += totalMass[tx + 16];
        centerMass[tx].x += centerMass[tx + 16].x;
        centerMass[tx].y += centerMass[tx + 16].y;
        centerMass[tx].z += centerMass[tx + 16].z;

        // Reducción de 8 a 4
        totalMass[tx] += totalMass[tx + 8];
        centerMass[tx].x += centerMass[tx + 8].x;
        centerMass[tx].y += centerMass[tx + 8].y;
        centerMass[tx].z += centerMass[tx + 8].z;

        // Reducción de 4 a 2
        totalMass[tx] += totalMass[tx + 4];
        centerMass[tx].x += centerMass[tx + 4].x;
        centerMass[tx].y += centerMass[tx + 4].y;
        centerMass[tx].z += centerMass[tx + 4].z;

        // Reducción de 2 a 1
        totalMass[tx] += totalMass[tx + 2];
        centerMass[tx].x += centerMass[tx + 2].x;
        centerMass[tx].y += centerMass[tx + 2].y;
        centerMass[tx].z += centerMass[tx + 2].z;

        // Último paso
        totalMass[tx] += totalMass[tx + 1];
        centerMass[tx].x += centerMass[tx + 1].x;
        centerMass[tx].y += centerMass[tx + 1].y;
        centerMass[tx].z += centerMass[tx + 1].z;
    }
}

/**
 * @brief Calcula el centro de masa para un nodo
 *
 * Optimizado para reducir la divergencia de warps y mejorar la coalescencia de memoria.
 */
__device__ void ComputeCenterMass(Node &curNode, Body *bodies, double *totalMass, double3 *centerMass, int start, int end)
{
    int tx = threadIdx.x;
    int total = end - start + 1;
    int bodyPerThread = (total + blockDim.x - 1) / blockDim.x;
    int startIdx = start + tx * bodyPerThread;
    int endIdx = min(startIdx + bodyPerThread - 1, end);

    double M = 0.0;
    double3 R = make_double3(0.0, 0.0, 0.0);

    // Cada hilo procesa varios cuerpos secuencialmente para mejor coalescencia
    for (int i = startIdx; i <= endIdx; i++)
    {
        if (i <= end)
        {
            Body body = bodies[i];
            M += body.mass;
            R.x += body.mass * body.position.x;
            R.y += body.mass * body.position.y;
            R.z += body.mass * body.position.z;
        }
    }

    // Almacenar resultados parciales del hilo
    totalMass[tx] = M;
    centerMass[tx] = R;

    // Reducción a nivel de bloque
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

    // Reducción a nivel de warp para los 32 hilos finales
    __syncthreads();
    if (tx < 32)
    {
        warpReduce(totalMass, centerMass, tx);
    }

    __syncthreads();

    // Actualizar nodo con resultados finales
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
 * @brief Contar cuerpos en cada octante
 *
 * Optimizado para mejorar el patrón de acceso a memoria.
 */
__device__ void CountBodies(Body *bodies, Vector topLeftFront, Vector botRightBack, int *count, int start, int end)
{
    int tx = threadIdx.x;

    // Inicializar contadores
    if (tx < 8)
    {
        count[tx] = 0;
    }
    __syncthreads();

    // Cada hilo procesa varios cuerpos secuencialmente para mejor rendimiento
    int step = blockDim.x;
    for (int i = start + tx; i <= end; i += step)
    {
        Body body = bodies[i];
        int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);
        atomicAdd(&count[oct - 1], 1);
    }
    __syncthreads();
}

/**
 * @brief Calcular desplazamientos para cada octante
 *
 * Calcula índices de inicio para cuerpos en cada octante basado en conteos.
 */
__device__ void ComputeOffset(int *count, int *baseOffset, int start)
{
    int tx = threadIdx.x;

    if (tx < 8)
    {
        int offset = start;
        for (int i = 0; i < tx; ++i)
        {
            offset += count[i];
        }
        baseOffset[tx] = offset;
    }
    __syncthreads();
}

/**
 * @brief Agrupar cuerpos por octante
 *
 * Reordena cuerpos en el buffer según su octante con patrones optimizados de acceso a memoria.
 */
__device__ void GroupBodies(Body *bodies, Body *buffer, Vector topLeftFront, Vector botRightBack, int *workOffset, int start, int end)
{
    int tx = threadIdx.x;
    int step = blockDim.x;

    // Cada hilo procesa varios cuerpos para mejorar el rendimiento
    for (int i = start + tx; i <= end; i += step)
    {
        Body body = bodies[i];
        int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);

        // Obtener índice de destino con incremento atómico
        int dest = atomicAdd(&workOffset[oct - 1], 1);

        // Copiar cuerpo a su nueva posición
        buffer[dest] = body;
    }
    __syncthreads();
}

/**
 * @brief Prepara la cola de trabajo de nodos
 *
 * Añade los nodos hijos a la cola de trabajo global.
 */
__device__ void EnqueueChildNodes(int nodeIndex, int *jobQueue, int *jobCount, int maxJobs, int nNodes)
{
    // Calcular índices de nodos hijos
    for (int i = 0; i < 8; i++)
    {
        int childIndex = nodeIndex * 8 + i + 1;
        if (childIndex < nNodes)
        {
            // Añadir el nodo hijo a la cola de trabajo con incremento atómico
            int jobId = atomicAdd(jobCount, 1);
            if (jobId < maxJobs)
            {
                jobQueue[jobId] = childIndex;
            }
        }
    }
}

/**
 * @brief Kernel optimizado para la construcción de árboles octales
 *
 * Implementa un enfoque persistente basado en cola para eliminar recursividad.
 * Los bloques de CUDA obtienen trabajo de una cola global.
 */
__global__ void OptimizedConstructOctTreeKernel(
    Node *nodes, Body *bodies, Body *buffer,
    int *jobQueue, int *jobCount, int initJobCount,
    int nNodes, int nBodies, int leafLimit)
{
    __shared__ int count[8];
    __shared__ double totalMass[BLOCK_SIZE];
    __shared__ double3 centerMass[BLOCK_SIZE];
    __shared__ int baseOffset[8];
    __shared__ int workOffset[8];
    __shared__ int nodeIndex;
    __shared__ bool isValidNode;

    int tx = threadIdx.x;

    // Inicializar el contador de trabajos si es el primer bloque y primer hilo
    if (blockIdx.x == 0 && tx == 0)
    {
        *jobCount = initJobCount;
    }

    __syncthreads();

    // Los bloques obtienen trabajo continuamente de la cola
    while (true)
    {
        // El primer hilo de cada bloque obtiene un trabajo de la cola
        if (tx == 0)
        {
            nodeIndex = QUEUE_EMPTY;
            isValidNode = false;

            int currentJobCount = *jobCount;
            if (currentJobCount > 0)
            {
                // Obtener trabajo con decremento atómico
                int jobId = atomicAdd(jobCount, -1) - 1;
                if (jobId >= 0 && jobId < MAX_JOBS)
                {
                    nodeIndex = jobQueue[jobId];
                    isValidNode = (nodeIndex < nNodes);
                }
            }
        }

        __syncthreads();

        // Si no hay más trabajo, salir
        if (nodeIndex == QUEUE_EMPTY || !isValidNode)
        {
            return;
        }

        // Obtener el nodo actual
        Node curNode = nodes[nodeIndex];
        int start = curNode.start;
        int end = curNode.end;

        // Saltar nodos vacíos
        if (start == -1 && end == -1)
        {
            continue;
        }

        Vector topLeftFront = curNode.topLeftFront;
        Vector botRightBack = curNode.botRightBack;

        // Calcular centro de masa para el nodo actual
        ComputeCenterMass(curNode, bodies, totalMass, centerMass, start, end);

        // Verificar si hemos alcanzado el límite de recursión o tenemos solo un cuerpo
        if (nodeIndex >= leafLimit || start == end)
        {
            // Copiar cuerpos con patrones coalescentes
            int step = blockDim.x;
            for (int i = start + tx; i <= end; i += step)
            {
                if (i <= end)
                {
                    buffer[i] = bodies[i];
                }
            }

            // No encolamos más trabajo para nodos hoja
            continue;
        }

        // Paso 1: Contar cuerpos en cada octante
        CountBodies(bodies, topLeftFront, botRightBack, count, start, end);

        // Paso 2: Calcular desplazamientos base para cada octante
        ComputeOffset(count, baseOffset, start);

        // Copiar desplazamientos a arrays locales para agrupación
        if (tx < 8)
        {
            workOffset[tx] = baseOffset[tx];
        }
        __syncthreads();

        // Paso 3: Agrupar cuerpos por octante
        GroupBodies(bodies, buffer, topLeftFront, botRightBack, workOffset, start, end);

        // Paso 4: Asignar rangos a nodos hijos (solo el hilo 0)
        if (tx == 0)
        {
            // Marcar nodo actual como no-hoja
            nodes[nodeIndex].isLeaf = false;

            // Para cada octante, configurar el nodo hijo
            for (int i = 0; i < 8; i++)
            {
                int childIdx = nodeIndex * 8 + (i + 1);
                if (childIdx < nNodes)
                {
                    Node &childNode = nodes[childIdx];

                    // Establecer cuadro delimitador del nodo hijo
                    UpdateChildBound(topLeftFront, botRightBack, childNode, i + 1);

                    // Asignar rango de cuerpos si este octante tiene cuerpos
                    if (count[i] > 0)
                    {
                        childNode.start = baseOffset[i];
                        childNode.end = baseOffset[i] + count[i] - 1;

                        // Encolar nodo hijo para procesamiento
                        int jobId = atomicAdd(jobCount, 1);
                        if (jobId < MAX_JOBS)
                        {
                            jobQueue[jobId] = childIdx;
                        }
                    }
                    else
                    {
                        childNode.start = -1;
                        childNode.end = -1;
                    }
                }
            }
        }

        __syncthreads();
    }
}

/**
 * @brief Inicializa la cola de trabajo con el nodo raíz
 *
 * Prepara la cola para el enfoque persistente.
 */
__global__ void InitJobQueueKernel(int *jobQueue, int *jobCount)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        jobQueue[0] = 0; // Nodo raíz
        *jobCount = 1;   // Un trabajo inicial
    }
}

/**
 * @brief Función host para construir el árbol octal
 *
 * Coordina los kernels para la construcción del árbol.
 *
 * @param nodes Arreglo de nodos del árbol octal
 * @param bodies Arreglo de cuerpos
 * @param tempBodies Arreglo temporal para reordenamiento de cuerpos
 * @param nNodes Número total de nodos
 * @param nBodies Número total de cuerpos
 * @param leafLimit Límite para nodos hoja (controla la profundidad de recursión)
 */
extern "C" void BuildOptimizedOctTree(
    Node *d_nodes, Body *d_bodies, Body *d_tempBodies,
    int nNodes, int nBodies, int leafLimit)
{
    // Asignar memoria para la cola de trabajo
    int *d_jobQueue;
    int *d_jobCount;
    cudaMalloc(&d_jobQueue, MAX_JOBS * sizeof(int));
    cudaMalloc(&d_jobCount, sizeof(int));

    // Inicializar la cola con el nodo raíz
    InitJobQueueKernel<<<1, 1>>>(d_jobQueue, d_jobCount);

    // Calcular configuración de lanzamiento
    int blockSize = BLOCK_SIZE;

    // Obtener número de multiprocesadores
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    int numSMs = props.multiProcessorCount;

    // Lanzar aproximadamente 2 bloques por SM para buena ocupación
    int numBlocks = numSMs * 2;

    // Lanzar kernel optimizado para construir el árbol
    OptimizedConstructOctTreeKernel<<<numBlocks, blockSize>>>(
        d_nodes, d_bodies, d_tempBodies,
        d_jobQueue, d_jobCount, 1,
        nNodes, nBodies, leafLimit);

    // Limpiar recursos
    cudaFree(d_jobQueue);
    cudaFree(d_jobCount);
}