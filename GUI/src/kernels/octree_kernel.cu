#include "../../include/common/types.cuh"
#include "../../include/common/constants.cuh"
#include <stdio.h>

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
__device__ void ComputeCenterMass(Node &curNode, Body *bodies, int *orderedIndices, bool useSFC,
                                  double *totalMass, double3 *centerMass, int start, int end)
{
    int tx = threadIdx.x;
    int total = end - start + 1;
    int bodyPerThread = (total + blockDim.x - 1) / blockDim.x;
    int startIdx = start + tx * bodyPerThread;
    int endIdx = min(startIdx + bodyPerThread - 1, end);

    double M = 0.0;
    double3 R = make_double3(0.0, 0.0, 0.0);

    // Si estamos usando índices ordenados por SFC
    if (useSFC && orderedIndices != nullptr)
    {
        // Procesamos los cuerpos usando indirección a través de los índices ordenados
        for (int i = startIdx; i <= endIdx; i++)
        {
            if (i <= end)
            {
                int bodyIndex = orderedIndices[i]; // Acceso indirecto a través del índice SFC
                Body body = bodies[bodyIndex];
                M += body.mass;
                R.x += body.mass * body.position.x;
                R.y += body.mass * body.position.y;
                R.z += body.mass * body.position.z;
            }
        }
    }
    else
    {
        // Procesamiento normal sin SFC
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
    }

    // Almacenar resultados parciales del hilo
    totalMass[tx] = M;
    centerMass[tx] = R;

    // Reducción a nivel de bloque (sin cambios)
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
__device__ void CountBodies(Body *bodies, int *orderedIndices, bool useSFC,
                            Vector topLeftFront, Vector botRightBack,
                            int *count, int start, int end)
{
    int tx = threadIdx.x;

    // Inicializar contadores
    if (tx < 8)
    {
        count[tx] = 0;
    }
    __syncthreads();

    // Cada hilo procesa varios cuerpos
    const int step = blockDim.x;

    if (useSFC && orderedIndices != nullptr)
    {
        // Usando índices SFC
        for (int i = start + tx; i <= end; i += step)
        {
            int bodyIndex = orderedIndices[i]; // Índice ordenado
            Body body = bodies[bodyIndex];
            int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);
            atomicAdd(&count[oct - 1], 1);
        }
    }
    else
    {
        // Sin usar índices SFC
        for (int i = start + tx; i <= end; i += step)
        {
            Body body = bodies[i];
            int oct = getOctant(topLeftFront, botRightBack, body.position.x, body.position.y, body.position.z);
            atomicAdd(&count[oct - 1], 1);
        }
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
 * @brief CUDA kernel for octree construction with octant ordering
 *
 * This kernel is similar to OptimizedConstructOctTreeKernel but has additional
 * support for octant ordering.
 */
__global__ void OptimizedConstructOctTreeKernelWithOctantOrder(
    Node *nodes, Body *bodies, Body *buffer,
    int *orderedIndices, bool useSFC,
    int *octantIndices, bool useOctantOrder,
    int *jobQueue, int *jobCount, int initJobCount,
    int nNodes, int nBodies, int leafLimit)
{
    // Block shared memory - properly sized and bounded
    __shared__ int count[8];      // Bodies per octant
    __shared__ int baseOffset[8]; // Starting offset for each octant
    __shared__ int workOffset[8]; // Current work offset
    __shared__ int nodeIndex;     // Current node index
    __shared__ bool isValidNode;  // Node validity flag

    // Dynamically sized shared memory for mass calculation
    // Use fixed size to avoid issues with block sizes
    extern __shared__ double sharedMem[];

    // Split shared memory for total mass and center of mass
    double *totalMass = sharedMem;
    double3 *centerMass = (double3 *)&sharedMem[blockDim.x];

    int tx = threadIdx.x;

    // Initialize job count for the first block
    if (blockIdx.x == 0 && tx == 0)
    {
        *jobCount = initJobCount;
    }

    __syncthreads();

    // Initialize totalMass and centerMass for this thread
    if (tx < blockDim.x)
    {
        totalMass[tx] = 0.0;
        centerMass[tx] = make_double3(0.0, 0.0, 0.0);
    }

    // Blocks continuously fetch work from the queue
    while (true)
    {
        // First thread in each block gets a job from the queue
        if (tx == 0)
        {
            nodeIndex = -1; // No work by default
            isValidNode = false;

            int currentJobCount = *jobCount;
            if (currentJobCount > 0)
            {
                // Get job with atomic decrement
                int jobId = atomicAdd(jobCount, -1) - 1;
                if (jobId >= 0)
                {
                    nodeIndex = jobQueue[jobId];
                    isValidNode = (nodeIndex < nNodes);
                }
            }
        }

        __syncthreads();

        // Exit if no more work
        if (nodeIndex < 0 || !isValidNode)
        {
            return;
        }

        // Get the current node
        Node curNode = nodes[nodeIndex];

        // Skip empty nodes
        if (curNode.start == -1 && curNode.end == -1)
        {
            continue;
        }

        Vector topLeftFront = curNode.topLeftFront;
        Vector botRightBack = curNode.botRightBack;
        int start = curNode.start;
        int end = curNode.end;

        // Compute center of mass for this node
        // Reset shared memory for calculation
        if (tx < blockDim.x)
        {
            totalMass[tx] = 0.0;
            centerMass[tx] = make_double3(0.0, 0.0, 0.0);
        }

        __syncthreads();

        // Parallel center of mass calculation with improved memory access
        if (useSFC && orderedIndices)
        {
            // Process a range of bodies per thread
            int numBodiesInNode = end - start + 1;
            int bodiesPerThread = (numBodiesInNode + blockDim.x - 1) / blockDim.x;
            int myStart = start + tx * bodiesPerThread;
            int myEnd = min(myStart + bodiesPerThread - 1, end);

            double myTotalMass = 0.0;
            double3 myCenterMass = make_double3(0.0, 0.0, 0.0);

            for (int idx = myStart; idx <= myEnd; idx++)
            {
                if (idx <= end)
                { // Ensure we don't go out of bounds
                    int bodyIndex = orderedIndices[idx];
                    Body body = bodies[bodyIndex];

                    myTotalMass += body.mass;
                    myCenterMass.x += body.mass * body.position.x;
                    myCenterMass.y += body.mass * body.position.y;
                    myCenterMass.z += body.mass * body.position.z;
                }
            }

            // Store thread results - make sure we're in bounds
            if (tx < blockDim.x)
            {
                totalMass[tx] = myTotalMass;
                centerMass[tx] = myCenterMass;
            }
        }
        else
        {
            // Without SFC ordering
            int numBodiesInNode = end - start + 1;
            int bodiesPerThread = (numBodiesInNode + blockDim.x - 1) / blockDim.x;
            int myStart = start + tx * bodiesPerThread;
            int myEnd = min(myStart + bodiesPerThread - 1, end);

            double myTotalMass = 0.0;
            double3 myCenterMass = make_double3(0.0, 0.0, 0.0);

            for (int idx = myStart; idx <= myEnd; idx++)
            {
                if (idx <= end)
                { // Ensure we don't go out of bounds
                    Body body = bodies[idx];

                    myTotalMass += body.mass;
                    myCenterMass.x += body.mass * body.position.x;
                    myCenterMass.y += body.mass * body.position.y;
                    myCenterMass.z += body.mass * body.position.z;
                }
            }

            // Store thread results - make sure we're in bounds
            if (tx < blockDim.x)
            {
                totalMass[tx] = myTotalMass;
                centerMass[tx] = myCenterMass;
            }
        }

        // Parallel reduction - only use threads within blockDim.x
        for (uint s = blockDim.x / 2; s > 32; s >>= 1)
        {
            __syncthreads();
            if (tx < s && tx + s < blockDim.x)
            {
                totalMass[tx] += totalMass[tx + s];
                centerMass[tx].x += centerMass[tx + s].x;
                centerMass[tx].y += centerMass[tx + s].y;
                centerMass[tx].z += centerMass[tx + s].z;
            }
        }

        __syncthreads();

        // Warp-level reduction (no sync needed within a warp) - prevent out-of-bounds access
        if (tx < 32)
        {
            // Only perform reduction if both indices are in range
            if (tx + 32 < blockDim.x)
            {
                totalMass[tx] += totalMass[tx + 32];
                centerMass[tx].x += centerMass[tx + 32].x;
                centerMass[tx].y += centerMass[tx + 32].y;
                centerMass[tx].z += centerMass[tx + 32].z;
            }

            // 16 -> 8
            if (tx + 16 < blockDim.x && tx < 16)
            {
                totalMass[tx] += totalMass[tx + 16];
                centerMass[tx].x += centerMass[tx + 16].x;
                centerMass[tx].y += centerMass[tx + 16].y;
                centerMass[tx].z += centerMass[tx + 16].z;
            }

            // 8 -> 4
            if (tx + 8 < blockDim.x && tx < 8)
            {
                totalMass[tx] += totalMass[tx + 8];
                centerMass[tx].x += centerMass[tx + 8].x;
                centerMass[tx].y += centerMass[tx + 8].y;
                centerMass[tx].z += centerMass[tx + 8].z;
            }

            // 4 -> 2
            if (tx + 4 < blockDim.x && tx < 4)
            {
                totalMass[tx] += totalMass[tx + 4];
                centerMass[tx].x += centerMass[tx + 4].x;
                centerMass[tx].y += centerMass[tx + 4].y;
                centerMass[tx].z += centerMass[tx + 4].z;
            }

            // 2 -> 1
            if (tx + 2 < blockDim.x && tx < 2)
            {
                totalMass[tx] += totalMass[tx + 2];
                centerMass[tx].x += centerMass[tx + 2].x;
                centerMass[tx].y += centerMass[tx + 2].y;
                centerMass[tx].z += centerMass[tx + 2].z;
            }

            // 1 -> 0
            if (tx + 1 < blockDim.x && tx < 1)
            {
                totalMass[tx] += totalMass[tx + 1];
                centerMass[tx].x += centerMass[tx + 1].x;
                centerMass[tx].y += centerMass[tx + 1].y;
                centerMass[tx].z += centerMass[tx + 1].z;
            }
        }

        __syncthreads();

        // Update node's center of mass and total mass
        if (tx == 0)
        {
            double nodeTotalMass = totalMass[0];
            double3 nodeCenterMass = centerMass[0];

            // Normalize center of mass
            if (nodeTotalMass > 0.0)
            {
                nodeCenterMass.x /= nodeTotalMass;
                nodeCenterMass.y /= nodeTotalMass;
                nodeCenterMass.z /= nodeTotalMass;
            }

            // Update node data
            nodes[nodeIndex].totalMass = nodeTotalMass;
            nodes[nodeIndex].centerMass = Vector(nodeCenterMass.x, nodeCenterMass.y, nodeCenterMass.z);
        }

        // Check if we're at a leaf node or reached recursion limit
        if (nodeIndex >= leafLimit || end - start <= 8 || end == start)
        {
            // Copy bodies for leaf nodes (vectorized copy)
            for (int i = start + tx; i <= end; i += blockDim.x)
            {
                if (i <= end)
                {
                    if (useSFC && orderedIndices)
                    {
                        int bodyIndex = orderedIndices[i];
                        buffer[i] = bodies[bodyIndex];
                    }
                    else
                    {
                        buffer[i] = bodies[i];
                    }
                }
            }

            continue; // Done with this node
        }

        // Initialize octant counters
        if (tx < 8)
        {
            count[tx] = 0;
        }

        __syncthreads();

        // Count bodies in each octant
        if (useSFC && orderedIndices)
        {
            // Process bodies in parallel with SFC ordering
            for (int i = start + tx; i <= end; i += blockDim.x)
            {
                if (i <= end)
                {
                    int bodyIndex = orderedIndices[i];
                    Body body = bodies[bodyIndex];

                    // Compute octant
                    int octant = getOctant(topLeftFront, botRightBack,
                                           body.position.x, body.position.y, body.position.z);

                    // Ensure valid octant (1-8)
                    if (octant >= 1 && octant <= 8)
                    {
                        // Increment counter atomically
                        atomicAdd(&count[octant - 1], 1);
                    }
                }
            }
        }
        else
        {
            // Process bodies in parallel without SFC
            for (int i = start + tx; i <= end; i += blockDim.x)
            {
                if (i <= end)
                {
                    Body body = bodies[i];

                    // Compute octant
                    int octant = getOctant(topLeftFront, botRightBack,
                                           body.position.x, body.position.y, body.position.z);

                    // Ensure valid octant (1-8)
                    if (octant >= 1 && octant <= 8)
                    {
                        // Increment counter atomically
                        atomicAdd(&count[octant - 1], 1);
                    }
                }
            }
        }

        __syncthreads();

        // Compute base offsets for each octant
        if (tx < 8)
        {
            int offset = start;
            for (int i = 0; i < tx; i++)
            {
                offset += count[i];
            }
            baseOffset[tx] = offset;
            workOffset[tx] = offset; // Initialize work offset
        }

        __syncthreads();

        // Distribute bodies into octants
        if (useSFC && orderedIndices)
        {
            // Group bodies using SFC ordering
            for (int i = start + tx; i <= end; i += blockDim.x)
            {
                if (i <= end)
                {
                    int bodyIndex = orderedIndices[i];
                    Body body = bodies[bodyIndex];

                    // Compute octant
                    int octant = getOctant(topLeftFront, botRightBack,
                                           body.position.x, body.position.y, body.position.z);

                    // Ensure valid octant
                    if (octant >= 1 && octant <= 8)
                    {
                        // Get destination index
                        int destIdx = atomicAdd(&workOffset[octant - 1], 1);

                        // Make sure destIdx is valid
                        if (destIdx >= start && destIdx <= end)
                        {
                            // Store body
                            buffer[destIdx] = body;
                        }
                    }
                }
            }
        }
        else
        {
            // Group bodies without SFC ordering
            for (int i = start + tx; i <= end; i += blockDim.x)
            {
                if (i <= end)
                {
                    Body body = bodies[i];

                    // Compute octant
                    int octant = getOctant(topLeftFront, botRightBack,
                                           body.position.x, body.position.y, body.position.z);

                    // Ensure valid octant
                    if (octant >= 1 && octant <= 8)
                    {
                        // Get destination index
                        int destIdx = atomicAdd(&workOffset[octant - 1], 1);

                        // Make sure destIdx is valid
                        if (destIdx >= start && destIdx <= end)
                        {
                            // Store body
                            buffer[destIdx] = body;
                        }
                    }
                }
            }
        }

        __syncthreads();

        // Thread 0 creates child nodes and adds them to the work queue
        if (tx == 0)
        {
            // Mark current node as internal
            nodes[nodeIndex].isLeaf = false;

            // Create child nodes and add them to the work queue
            for (int octant = 0; octant < 8; octant++)
            {
                if (count[octant] > 0)
                {
                    // Compute child index
                    int childIdx;

                    if (useOctantOrder && octantIndices)
                    {
                        // Use octant ordering
                        childIdx = octantIndices[nodeIndex * 8 + octant + 1];
                    }
                    else
                    {
                        // Standard ordering
                        childIdx = nodeIndex * 8 + octant + 1;
                    }

                    // Make sure index is valid
                    if (childIdx < nNodes)
                    {
                        // Set child node properties
                        Node &childNode = nodes[childIdx];

                        // Update bounding box
                        UpdateChildBound(topLeftFront, botRightBack, childNode, octant + 1);

                        // Set body range
                        childNode.start = baseOffset[octant];
                        childNode.end = baseOffset[octant] + count[octant] - 1;

                        // Add to work queue (if not a small leaf)
                        if (childNode.end - childNode.start > 8)
                        {
                            int jobId = atomicAdd(jobCount, 1);
                            if (jobId < MAX_JOBS)
                            {
                                jobQueue[jobId] = childIdx;
                            }
                        }
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
    int *orderedIndices, bool useSFC,
    int *octantIndices, bool useOctantOrder,
    int nNodes, int nBodies, int leafLimit)
{
    // Allocate memory for the work queue
    int *d_jobQueue;
    int *d_jobCount;
    cudaMalloc(&d_jobQueue, MAX_JOBS * sizeof(int));
    cudaMalloc(&d_jobCount, sizeof(int));

    // Initialize the queue with the root node
    int initJobCount = 1;
    cudaMemcpy(d_jobCount, &initJobCount, sizeof(int), cudaMemcpyHostToDevice);
    int rootIndex = 0;
    cudaMemcpy(d_jobQueue, &rootIndex, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate launch configuration
    int blockSize = 256; // Use smaller block size for better occupancy

    // Compute shared memory size for double (totalMass) and double3 (centerMass)
    size_t sharedMemSize = blockSize * sizeof(double) + blockSize * sizeof(double3);

    // Get number of multiprocessors
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    int numSMs = props.multiProcessorCount;

    // Launch approximately 2 blocks per SM for good occupancy
    int numBlocks = numSMs * 2;

    // Launch the fixed kernel with proper shared memory size
    OptimizedConstructOctTreeKernelWithOctantOrder<<<numBlocks, blockSize, sharedMemSize>>>(
        d_nodes, d_bodies, d_tempBodies,
        orderedIndices, useSFC,
        octantIndices, useOctantOrder,
        d_jobQueue, d_jobCount, 1,
        nNodes, nBodies, leafLimit);

    // Synchronize to ensure completion
    cudaDeviceSynchronize();

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // Free resources
    cudaFree(d_jobQueue);
    cudaFree(d_jobCount);
}
