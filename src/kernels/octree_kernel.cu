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
 * @brief Kernel optimizado para la construcción de árboles octales
 *
 * Implementa un enfoque persistente basado en cola para eliminar recursividad.
 * Los bloques de CUDA obtienen trabajo de una cola global.
 */
__global__ void OptimizedConstructOctTreeKernel(
    Node *nodes, Body *bodies, Body *buffer,
    int *orderedIndices, bool useSFC,
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
        ComputeCenterMass(curNode, bodies, orderedIndices, useSFC,
                          totalMass, centerMass, start, end);

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
        CountBodies(bodies, orderedIndices, useSFC, topLeftFront, botRightBack, count, start, end);

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
    __shared__ int count[8];
    __shared__ double totalMass[BLOCK_SIZE];
    __shared__ double3 centerMass[BLOCK_SIZE];
    __shared__ int baseOffset[8];
    __shared__ int workOffset[8];
    __shared__ int nodeIndex;
    __shared__ bool isValidNode;

    int tx = threadIdx.x;

    // Initialize job counter for the first block
    if (blockIdx.x == 0 && tx == 0)
    {
        *jobCount = initJobCount;
    }

    __syncthreads();

    // Blocks continuously fetch work from the queue
    while (true)
    {
        // First thread in each block gets a job from the queue
        if (tx == 0)
        {
            nodeIndex = QUEUE_EMPTY;
            isValidNode = false;

            int currentJobCount = *jobCount;
            if (currentJobCount > 0)
            {
                // Get job with atomic decrement
                int jobId = atomicAdd(jobCount, -1) - 1;
                if (jobId >= 0 && jobId < MAX_JOBS)
                {
                    nodeIndex = jobQueue[jobId];
                    isValidNode = (nodeIndex < nNodes);
                }
            }
        }

        __syncthreads();

        // Exit if no more work
        if (nodeIndex == QUEUE_EMPTY || !isValidNode)
        {
            return;
        }

        // Get the current node
        Node curNode = nodes[nodeIndex];
        int start = curNode.start;
        int end = curNode.end;

        // Skip empty nodes
        if (start == -1 && end == -1)
        {
            continue;
        }

        Vector topLeftFront = curNode.topLeftFront;
        Vector botRightBack = curNode.botRightBack;

        // Calculate center of mass for the current node
        ComputeCenterMass(curNode, bodies, orderedIndices, useSFC,
                          totalMass, centerMass, start, end);

        // Check if we've reached the recursion limit or have only one body
        if (nodeIndex >= leafLimit || start == end)
        {
            // Copy bodies with coalesced patterns
            int step = blockDim.x;
            for (int i = start + tx; i <= end; i += step)
            {
                if (i <= end)
                {
                    buffer[i] = bodies[i];
                }
            }

            // Don't enqueue more work for leaf nodes
            continue;
        }

        // Step 1: Count bodies in each octant
        CountBodies(bodies, orderedIndices, useSFC, topLeftFront, botRightBack, count, start, end);

        // Step 2: Compute base offsets for each octant
        ComputeOffset(count, baseOffset, start);

        // Copy offsets to working arrays for grouping
        if (tx < 8)
        {
            workOffset[tx] = baseOffset[tx];
        }
        __syncthreads();

        // Step 3: Group bodies by octant
        GroupBodies(bodies, buffer, topLeftFront, botRightBack, workOffset, start, end);

        // Step 4: Assign ranges to child nodes (only thread 0)
        if (tx == 0)
        {
            // Mark current node as non-leaf
            nodes[nodeIndex].isLeaf = false;

            // For each octant, set up the child node
            for (int i = 0; i < 8; i++)
            {
                // Get child node index with octant ordering if enabled
                int childOctant = i;

                // If using octant ordering, remap the child indices
                int childIdx;
                if (useOctantOrder && octantIndices != nullptr)
                {
                    // Calculate base child index
                    childIdx = nodeIndex * 8 + (childOctant + 1);

                    // If valid index, map through octant ordering
                    if (childIdx < nNodes)
                    {
                        // Find the ordered index for this child
                        for (int j = 0; j < 8; j++)
                        {
                            int orderedChildIdx = octantIndices[nodeIndex * 8 + j + 1];
                            if (orderedChildIdx == childIdx)
                            {
                                childOctant = j;
                                break;
                            }
                        }
                    }
                }

                // Calculate the actual child index
                childIdx = nodeIndex * 8 + (childOctant + 1);

                if (childIdx < nNodes)
                {
                    Node &childNode = nodes[childIdx];

                    // Set child node's bounding box
                    UpdateChildBound(topLeftFront, botRightBack, childNode, childOctant + 1);

                    // Assign body range if this octant has bodies
                    if (count[childOctant] > 0)
                    {
                        childNode.start = baseOffset[childOctant];
                        childNode.end = baseOffset[childOctant] + count[childOctant] - 1;

                        // Enqueue child node for processing
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
    InitJobQueueKernel<<<1, 1>>>(d_jobQueue, d_jobCount);

    // Calculate launch configuration
    int blockSize = BLOCK_SIZE;

    // Get number of multiprocessors
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);
    int numSMs = props.multiProcessorCount;

    // Launch approximately 2 blocks per SM for good occupancy
    int numBlocks = numSMs * 2;

    // Launch the appropriate kernel based on ordering mode
    if (useOctantOrder && octantIndices != nullptr)
    {
        // Use the octant ordering kernel
        OptimizedConstructOctTreeKernelWithOctantOrder<<<numBlocks, blockSize>>>(
            d_nodes, d_bodies, d_tempBodies,
            orderedIndices, useSFC,
            octantIndices, true,
            d_jobQueue, d_jobCount, 1,
            nNodes, nBodies, leafLimit);
    }
    else
    {
        // Use the standard kernel with particle ordering only
        OptimizedConstructOctTreeKernel<<<numBlocks, blockSize>>>(
            d_nodes, d_bodies, d_tempBodies,
            orderedIndices, useSFC,
            d_jobQueue, d_jobCount, 1,
            nNodes, nBodies, leafLimit);
    }

    // Free resources
    cudaFree(d_jobQueue);
    cudaFree(d_jobCount);
}